"""CoT-SFT for Countdown: fine-tune Llama-3.1-8B-Instruct on trajectories."""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig,
                          get_cosine_schedule_with_warmup)
from peft import (LoraConfig, get_peft_model, TaskType,
                  prepare_model_for_kbit_training)

from src.dataset_cd import CountdownSFTDataset, collate_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_cd.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    model_name = config["model"]["base_model"]
    if rank == 0:
        print(f"[Rank {rank}] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["model"]["lora_r"],
        lora_alpha=config["model"]["lora_alpha"],
        lora_dropout=config["model"]["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    dataset = CountdownSFTDataset(
        tokenizer,
        config["data"]["train_data"],
        max_seq_len=config["data"]["max_seq_len"],
    )

    sampler = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config["training"]["lr"],
                      weight_decay=0.01)

    epochs = config["training"]["epochs"]
    grad_accum = config["training"]["grad_accum"]
    total_steps = (len(dataloader) * epochs) // grad_accum
    warmup_steps = int(total_steps * config["training"]["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    output_dir = config["training"]["output_dir"]
    log_dir = config["training"]["log_dir"]
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "sft_cd_train.jsonl")

    if rank == 0:
        print(f"Training: {len(dataset)} samples, {epochs} epochs, "
              f"batch={config['training']['batch_size']}, "
              f"grad_accum={grad_accum}, total_steps={total_steps}")

    global_step = 0
    model.train()

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        accum_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % 10 == 0:
                    print(f"  step {global_step}/{total_steps} | "
                          f"loss={accum_loss:.4f} | "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                    with open(log_file, "a") as f:
                        f.write(json.dumps({
                            "step": global_step, "epoch": epoch,
                            "loss": round(accum_loss, 4),
                            "lr": scheduler.get_last_lr()[0],
                        }) + "\n")

                epoch_loss += accum_loss
                accum_loss = 0.0

        if rank == 0:
            avg = epoch_loss / max(global_step, 1)
            print(f"Epoch {epoch}: avg_loss={avg:.4f}")

    if rank == 0:
        print(f"Saving LoRA adapter to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Done.")


if __name__ == "__main__":
    main()
