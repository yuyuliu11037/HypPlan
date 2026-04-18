"""Merge LoRA adapter into base model and save to disk."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    print(f"Loading base: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
