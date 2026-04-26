#!/bin/bash
# Continuation pipeline: BW → GC → eval all 3.
# Assumes PQ already done (or will be launched separately).
set -e
cd /data/yuyu/HypPlan

PYTHON=/usr/bin/python3
TORCHRUN=/data/yuyu/.local/bin/torchrun
mkdir -p logs/dagger_ood results/eval_stage2_indomain

# --- Wait helper: until enough free GPUs from the candidate list ---
free_gpu_list() {
    out=""
    for g in $1; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
        if [ -n "$mem" ] && [ "$mem" -lt 1000 ]; then
            out="$out $g"
        fi
    done
    echo "$out" | xargs
}

wait_for_n_gpus() {
    candidates="$1"
    need=$2
    while true; do
        free=$(free_gpu_list "$candidates")
        n=$(echo $free | wc -w)
        if [ "$n" -ge "$need" ]; then
            FREE_GPUS=$(echo $free | tr ' ' ',' | sed 's/^,//;s/,$//')
            N=$n
            echo "[$(date +%H:%M:%S)] $n free ($free)"
            return
        fi
        echo "[$(date +%H:%M:%S)] only $n free ($free); sleeping 180s"
        sleep 180
    done
}

# --- Train ---
for task in bw gc; do
    echo "=== TRAIN Stage-2 $task in-domain ($(date +%H:%M:%S)) ==="
    wait_for_n_gpus "1 2 3 4 6 7" 2
    CUDA_VISIBLE_DEVICES=$FREE_GPUS HYPPLAN_DIST_BACKEND=gloo \
        $TORCHRUN --nproc_per_node=$N \
        --master_port=$((29570 + RANDOM % 100)) \
        -m src.train_stage2_dagger_ood \
        --task $task \
        --config configs/stage2_dagger_${task}_qwen14b.yaml \
        --use_z 1 \
        > logs/dagger_ood/train_${task}.log 2>&1
    echo "  $task train done"
done

# --- Eval ---
declare -A TEST_DATA=( ["pq"]=data/prontoqa_test.jsonl
                        ["bw"]=data/blocksworld_test.jsonl
                        ["gc"]=data/graphcolor_test.jsonl )
declare -A HEAD_PATH=( ["pq"]=checkpoints/head_pronto_qwen14b_rank/head.pt
                        ["bw"]=checkpoints/head_blocksworld_qwen14b_rank/head.pt
                        ["gc"]=checkpoints/head_graphcolor_qwen14b_rank/head.pt )
declare -A CKPT=( ["pq"]=checkpoints/dagger_stage2_pq_indomain
                   ["bw"]=checkpoints/dagger_stage2_bw_indomain
                   ["gc"]=checkpoints/dagger_stage2_gc_indomain )
declare -A MAX_STEPS=( ["pq"]=12 ["bw"]=16 ["gc"]=10 )

for task in pq bw gc; do
    echo "=== EVAL Stage-2 $task in-domain ($(date +%H:%M:%S)) ==="
    wait_for_n_gpus "1 2 3 4 6 7" 2
    mkdir -p results/eval_stage2_indomain/$task
    i=0
    for g in $(echo $FREE_GPUS | tr ',' ' '); do
        CUDA_VISIBLE_DEVICES=$g $PYTHON -m src.eval_stage2_indomain \
            --task $task \
            --ckpt_dir ${CKPT[$task]} \
            --head_path ${HEAD_PATH[$task]} \
            --test_data ${TEST_DATA[$task]} \
            --output results/eval_stage2_indomain/$task/${task}_shard${i}.jsonl \
            --limit 100 \
            --max_steps ${MAX_STEPS[$task]} \
            --shard_rank $i --shard_world $N \
            > logs/dagger_ood/eval_${task}_shard${i}.log 2>&1 &
        i=$((i+1))
    done
    wait
    echo "  $task eval done"
done

echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
