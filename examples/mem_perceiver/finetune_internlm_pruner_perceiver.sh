#!/bin/bash
#SBATCH --job-name=eval_all_pruner
#SBATCH --ntasks 1
#SBATCH -p a800
#SBATCH --gres=gpu:1
#SBATCH --output=/remote-home/zyzeng/collie/logs/eval/eval_all_%A_%a.out
#SBATCH --error=/remote-home/zyzeng/collie/logs/eval/eval_all_%A_%a.err

export MASTER_PORT="12346"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
memory_type='Incremental_Chunk_Streaming_Dynamic_History'
pruner_type='perceiver'
perceiver_path="ckpts/llm[llama2_7b]_pruner[perceiver]_lr[2e-05]_memory[Incremental_Chunk_Streaming_Dynamic_History]/epoch_0-batch_1000"

srun -p moss -n4 --gres=gpu:4 -u ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_train \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model internlm2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size 512 \
  --query_len 64 \
  --d_query 1024 \
  --use_flash \
  --num_gpus 4 \
  --compressed_chunk_size 64 \
  --temperature 1.0
  # --perceiver_path $perceiver_path \