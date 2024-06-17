#!/bin/bash
#SBATCH --job-name=eval_all_pruner
#SBATCH --ntasks 1
#SBATCH -p a800
#SBATCH --gres=gpu:1
#SBATCH --output=/remote-home/zyzeng/collie/logs/eval/eval_all_%A_%a.out
#SBATCH --error=/remote-home/zyzeng/collie/logs/eval/eval_all_%A_%a.err

export MASTER_PORT="12347"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
memory_type='dynamic_incremental_double_compress'
pruner_type='perceiver'
perceiver_path="/remote-home/zyzeng/collie/ckpts/llmllama2_7b#prunerperceiver#memoryIncremental_Chunk_Streaming_Dynamic_History#lr2e-05#chunk512#temperature1.0/epoch_3"

srun -p a800 -n1 --mem-per-cpu=4G --gres=gpu:1 -u ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model llama2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size 512 \
  --query_len 64 \
  --d_query 1024 \
  --use_flash \
  --num_gpus 1 \
  --compressed_chunk_size 64 \
  --temperature 1.0 \
  --eval_every 500
  # --perceiver_path $perceiver_path \