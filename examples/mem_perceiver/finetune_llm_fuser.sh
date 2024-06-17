#!/bin/bash
#SBATCH --job-name=eval_all_pruner
#SBATCH --ntasks 1
#SBATCH -p a800
#SBATCH --gres=gpu:1
#SBATCH --output=/remote-home/zyzeng/collie/logs/eval/eval_all_%A_%a.out
#SBATCH --error=/remote-home/zyzeng/collie/logs/eval/eval_all_%A_%a.err
export NCCL_DEBUG=INFO
export MASTER_PORT="12346"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
dynamic_incremental='Incremental_Chunk_Streaming_Dynamic_History'
fixed_incremental='Incremental_Chunk_Streaming_Fixed_History'
double_incremental="dynamic_incremental_double_compress"
chunk_streaming='Chunk_Streaming'

perceiver_path="ckpts/llmllama2_7b#fuserllm#memoryIncremental_Chunk_Streaming_Dynamic_History#lr2e-05#chunk512#compress64/epoch_0-batch_1000/"
export CUDA_HOME="/remote-home/zyzeng/cuda-11.8"
num_gpus=4

list=($fixed_incremental $double_incremental $dynamic_incremental)
for memory_type in "${list[@]}"; do
srun -p moss -n ${num_gpus} --mem-per-cpu=4G --gres=gpu:${num_gpus} -u ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --fuser_type llm \
  --do_train \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model llama2_7b \
  --num_train_samples 8000 \
  --num_eval_samples 128 \
  --chunk_size 512 \
  --use_flash \
  --num_gpus ${num_gpus} \
  --compressed_chunk_size 64 \
  --query_len 64 \
  --temperature 1.0 \
  --eval_every 500 \
  --eval_len 16384
done