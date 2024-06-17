export MASTER_PORT="12348"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
dynamic_incremental='Incremental_Chunk_Streaming_Dynamic_History'
fixed_incremental='Incremental_Chunk_Streaming_Fixed_History'
double_incremental="dynamic_incremental_double_compress"
CHUNK_STREAMING_SPARSE="Chunk_Streaming_Sparse"
chunk_streaming='Chunk_Streaming'
pruner_types=('streaming_llm' 'random' 'tova' 'roco' 'conv')
memory_types=($CHUNK_STREAMING_SPARSE)
eval_lens=(16384)

for eval_len in "${eval_lens[@]}"; do
for memory_type in "${memory_types[@]}"; do
for pruner_type in "${pruner_types[@]}"; do
echo $eval_len $memory_type $pruner_type
srun -p moss -n1 --gres=gpu:1 -u ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model llama2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size 512 \
  --use_flash \
  --num_gpus 1 \
  --eval_len $eval_len \
  --compressed_chunk_size 64
done
done
done