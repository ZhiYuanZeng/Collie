export MASTER_PORT="12348"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
dynamic_incremental='Incremental_Chunk_Streaming_Dynamic_History'
fixed_incremental='Incremental_Chunk_Streaming_Fixed_History'
double_incremental="dynamic_incremental_double_compress"
chunk_streaming='Chunk_Streaming'

pruner_types=('conv')
eval_len=32768
chunk_sizes=(1024)
num_gpus=4
memory_types=($fixed_incremental)
memory_sizes=(1024 512 256)

for mt in "${memory_types[@]}"; do
for chunk_size in "${chunk_sizes[@]}"; do
for pruner_type in "${pruner_types[@]}"; do
for ms in "${memory_sizes[@]}"; do
# srun -n${num_gpus} -p a800 --gres=gpu:${num_gpus} -u ${python} 
${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $mt \
  --llm_model llama2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size $chunk_size \
  --use_flash \
  --num_gpus ${num_gpus} \
  --eval_len $eval_len \
  --memory_size_limit $ms \
  --tag compare_fixed_and_incremental_v2
done
done
done
done

pruner_types=('streaming_llm')
eval_len=32768
chunk_sizes=(8192)
num_gpus=4
memory_types=($fixed_incremental)
memory_sizes=(2048 4096 8192)

for mt in "${memory_types[@]}"; do
for chunk_size in "${chunk_sizes[@]}"; do
for pruner_type in "${pruner_types[@]}"; do
for ms in "${memory_sizes[@]}"; do
# srun -n${num_gpus} -p a800 --gres=gpu:${num_gpus} -u ${python} 
${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $mt \
  --llm_model llama2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size $chunk_size \
  --use_flash \
  --num_gpus ${num_gpus} \
  --eval_len $eval_len \
  --memory_size_limit $ms \
  --tag compare_fixed_and_incremental_v2
done
done
done
done

# for chunk_size in "${chunk_sizes[@]}"; do
# for pruner_type in "${pruner_types[@]}"; do
# for ms in "${memory_sizes[@]}"; do
# ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
#   --pruner_type $pruner_type \
#   --do_eval \
#   --lr 0.00002 \
#   --memory_type $dynamic_incremental \
#   --llm_model llama2_7b \
#   --num_train_samples 15000 \
#   --num_eval_samples 128 \
#   --chunk_size $chunk_size \
#   --use_flash \
#   --num_gpus ${num_gpus} \
#   --eval_len $eval_len \
#   --memory_size_limit $ms \
#   --tag compare_decremental_v2 \
#   --decremental_chunk
# done
# done
done

# memory_sizes=(256 512 1024 2048)
# for pruner_type in "${pruner_types[@]}"; do
# for ms in "${memory_sizes[@]}"; do
# num_chunks=$(expr $eval_len / $chunk_size)
# ccs=$(expr $ms / $num_chunks)
# srun -n${num_gpus} -p moss --gres=gpu:${num_gpus} ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
#   --pruner_type $pruner_type \
#   --do_eval \
#   --lr 0.00002 \
#   --memory_type $fixed_incremental \
#   --llm_model llama2_7b \
#   --num_train_samples 15000 \
#   --num_eval_samples 128 \
#   --chunk_size $chunk_size \
#   --use_flash \
#   --num_gpus ${num_gpus} \
#   --eval_len $eval_len \
#   --compressed_chunk_size $ccs \
#   --memory_size_limit $ms \
#   --tag compare_fixed_and_incremental
# done
# done
