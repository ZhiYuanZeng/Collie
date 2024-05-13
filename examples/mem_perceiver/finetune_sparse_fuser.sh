export MASTER_PORT="12347"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
srun -p moss --gres=gpu:4 -u -n4 ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
--fuser_type sparse_fuser \
--memory_type Incremental_Chunk_Streaming_Dynamic_History \
--do_train \
--lr 0.00002 \
--llm_model llama2_7b \
--num_train_samples 15000 \
--num_eval_samples 128 \
--chunk_size 512 \
--query_len 64 \
--d_query 1024 \
--use_flash