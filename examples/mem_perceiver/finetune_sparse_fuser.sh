export MASTER_PORT="12347"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
perceiver_path='ckpts/llmllama2_7b#fuserperceiver#memoryChunk_Streaming#lr2e-05#chunk512#temperature1.0/epoch_1-batch_3000'
srun -p moss --gres=gpu:1 -u -n1 ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
    --fuser_type perceiver \
    --memory_type Chunk_Streaming \
    --do_eval \
    --lr 0.00002 \
    --llm_model llama2_7b \
    --num_train_samples 15000 \
    --num_eval_samples 128 \
    --chunk_size 512 \
    --query_len 64 \
    --d_query 1024 \
    --use_flash \
    --num_gpus 8 \
    --compressed_chunk_size 64 \
    --temperature 1.0 \
    --eval_every 500 \
    --perceiver_path $perceiver_path