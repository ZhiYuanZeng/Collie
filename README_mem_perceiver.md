# AutoCompresser
将序列划分成多个chunk，并迭代压缩chunk。
特点：
- 将LLM和Compresser解耦，因此无需为模型定制Compresser，只要模型输出包含KV Cache，attention scores就能使用本框架
- 支持多种Compresser，目前经过良好测试的主要是KV Cache Pruner
- 支持多种Memory管理，例如Chunk_Streaming: 固定memory大小，每次压缩一个chunk都刷新memory


## Compresser
支持的Compressers:
Pruner: pruner保留一部分kv cache，注意很多kv cache压缩方法需要attention scores，以至于无法使用flash attention。导致压缩后的速度可能反而更慢，显存反而更多。

```python
class PrunerType:
    H2O="h2o"
    STREAMING="streaming_llm"
    CHUNK_PREFIX="chunk_prefix"
    TOVA="tova"
    RANDOM="random"
    LOCAL_WINDOW="local_window"
    NO_COMPRESS="no_compress"
    PERCEIVER="perceiver"
    ROCO="roco"
    CONV="conv"
```

需要attention scores的pruning方法：
- H2O
- TOVA
- ROCO
- CONV

不需要attention scores的pruning方法：
- Streaming
- Chunk_Prefix
- RANDOM
- LOCAL_WINDOW
- NO_COMPRESS

Fuser:
```python
class FuserType:
    PERCEIVER='perceiver'
    LLM='llm'
```

## Memory
支持的memory类型：
```python
class MemoryType:
    CHUNK_STREAMING="Chunk_Streaming" # Fixed-Size Memory
    FIXED_INCREMENTAL="Incremental_Chunk_Streaming_Fixed_History" # Incremental Fixed Memory
    DYNAMIC_INCREMENTAL="Incremental_Chunk_Streaming_Dynamic_History" # Incremental Dynamic Memory
    DYNAMIC_INCREMENTAL_DOUBLE_COMPRESS="dynamic_incremental_double_compress" # Incremental Dynamic Memory with Double Compression
```

- CHUNK_STREAMING: 固定memory大小，每次压缩新的chunk更新Memory
- FIXED_INCREMENTAL: 每次压缩新的chunk会在原有memory的基础上拼接
- DYNAMIC_INCREMENTAL: 每次压缩新的chunk，会刷新memory，并且增加memory size
- DYNAMIC_INCREMENTAL_DOUBLE_COMPRESS: 每次压缩新的chunk，会刷新memory，并且增加memory size，但是还会对memory再做一次压缩，第二次压缩的结果会输入LLM

## Usage
参考
- pruner: `tests/models/mem_perceiver/test_decoding.py`
- fuser: `tests/models/mem_perceiver/test_llm_fuser.py`
```python
# 创建collie config
config = CollieConfig.from_pretrained(llm_name_or_path,
        trust_remote_code=True)
config.checkpointing = True
config.use_flash = True
# 创建compresser config
mem_perceiver_config = {
    # llm config，从llm config获取
    "d_model": d_model,
    "num_heads": num_heads,
    "num_layers": num_layers,

    # custom config，需要自定义的参数
    "memory_type": MemoryType.DYNAMIC_INCREMENTAL, # 选择哪种memory
    "chunk_size": chunk_size, # 滚动压缩的chunk size
    "compressed_chunk_size": compressed_chunk_size, # 每个chunk压缩后的长度 
    "query_len": compressed_chunk_size, # 只有perceiver需要这个参数
    "d_query": d_query, # 只有perceiver需要这个参数，应该设置的小一点
    "num_sink_tokens": 4, # sink token的数量，参考streaming llm
}
# 调用AutoXXX.from_pretrained
mem_perceiver = AutoPruner.from_pretrained(
    pruner_type=PrunerType.PERCEIVER,
    config=config,
    pretrained_model_name_or_path=llm_name_or_path)
mem_perceiver.forward(...)
mem_perceiver.generate(...)
```

## 训练
训练perceiver-pruner:
`bash examples/mem_perceiver/finetune_pruner_perceiver.ssh`

训练perceiver-fuser:
`bash examples/mem_perceiver/finetune_sparse_fuser.sh`

训练llm-fuser:
`bash examples/mem_perceiver/finetune_llm_fuser.sh`

评测tova的ppl:
`bash examples/mem_perceiver/finetune_tova_pruner.sh`