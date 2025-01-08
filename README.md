# Memorize Step by Step
Implementation for the paper "Memorize Step by Step: Efficient Long-Context Prefilling with Incremental Memory and Decremental Chunk" 
(https://aclanthology.org/2024.emnlp-main.1169/)

## Features
将序列划分成多个chunk，并迭代压缩chunk，支持无限长输入的prefill。
- 将LLM和Compresser解耦，因此无需为模型定制Compresser，只要模型输出包含KV Cache，attention scores就能使用本框架
- 支持多种Compresser，目前支持的都是kv-cache pruner
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

## Memory
支持的memory类型：
```python
class MemoryType:
    CHUNK_STREAMING="FM"
    DualMemory="dual_memory"
    FIXED_INCREMENTAL="Incremental_Chunk_Streaming_Fixed_History"
    DYNAMIC_INCREMENTAL="IM"
    DYNAMIC_INCREMENTAL_DOUBLE_COMPRESS="dynamic_incremental_double_compress"
    RETRIEVE_INCREMENTAL="Incremental_Chunk_Streaming_Retrieved_History"
    RETRIEVE_ALL_KV="Cache_All_KV_Retrieve_History"
```

- CHUNK_STREAMING: 固定memory大小，每次压缩新的chunk更新Memory
- FIXED_INCREMENTAL: 每次压缩新的chunk会在原有memory的基础上拼接
- DYNAMIC_INCREMENTAL: 每次压缩新的chunk，会刷新memory，并且增加memory size
- DYNAMIC_INCREMENTAL_DOUBLE_COMPRESS: 每次压缩新的chunk，会刷新memory，并且增加memory size，但是还会对memory再做一次压缩，第二次压缩的结果会输入LLM

## Usage
参考
`tests/models/mem_perceiver/test_decoding.py`

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
mem_perceiver.forward(kwargs)
mem_perceiver.generate(kwargs)
```

评测ppl:
`bash examples/mem_perceiver/eval_all_pruners.sh`

## Memory Probing
`collie/models/mem_perceiver/utils.py`定义了一些MemoryState，每个MemoryState对应一种Probing，例如MemoryForgive调查的是memory的遗忘曲线

以`tests/models/mem_perceiver/test_forgiving.py`为例，调用model.report_memory_state()可以导出各种probing的状态。

`tests/models/mem_perceiver/compare_forgiving.ipynb`可视化了memory的遗忘曲线

## TODO
目前的实现和collie绑定，但是其实可以和transformers直接兼容