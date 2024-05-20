# README

## Availble Memory
支持的memory类型：
```python
class MemoryType:
    CHUNK_STREAMING="Chunk_Streaming" # Fixed-Size Memory
    FIXED_INCREMENTAL="Incremental_Chunk_Streaming_Fixed_History" # Incremental Fixed Memory
    DYNAMIC_INCREMENTAL="Incremental_Chunk_Streaming_Dynamic_History" # Incremental Dynamic Memory
    DYNAMIC_INCREMENTAL_DOUBLE_COMPRESS="dynamic_incremental_double_compress" # Incremental Dynamic Memory with Double Compression
```
## Available Compresser
支持的Compressers:

Pruner:
```python
class AutoPruner:
    @staticmethod
    def from_pretrained(pruner_type, pretrained_model_name_or_path, config, perceiver_path=None):
        if pruner_type == PrunerType.H2O:
            config.use_flash = False
            print('Warning: the h2o pruner requires attention scores, therefore the flash_attention is set to False!')
            pruner = H2oPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.STREAMING:
            pruner = StreamingLMPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.CHUNK_PREFIX:
            pruner = ChunkPrefix.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.TOVA:
            config.use_flash = False
            print('Warning: the h2o pruner requires attention scores, therefore the flash_attention is set to False!')
            pruner = TovaPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.RANDOM:
            pruner = RandomPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.LOCAL_WINDOW: # remove context
            config.mem_perceiver_config['compressed_chunk_size'] = 0
            pruner = TovaPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif pruner_type == PrunerType.NO_COMPRESS:
            pruner = build_llm_from_name_or_path(pretrained_model_name_or_path, config) # no compress
        # parameters required
        elif pruner_type == PrunerType.PERCEIVER:
            pruner = PerceiverPruner.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        else:
            raise NotImplementedError
        if pruner_type in (PrunerType.H2O):
            assert config.mem_perceiver_config['memory_type'] in (MemoryType.CHUNK_STREAMING, MemoryType.DYNAMIC_INCREMENTAL)

        return pruner
```

Fuser:
```python
class AutoFuser:
    @staticmethod
    def from_pretrained(fuser_type, pretrained_model_name_or_path, config, perceiver_path):
        if fuser_type == 'perceiver':
            fuser = SparseFuserPerceiver.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        elif fuser_type == 'llm':
            fuser = LLMFuser.from_pretrained(pretrained_model_name_or_path, config, perceiver_path)
        else:
            raise NotImplementedError
        return fuser
```

## Usage
参考
- `tests/models/mem_perceiver/test_decoding.py`
- `tests/models/mem_perceiver/test_llm_fuser.py`
```python
# 创建collie config
config = CollieConfig.from_pretrained(llm_name_or_path,
        trust_remote_code=True)
config.checkpointing = True
config.use_flash = True
# 创建compresser config
mem_perceiver_config = {
    # llm config
    "d_model": d_model,
    "num_heads": num_heads,
    "num_layers": num_layers,
    # custom config
    "memory_type": MemoryType.DYNAMIC_INCREMENTAL,
    "query_len": compressed_chunk_size,
    "compressed_chunk_size": compressed_chunk_size,
    "d_query": d_query,
    "chunk_size": chunk_size,
    "num_sink_tokens": 4,
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

评测tova的ppl:
`bash examples/mem_perceiver/finetune_tova_pruner.sh`