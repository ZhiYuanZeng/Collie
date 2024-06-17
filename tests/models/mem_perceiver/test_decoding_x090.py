import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig
from collie.config import CollieConfig

from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType
import torch
import datasets
import IPython

def prepare_data_for_eval(samples, bsz, seq_len):
    return torch.zeros(bsz, seq_len).long()

# llm_name_or_path = "/remote-home/share/models/internlm2-7b-base-hf/"
# llm_name_or_path = "/remote-home/share/personal/zyzeng/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
llm_name_or_path = "/remote-home/share/models/llama_v2_hf/7b/"
# llm_name_or_path = "/remote-home/share/models/llama_v2_hf/70b/"
# llm_name_or_path = "/remote-home/share/storage/zyyin/moss2Huawei"

# test generating
data_path = "./eval_datasets/github_65k_llama_tokenized/"

tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True)

class MeasureGPUTime():
    def __init__(self, norm=1) -> None:
        self.norm = norm

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()  # 记录开始时间

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()  # 记录结束时间
        torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        self.execution_time = self.start_event.elapsed_time(self.end_event) / 1000.0 / self.norm  # 计算执行时间（以秒为单位）
        print(f"函数的执行时间为: {self.execution_time:.6f} 秒", flush=True)

def eval_speed(pruner_type, memory_type, chunk_size, memory_size_limit, input_len, generate_len, incremental_type=None, decremental_chunk=False):
    # config
    config = CollieConfig.from_pretrained(llm_name_or_path,
            trust_remote_code=True)
    config.checkpointing = True
    config.use_flash = True

    batch_size=1
    d_model=config.hidden_size // config.num_attention_heads * config.num_key_value_heads
    num_heads=config.num_key_value_heads
    num_layers=config.model_config.num_hidden_layers


    pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
            'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 2048, 
            'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }
    setattr(config.model_config, 'pe_config', pe_config)

    mem_perceiver_config = {
            # llm config
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            # custom config
            "memory_type": memory_type,
            "compress_ratio": 0.123,
            "chunk_size": chunk_size,
            "num_sink_tokens": 4,
            "memory_size_limit": memory_size_limit,
            "incremental_type": incremental_type,
            "decremental_chunk": decremental_chunk
        }
    # print(mem_perceiver_config)
    setattr(config, 'mem_perceiver_config', mem_perceiver_config) 
    mem_perceiver = AutoPruner.from_pretrained(
        pruner_type=pruner_type,
        config=config,
        pretrained_model_name_or_path=llm_name_or_path)

    mem_perceiver = mem_perceiver.to(torch.bfloat16)
    mem_perceiver = mem_perceiver.cuda()
    mem_perceiver.eval()


    input_ids = prepare_data_for_eval(datasets.load_from_disk(data_path)['train'], batch_size, input_len).cuda()

    print('-'*30)
    print(f'chunk_size: {chunk_size} | memory size limit: {memory_size_limit} | \
          memory type: {memory_type} | pruner type: {pruner_type} | input {input_len} tokens | generate {generate_len} tokens'+'='*10, flush=True)
    print('-'*30)
    
    gen_config = GenerationConfig(max_new_tokens=generate_len, min_new_tokens=generate_len, early_stopping=True, eos_token_id=2)
    model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config) # warmup GPU
    repeat_num = 3
    with MeasureGPUTime(repeat_num,):
        for _ in range(repeat_num):
            model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config)
            assert model_outputs.shape[-1] == input_len + generate_len, f'{model_outputs.shape=}, {input_ids.shape=}'
    print(torch.cuda.max_memory_allocated(torch.device("cuda:0")))
    torch.cuda.reset_max_memory_allocated(torch.device("cuda:0"))

def batch_eval(incremental_types, memory_size_limits, pruner_types, memory_types, input_lens, generate_lens, decremental_chunk):
    for it in incremental_types:
        for ms in memory_size_limits:
            for chunk_size in chunk_sizes:
                for pruner_type in pruner_types: 
                    for memory_type in memory_types:
                        for input_len in input_lens:
                            for generate_len in generate_lens:
                                try:
                                    eval_speed(pruner_type=pruner_type, memory_type=memory_type, chunk_size=chunk_size, memory_size_limit=ms, input_len=input_len, 
                                            generate_len=generate_len, incremental_type=it, decremental_chunk=decremental_chunk)
                                except Exception as e:
                                    print(e)
                                    continue

# pruner_types = [PrunerType.CONV]
# memory_types = [MemoryType.CHUNK_STREAMING, MemoryType.DYNAMIC_INCREMENTAL]
# decremental_chunk = False
# incremental_types = ["linear"]
# chunk_sizes = [512]
# memory_size_limits=[128, 256, 512]
# input_lens = [8192]
# generate_lens = [1]

# batch_eval(incremental_types, memory_size_limits, pruner_types, memory_types, input_lens, generate_lens, decremental_chunk)

# print('\n'*2)
# print('--------------eval decremental chunk----------------')
# memory_types = [MemoryType.DYNAMIC_INCREMENTAL]
# decremental_chunk = True
# batch_eval(incremental_types, memory_size_limits, pruner_types, memory_types, input_lens, generate_lens, decremental_chunk)

# print('\n'*5)
# print('=============eval streaming llm============')
# pruner_types = [PrunerType.STREAMING]
# memory_types = [MemoryType.CHUNK_STREAMING, MemoryType.DYNAMIC_INCREMENTAL]
# decremental_chunk = False
# incremental_types = ["linear"]
# chunk_sizes = [2048]
# memory_size_limits=[512, 1024, 2048]
# input_lens = [8192]
# generate_lens = [1]
# batch_eval(incremental_types, memory_size_limits, pruner_types, memory_types, input_lens, generate_lens, decremental_chunk)


print('\n'*2)
pruner_types = [PrunerType.STREAMING]
memory_types = [MemoryType.DYNAMIC_INCREMENTAL]
incremental_types = ["linear"]
chunk_sizes = [2048]
memory_size_limits=[2048]
input_lens = [8192]
generate_lens = [1]
print('--------------eval decremental chunk----------------')
memory_types = [MemoryType.DYNAMIC_INCREMENTAL]
decremental_chunk = True
batch_eval(incremental_types, memory_size_limits, pruner_types, memory_types, input_lens, generate_lens, decremental_chunk)