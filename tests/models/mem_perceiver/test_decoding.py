import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig
from collie.config import CollieConfig

from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType, llm_dict
import torch
import datasets

def prepare_data_for_eval(samples, bsz, seq_len):
    sub_samples = []
    for sample in samples:
        sub_samples.append(sample)
        if len(sub_samples) == bsz:
            break
    input_ids = torch.tensor(
        [sample['input_ids'][:seq_len] for sample in sub_samples]
    )
    return input_ids

# test generating
data_path = "./eval_datasets/github_65k_llama_tokenized/"

# tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True, )

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

def eval_speed(llm_name, pruner_type, memory_type, chunk_size, memory_size_limit, input_len, generate_len, incremental_type=None, decremental_chunk=False):
    # config
    llm_name_or_path = llm_dict[llm_name]
    config = CollieConfig.from_pretrained(llm_name_or_path,
            trust_remote_code=True)
    config.checkpointing = True
    config.use_flash = True

    batch_size=1
    d_model=config.hidden_size // config.num_attention_heads * config.num_key_value_heads
    num_heads=config.num_key_value_heads
    num_layers=config.model_config.num_hidden_layers

    mem_perceiver_config = {
            # llm config
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            # custom config
            "memory_type": memory_type,
            "chunk_size": chunk_size,
            "num_sink_tokens": 4,
            "memory_size_limit": memory_size_limit,
            "incremental_type": incremental_type,
            "decremental_chunk": decremental_chunk,
            # "review_scheduler": "exp",
            # "review_times": 4,
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

    print('-'*100)
    print(f'chunk_size: {chunk_size} | memory size limit: {memory_size_limit} | \
          memory type: {memory_type} | pruner type: {pruner_type} | input {input_len} tokens | generate {generate_len} tokens'+'='*10, flush=True)
    print('-'*100)
    
    gen_config = GenerationConfig(max_new_tokens=generate_len, min_new_tokens=generate_len, early_stopping=True, eos_token_id=2)
    model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config) # warmup GPU
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True, )
    print(tokenizer.decode(model_outputs[0]))
    # repeat_num = 1
    # # mem_perceiver.report_memory_state()
    # with MeasureGPUTime(repeat_num,):
    #     for _ in range(repeat_num):
    #         model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config)
    #         assert model_outputs.shape[-1] == input_len + generate_len, f'{model_outputs.shape=}, {input_ids.shape=}'
    # print(torch.cuda.max_memory_allocated(torch.device("cuda:0")))
    # torch.cuda.reset_max_memory_allocated(torch.device("cuda:0"))


pruner_types = [PrunerType.CONV]
memory_types = [MemoryType.CHUNK_STREAMING]
decremental_chunk = False
incremental_types = ["linear"]
chunk_sizes = [1024]
memory_size_limits=[1024]
input_lens = [8192]
generate_lens = [1]
for it in incremental_types:
    for ms in memory_size_limits:
        for chunk_size in chunk_sizes:
            for pruner_type in pruner_types: 
                for memory_type in memory_types:
                    for input_len in input_lens:
                        for generate_len in generate_lens:
                            num_chunks = input_len // chunk_size
                            # try:
                            eval_speed(llm_name='llama2-7b', pruner_type=pruner_type, memory_type=memory_type, chunk_size=chunk_size, memory_size_limit=ms, input_len=input_len, 
                                    generate_len=generate_len, incremental_type=it, decremental_chunk=decremental_chunk)
                            # except Exception as e:
                            #     print(e)
                            #     print('!'*30)
                            #     continue

