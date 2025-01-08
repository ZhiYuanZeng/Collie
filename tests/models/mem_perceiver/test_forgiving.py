import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig
from collie.config import CollieConfig

from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType, llm_dict
import torch
import datasets

def prepare_data_for_eval(data_path, model_name, seq_len):
    with open(data_path, 'r') as f:
        text = f.read()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(
        [input_ids[:seq_len],]
    )
    print(f"the size of data: {input_ids.shape}")
    return input_ids

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

def eval_speed(llm, pruner_type, memory_type, chunk_size, memory_size_limit, input_len, generate_len, decremental_chunk=False, review_type=None, review_time=None, review_length=None):
    review_scheduler = {
        'scheduler': review_type,
        'times': review_time,
        'length': review_length
    }
    # config
    llm_path = llm_dict[llm]
    config = CollieConfig.from_pretrained(llm_path,
            trust_remote_code=True)
    config.checkpointing = True
    config.use_flash = True

    d_model=config.hidden_size // config.num_attention_heads * config.num_key_value_heads
    num_heads=config.num_key_value_heads
    num_layers=config.model_config.num_hidden_layers


    pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
            'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 8192, 
            'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }
    setattr(config.model_config, 'pe_config', pe_config)

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
            "decremental_chunk": decremental_chunk,
            "review_scheduler": review_scheduler,
        }
    # print(mem_perceiver_config)
    setattr(config, 'mem_perceiver_config', mem_perceiver_config)
    mem_perceiver = AutoPruner.from_pretrained(
        pruner_type=pruner_type,
        config=config,
        pretrained_model_name_or_path=llm_path)

    mem_perceiver = mem_perceiver.to(torch.bfloat16)
    mem_perceiver = mem_perceiver.cuda()
    mem_perceiver.eval()


    input_ids = prepare_data_for_eval(data_path="./mtob_book.txt", model_name=llm_path, seq_len=input_len).cuda()

    print('-'*100)
    print(f'llm: {llm} | review_sheduler: {review_scheduler} | chunk_size: {chunk_size} | memory size limit: {memory_size_limit} | \
          memory type: {memory_type} | pruner type: {pruner_type} | input {input_len} tokens | generate {generate_len} tokens'+'='*10, flush=True)
    print('-'*100)
    
    gen_config = GenerationConfig(max_new_tokens=generate_len, min_new_tokens=generate_len, early_stopping=True, eos_token_id=2)
    model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config) # warmup GPU
    repeat_num = 1
    mem_perceiver.report_memory_state()
    with MeasureGPUTime(repeat_num,):
        for _ in range(repeat_num):
            model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config)
            assert model_outputs.shape[-1] == input_len + generate_len, f'{model_outputs.shape=}, {input_ids.shape=}'
    print(torch.cuda.max_memory_allocated(torch.device("cuda:0")))
    torch.cuda.reset_max_memory_allocated(torch.device("cuda:0"))


llms = ['llama2-7b', 'llama3-8b']
# review_types = ['late']
review_types = ['late']
review_times = [0]
review_lens = [8]
pruner_types = [PrunerType.CONV]
memory_types = [MemoryType.CHUNK_STREAMING]
decremental_chunk = False
chunk_sizes = [1024]
memory_size_limits=[1024]
num_chunks = 8
input_lens = [num_chunks * chunk_sizes[0]]
generate_lens = [1]
for llm in llms:
    for review_type in review_types:
        for ms in memory_size_limits:
            for chunk_size in chunk_sizes:
                for pruner_type in pruner_types: 
                    for memory_type in memory_types:
                        for input_len in input_lens:
                            for generate_len in generate_lens:
                                for review_len in review_lens:
                                    for review_time in review_times:
                                        # try:
                                        eval_speed(llm=llm, pruner_type=pruner_type, memory_type=memory_type, chunk_size=chunk_size, memory_size_limit=ms, input_len=input_len, 
                                                generate_len=generate_len, decremental_chunk=decremental_chunk, review_type=review_type, review_time=review_time, review_length=review_len)
                                        # except Exception as e:
                                        #     print(e)
                                        #     print('!'*30)
                                        #     continue

# for input_len in input_lens:
#     eval_speed(pruner_type=PrunerType.STREAMING, memory_type=MemoryType.CHUNK_STREAMING, chunk_size=4096, memory_size_limit=512, input_len=input_len, generate_len=1)