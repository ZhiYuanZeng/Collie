import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType
from collie import CollieConfig, env, Trainer
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from collie.data import CollieDatasetForTraining, CollieDatasetForGeneration
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

llm_name_or_path = "/remote-home/share/personal/zyzeng/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"

# config
config = CollieConfig.from_pretrained(llm_name_or_path,
        trust_remote_code=True)
config.checkpointing = True
config.use_flash = False

batch_size=2
seq_len=1028
chunk_size=512
d_model=config.hidden_size
d_query=config.hidden_size // 4
d_ffn=config.hidden_size // 2
num_heads=config.num_key_value_heads
query_len=chunk_size // 8
num_layers=config.model_config.num_hidden_layers

mem_perceiver_config = {
    # llm config
    "d_model": d_model,
    "num_heads": num_heads,
    "num_layers": num_layers,
    # custom config
    "memory_type": MemoryType.CHUNK_STREAMING,
    "query_len": query_len,
    "d_query": d_query,
    "chunk_size": chunk_size,
    "num_sink_tokens": 4,
}
setattr(config, 'mem_perceiver_config', mem_perceiver_config) 

pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
          'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 2048, 
          'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }
setattr(config.model_config, 'pe_config', pe_config)


mem_perceiver = AutoPruner.from_pretrained(
    pruner_type=PrunerType.PERCEIVER,
    config=config,
    pretrained_model_name_or_path=llm_name_or_path,
    perceiver_path="ckpts/parallel_sparse_lr2e-05_memory_update_incremental_compressed_read_all_compressed/epoch_2")
mem_perceiver = mem_perceiver.cuda()
mem_perceiver.eval()

# test generating
data_path = "./eval_datasets/github_65k_llama_tokenized/"

tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, trust_remote_code=True)

class MeasureGPUTime:
    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()  # 记录开始时间

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()  # 记录结束时间
        torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        self.execution_time = self.start_event.elapsed_time(self.end_event) / 1000.0  # 计算执行时间（以秒为单位）
        print(f"函数的执行时间为: {self.execution_time:.6f} 秒")

mem_perceiver = mem_perceiver.to(torch.bfloat16)

# seq_lens = [8192]
# num_tokens_to_generate = [2]
seq_lens = [8192, 16000, 32000]
num_tokens_to_generate = [2, 64, 256, 1024]
for s in seq_lens:
    print(f'input {s} tokens'+'.'*20)
    input_ids = prepare_data_for_eval(datasets.load_from_disk(data_path)['train'], batch_size, s).cuda()

    for ntg in num_tokens_to_generate:
        print(f'generate {ntg} tokens', flush=True)
        gen_config = GenerationConfig(max_new_tokens=ntg, min_new_tokens=ntg, early_stopping=True, eos_token_id=2)

        with MeasureGPUTime():
            model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config)
            assert model_outputs.shape[-1] == s + ntg, f'{model_outputs.shape=}, {input_ids.shape=}'
