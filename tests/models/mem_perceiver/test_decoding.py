import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import AutoPruner
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
    "memory_type": "increment_compressed_read_all_compressed",
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
    pruner_type="parallel_sparse",
    config=config,
    pretrained_model_name_or_path=llm_name_or_path,
    perceiver_path="ckpts/parallel_sparse_lr2e-05_memory_update_incremental_compressed_read_all_compressed/epoch_2")
mem_perceiver = mem_perceiver.cuda()
mem_perceiver.eval()

# test generating
data_path = "/remote-home/share/personal/zyzeng/data/redpajama-15k-4k-llama/"
input_ids = prepare_data_for_eval(datasets.load_from_disk(data_path), batch_size, seq_len).cuda()

gen_config = GenerationConfig(max_new_tokens=20, early_stopping=True, eos_token_id=2)
tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)
with autocast(): 
    model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config)

for i, generated_sequence in enumerate(model_outputs):
    generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    # print(f"Generated text {i+1}: {generated_text}")
