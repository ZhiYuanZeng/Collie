import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import MemPerceiver
from collie import  CollieConfig, env
import torch
from torch.cuda.amp import autocast

config = CollieConfig.from_pretrained("huggyllama/llama-7b",
        trust_remote_code=True)
config.checkpointing=False
config.num_hidden_layers=2 # reduce model size
model = LlamaForCausalLM(config=config)

batch_size=1
seq_len=2048
chunk_size=512
d_model=config.hidden_size
d_query=config.hidden_size // 4
d_ffn=config.hidden_size // 2
num_heads=config.num_key_value_heads
query_len=chunk_size // 8
num_layers=config.num_hidden_layers

mem_perceiver_config = {
    "d_model": d_model,
    "d_query": d_query,
    "d_ffn": d_ffn,
    "chunk_size": chunk_size,
    "query_len": query_len,
    "num_heads": num_heads,
    "num_layers": num_layers
}
setattr(config, 'mem_perceiver_config', mem_perceiver_config) 

tokens=torch.zeros(batch_size, seq_len).long().cuda()

mem_perceiver = MemPerceiver.from_config(config, model).cuda()

gen_config = GenerationConfig(max_new_tokens=8, early_stopping=True, eos_token_id=2)

mem_perceiver.eval()
with autocast():
    with torch.no_grad():
        outs = mem_perceiver.generate(tokens, generation_config=gen_config)