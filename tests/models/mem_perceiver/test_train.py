import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import MemPerceiver
from collie import  CollieConfig, env
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

config = CollieConfig.from_pretrained("huggyllama/llama-7b",
        trust_remote_code=True)
# config.dp_size = 8
# config.pp_size = 1
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

tokens=torch.zeros(batch_size, seq_len).long().cuda()
attention_mask=torch.zeros(seq_len, seq_len).long().cuda()

mem_perceiver = MemPerceiver(
    model=model, 
    num_layers=num_layers, 
    query_len=query_len, 
    d_query=d_query, 
    d_model=d_model, 
    d_ffn=d_ffn, 
    num_heads=num_heads, 
    chunk_size=chunk_size).cuda()
mem_perceiver.train()

with autocast():
    model_outputs = mem_perceiver(tokens, attention_mask)
logits = model_outputs.logits # (bsz, seq_len, vocab)
loss = logits[:, :, 0].mean()
loss.backward()