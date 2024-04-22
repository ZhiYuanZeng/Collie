import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import MemPerceiver, ParallelMemPerceiver, H2oPruner, SparseParallelPerceiver
from collie import CollieConfig, env, Trainer
from collie import  CollieConfig, env
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

config = CollieConfig.from_pretrained("huggyllama/llama-7b",
        trust_remote_code=True)
# config.dp_size = 8
# config.pp_size = 1
config.num_hidden_layers=2 # reduce model size
config.checkpointing = True
config.use_flash = False

batch_size=1
seq_len=1024
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
    # "d_ffn": d_ffn,
    "chunk_size": chunk_size,
    "query_len": query_len,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "num_sink_tokens": 4,
}
setattr(config, 'mem_perceiver_config', mem_perceiver_config) 

model = LlamaForCausalLM(config=config)

tokens=torch.zeros(batch_size, seq_len).long().cuda()
attention_mask=torch.zeros(batch_size, seq_len).long().cuda()

mem_perceiver = SparseParallelPerceiver.from_config(
    config=config,
    model=model)
mem_perceiver = mem_perceiver.cuda()
mem_perceiver.eval()

# test forward
# with autocast():
#     model_outputs = mem_perceiver(tokens, attention_mask)
#     logits = model_outputs.logits # (bsz, seq_len, vocab)

# test generating
gen_config = GenerationConfig(max_new_tokens=20, early_stopping=True, eos_token_id=2)
with autocast(): 
    model_outputs = mem_perceiver.generate(tokens, generation_config=gen_config)

# test saving
trainer = Trainer(mem_perceiver, config=config)
ckpt_path = './ckpts/demo/'
trainer.save_model(ckpt_path)

# test loading
trainer.load_model(ckpt_path)