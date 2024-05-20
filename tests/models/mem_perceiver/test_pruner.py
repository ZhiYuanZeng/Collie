import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import AutoPruner
from collie import CollieConfig, env, Trainer
from collie import  CollieConfig, env
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

llm_name_or_path = "/remote-home/share/personal/zyzeng/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"

config = CollieConfig.from_pretrained(llm_name_or_path,
        trust_remote_code=True)
# config.dp_size = 8
# config.pp_size = 1
config.checkpointing = True
config.use_flash = False

batch_size=1
seq_len=2048
chunk_size=512
d_model=config.hidden_size
d_query=config.hidden_size // 4
d_ffn=config.hidden_size // 2
num_heads=config.num_attention_heads
compressed_chunk_size=chunk_size // 8
num_layers = config.num_hidden_layers
assert num_layers > 0
mem_perceiver_config = {
    "d_model": d_model,
    "d_query": d_query,
    # "d_ffn": d_ffn,
    "compressed_chunk_size": compressed_chunk_size,
    "chunk_size": chunk_size,
    "query_len": compressed_chunk_size,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "num_sink_tokens": 4,
    "memory_type": "Chunk_Streaming"
}
pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
          'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 2048, 
          'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }

setattr(config, 'mem_perceiver_config', mem_perceiver_config) 
setattr(config.model_config, 'pe_config', pe_config)

tokens=torch.zeros(batch_size, seq_len).long().cuda()
attention_mask=torch.zeros(batch_size, seq_len).long().cuda()

mem_perceiver = AutoPruner.from_pretrained(
    pruner_type='perceiver',
    pretrained_model_name_or_path=llm_name_or_path,
    config=config,
    perceiver_path=None
    )
mem_perceiver = mem_perceiver.cuda()
mem_perceiver.train()

# test forward
with autocast():
    model_outputs = mem_perceiver(tokens, attention_mask)
    logits = model_outputs.logits # (bsz, seq_len, vocab)

# test generating
gen_config = GenerationConfig(max_new_tokens=20, early_stopping=True, eos_token_id=2)
with autocast(): 
    model_outputs = mem_perceiver.generate(tokens, generation_config=gen_config)

# test saving
# trainer = Trainer(mem_perceiver, config=config)
# ckpt_path = './ckpts/demo/'
# trainer.save_model(ckpt_path)

# # test loading
# trainer.load_model(ckpt_path)