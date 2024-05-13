import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig

from collie.models import LlamaForCausalLM
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
pe_config = {'exp': False, '1d': False, 'imp': False, 'log': False, 
          'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
          'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }
setattr(config.model_config, 'pe_config', pe_config)
model = LlamaForCausalLM(config=config).cuda()


batch_size=1
seq_len=2048
tokens=torch.zeros(batch_size, seq_len).long().cuda()
attention_mask=torch.zeros(batch_size, seq_len).long().cuda()


# test decoding
gen_config = GenerationConfig(max_new_tokens=8, early_stopping=True, eos_token_id=2)

model.eval()
with autocast():
    with torch.no_grad():
        outs = model.generate(tokens, generation_config=gen_config)