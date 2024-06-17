
from transformers import AutoTokenizer, GenerationConfig
from collie import CollieConfig, env, Trainer
from collie import  CollieConfig, env
from collie.models import LlamaForCausalLM, InternLM2ForCausalLM
from collie.models.mem_perceiver import AutoFuser, MemoryType, AutoPruner
from torch.cuda.amp import autocast
import torch
from transformers import AutoModelForCausalLM
import datasets

llm_name_or_path = "/remote-home/share/models/models--internlm2-7b"

config = CollieConfig.from_pretrained(llm_name_or_path,
        trust_remote_code=True)

config.checkpointing = True
config.use_flash = True

batch_size=1
seq_len=64
chunk_size=8
d_model=config.hidden_size
num_heads=config.num_key_value_heads
compressed_chunk_size=chunk_size // 8
num_layers=config.model_config.num_hidden_layers

mem_perceiver_config = {
    "d_model": d_model,
    # "d_ffn": d_ffn,
    "query_len": compressed_chunk_size,
    "chunk_size": chunk_size,
    "num_heads": num_heads,
    "num_sink_tokens": 4,
    "memory_type": MemoryType.CHUNK_STREAMING,
    "num_layers": num_layers,
    "memory_size_limit": 4
}

pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
          'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 2048, 
          'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'None', 'ntk_alpha': 1., }
setattr(config.model_config, 'pe_config', pe_config)
setattr(config, 'mem_perceiver_config', mem_perceiver_config) 

tokens = torch.tensor(datasets.load_from_disk("/remote-home/share/personal/wtshu/data/redpajama-15k-8k-internlm/test/github")[0]['input_ids'][:100]).unsqueeze(dim=0)
tokens=tokens.long().cuda()
gen_config = GenerationConfig(max_new_tokens=32, min_new_tokens=32, early_stopping=True, eos_token_id=2, return_dict_in_generate=True, output_scores=True)
# collie_model = InternLM2ForCausalLM.from_pretrained(llm_name_or_path, config=config).cuda().to(torch.bfloat16)
collie_model = AutoPruner.from_pretrained(
    pruner_type="streaming_llm",
    pretrained_model_name_or_path=llm_name_or_path,
    config=config,
    perceiver_path=None).cuda().to(torch.bfloat16)
hf_model = AutoModelForCausalLM.from_pretrained(llm_name_or_path, trust_remote_code=True).cuda().to(torch.bfloat16)
with torch.no_grad():
    model_outputs1 = collie_model.generate(tokens, generation_config=gen_config)
    model_outputs2 = hf_model.generate(tokens, generation_config=gen_config)
    logits1 = torch.stack(model_outputs1.scores, dim=1)
    logits2 = torch.stack(model_outputs2.scores, dim=1)
    generated_sequence1 = model_outputs1.sequences[0]
    generated_sequence2 = model_outputs2.sequences[0]
    # logits1 = model_outputs1.logits # (bsz, seq_len, vocab)
    # logits2 = model_outputs2.logits
    print(torch.max(logits1 - logits2).abs())
    print(generated_sequence1[-32:])
    print(generated_sequence2[-32:])
    assert torch.all(generated_sequence1 == generated_sequence2)
