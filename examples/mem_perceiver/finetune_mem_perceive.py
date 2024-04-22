import sys
sys.path.append('..')
from transformers import AutoTokenizer

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining

from collie.optim.lomo import Lomo

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import MemPerceiver, ParallelMemPerceiver, H2oPruner, SparseParallelPerceiver, StreamingLMPruner, RandomPruner

from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from collie.metrics import DecodeMetric, PPLMetric
from collie.module import GPTLMLoss
import datasets
from datasets import Dataset
import torch
import os
import argparse


# 1. 设置路径
# 1.1 预训练模型路径
# pretrained_model = "/remote-home/share/models/open_llama_3b_v2/"
pretrained_model = "/remote-home/share/personal/zyzeng/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"

# 2. 设置配置
# 2.1 加载配置
config = CollieConfig.from_pretrained(pretrained_model, trust_remote_code=True)
config.tp_size = 1
config.dp_size = 1
config.pp_size = 1
config.train_epochs = 100
config.eval_per_n_steps = 0
config.eval_per_n_epochs = 1 
config.train_micro_batch_size = 4
config.gradient_accumulation_steps = 1
config.eval_batch_size = 10
config.use_flash = False
config.ds_config = {
        "fp16": {
            "enabled": True
        },
        # "zero_allow_untested_optimizer": True,
        # "zero_force_ds_cpu_optimizer": False,
        # "zero_optimization": {
        #     "stage": 2,
        #     "offload_optimizer": {
        #         "device": "cpu",
        #         "pin_memory": False
        #     }
        # }
}
config.checkpointing = True

chunk_size=512
d_model=config.hidden_size
d_query=config.hidden_size // 4
d_ffn=config.hidden_size // 2
num_heads=config.num_attention_heads
query_len=chunk_size // 8
num_layers=config.num_hidden_layers

mem_perceiver_config = {
    "d_model": d_model,
    "d_query": d_query,
    "d_ffn": d_ffn,
    "chunk_size": chunk_size,
    "query_len": query_len,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "num_sink_tokens": 16,
    "temperature": 0.1
}
setattr(config, 'mem_perceiver_config', mem_perceiver_config) 


# 3. 设置tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

# 4. 加载数据集
def prepare_train_dataset(samples, max_seq_len):
    sub_samples = []
    for sample in samples:
        sub_samples.append(sample)
        if len(sub_samples) == num_train_data:
            break
    dataset = CollieDatasetForTraining([
        {
            'tokens': torch.tensor(sample['input_ids'])[:max_seq_len],
            'output': torch.tensor(sample['labels'])[:max_seq_len],
            'attention_mask': torch.tensor(sample['attention_mask'])[:max_seq_len],
        } for sample in sub_samples
    ])
    return dataset

def prepare_eval_dataset(samples, eval_context_len, eval_predict_len):
    sub_samples = []
    for sample in samples:
        sample['labels'] = deepcopy(sample['input_ids'])
        if eval_context_len > 0:
            sample['labels'][:eval_context_len] = [-100 for _ in range(eval_context_len)]
        assert len(sample['labels']) == len(sample['input_ids'])
        sub_samples.append(sample)
        if len(sub_samples) == num_eval_data:
            break
    dataset = CollieDatasetForTraining([
        {
            'tokens': torch.tensor(sample['input_ids'])[:eval_context_len+eval_predict_len],
            'output': torch.tensor(sample['labels'])[:eval_context_len+eval_predict_len],
            'attention_mask': torch.ones(len(sample['input_ids'])).long()[:eval_context_len+eval_predict_len],
        } for sample in sub_samples
    ])
    return dataset

train_dataset = prepare_train_dataset(datasets.load_from_disk(train_data_path), max_train_len)
github_eval_dataset = prepare_eval_dataset(datasets.load_dataset(eval_data_paths[0])['train'], eval_context_len, eval_predict_len)
arxiv_eval_dataset = prepare_eval_dataset(datasets.load_dataset(eval_data_paths[1])['train'], eval_context_len, eval_predict_len)

print('example data: {}'.format(tokenizer.decode(train_dataset[0]['input_ids'])))

# 5. 加载预训练模型
model = LlamaForCausalLM.from_pretrained(pretrained_model, config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--compress_type", type=str, choices=['parallel_fuse', 'pipeline_fuse', 'h2o', 'parallel_sparse', 'local_window', 'no_compress', 'streaming', 'random_prune'])
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
args = parser.parse_args()
 
compress_type = args.compress_type
if compress_type == 'parallel_fuse':
    mem_perceiver = ParallelMemPerceiver.from_config(config, model)
elif compress_type == 'pipeline_fuse':
    mem_perceiver = MemPerceiver.from_config(config, model)
elif compress_type == 'h2o':
    mem_perceiver = H2oPruner.from_config(config, model)
elif compress_type == 'streaming':
    mem_perceiver = StreamingLMPruner.from_config(config, model)
elif compress_type == 'random_prune':
    mem_perceiver = RandomPruner.from_config(config, model) # no compress
elif compress_type == 'parallel_sparse':
    mem_perceiver = SparseParallelPerceiver.from_config(config, model)
elif compress_type == 'local_window': # remove context
    config.mem_perceiver_config['query_len'] = 0
    mem_perceiver = H2oPruner.from_config(config, model)
elif compress_type == 'no_compress':
    mem_perceiver = model # no compress
else:
    raise NotImplementedError
model_for_training = mem_perceiver

# 6. 设置优化器
if args.do_train:
    perceiver_parameters = []
    for n,p in model_for_training.named_parameters():
        if 'model.' not in n:
            print(n)
            perceiver_parameters.append(p)
    optimizer = torch.optim.Adam(perceiver_parameters, lr=2e-5)
else:
    optimizer = None

# 7. 添加监视器
monitors = [
    StepTimeMonitor(config),
    TGSMonitor(config),
    MemoryMonitor(config),
    LossMonitor(config),
    EvalMonitor(config)
]

# 8. 添加Evaluator
evaluator_ppl = EvaluatorForPerplexity(
    model = model_for_training,
    config = config,
    dataset = eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'ppl': PPLMetric()
    }
)
evaluator_decode = EvaluatorForGeneration(
    model = model_for_training,
    config = config,
    tokenizer = tokenizer,
    dataset = eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'decode': DecodeMetric()
    }

)

# 9. 实例化trainer
trainer = Trainer(
    model = model_for_training,
    config = config,
    loss_fn = GPTLMLoss(-100),
    optimizer = optimizer,
    train_dataset = train_dataset,
    monitors = monitors,
    evaluators = [evaluator_ppl]
)

if args.do_train:
    # 10. 训练/验证
    trainer.train()

if args.do_eval:
    trainer.eval()

#  Command CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 finetune_moss_for_training.py