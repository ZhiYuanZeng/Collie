import sys
sys.path.append('..')
from transformers import AutoTokenizer

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining

from collie.optim.lomo import Lomo

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import MemPerceiver, ParallelMemPerceiver, AutoPruner, AutoFuser, PrunerType, MemoryType, PrunerLoss

from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor, get_monitor
from collie.metrics import DecodeMetric, PPLMetric
from collie.module import GPTLMLoss
import datasets
from datasets import Dataset
import torch
import os
import argparse
from collie.callbacks import CheckpointCallback
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument("--pruner_type", type=str, choices=[
                        PrunerType.CHUNK_PREFIX, PrunerType.H2O, PrunerType.LOCAL_WINDOW, PrunerType.NO_COMPRESS, 
                        PrunerType.PERCEIVER, PrunerType.RANDOM, PrunerType.STREAMING, PrunerType.TOVA, None
                    ], default=None)
parser.add_argument("--fuser_type", type=str, choices=['sparse_fuser', None], default=None)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--perceiver_path", type=str, default=None)
parser.add_argument("--memory_type", type=str, choices=[
                        MemoryType.CHUNK_STREAMING, MemoryType.DYNAMIC_INCREMENTAL, MemoryType.FIXED_INCREMENTAL,
                        MemoryType.RETRIEVE_ALL_KV, MemoryType.RETRIEVE_INCREMENTAL
                    ], default=None)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--llm_model", type=str, choices=["llama2_7b", "tiny_llama", "pangu2_6b", "internlm2_7b"], default="llama2")
parser.add_argument("--num_train_samples", type=int, default=None)
parser.add_argument("--num_eval_samples", type=int, default=None)
parser.add_argument("--chunk_size", type=int, default=None)
parser.add_argument("--query_len", type=int, default=None)
parser.add_argument("--compressed_chunk_size", type=int, default=None)
parser.add_argument("--d_query", type=int, default=None)
parser.add_argument("--use_flash", action='store_true', default=False)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()
# 1. 设置路径
# 1.1 预训练模型路径
if args.llm_model == "llama2_7b":
    pretrained_model = "/remote-home/share/models/llama_v2_hf/7b/"
elif args.llm_model == "tiny_llama":
    pretrained_model = "/remote-home/share/personal/zyzeng/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/"
elif args.llm_model == "pangu2_6b":
    pretrained_model = "imone/pangu_2_6B"
elif args.llm_model == "internlm2_7b":
    pretrained_model = "/remote-home/share/models/models--internlm2-7b"
else:
    raise NotImplementedError

# 2. 设置配置
# 2.1 加载配置
config = CollieConfig.from_pretrained(pretrained_model, trust_remote_code=True)
config.tp_size = 1
config.dp_size = args.num_gpus
config.pp_size = 1
config.train_epochs = 3
config.eval_per_n_steps = 1000
config.eval_per_n_epochs = 1
config.train_micro_batch_size = 2
config.gradient_accumulation_steps = 1
config.eval_batch_size = 1
config.use_flash = args.use_flash

if args.pruner_type is not None:
    # tensorboard_dir = f"./ds_tb_logs/llm{args.llm_model}#pruner{args.pruner_type}#memory{args.memory_type}#lr{args.lr}#chunk{args.chunk_size}#temperature{args.temperature}"
    group=f"pruner_{args.pruner_type}"
    tag = f"llm{args.llm_model}#pruner{args.pruner_type}#memory{args.memory_type}#lr{args.lr}#chunk{args.chunk_size}"
else:
    group=f"fuser_{args.fuser_type}"
    # tensorboard_dir = f"./ds_tb_logs/llm{args.llm_model}#fuser{args.fuser_type}#memory{args.memory_type}#lr{args.lr}#chunk{args.chunk_size}#temperature{args.temperature}"
    tag = f"llm{args.llm_model}#fuser{args.fuser_type}#memory{args.memory_type}#lr{args.lr}#chunk{args.chunk_size}"
config.ds_config = {
        "bf16": {
            "enabled": True
        },
        "monitor_config": {
            "enabled": True,
            "tag": tag,  # job name
            "wandb": {
                "enabled": True,
                "project": "kvcache-compress",
                "team": "zyzeng",
                "group": group,
            },
            "tensorboard": {
                "enabled": True,
                "output_path": "./ds_tb_logs/",
            },

            "csv_monitor": {
                "enabled": True,
                "output_path": "./ds_csv_logs/",
            }
        },

        # "zero_optimization": {
        #     "stage": 3,
        # }
}
config.checkpointing = True
pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
          'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
          'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }
setattr(config.model_config, 'pe_config', pe_config)


mem_perceiver_config = {
    "d_model": config.hidden_size // config.num_attention_heads * config.num_key_value_heads,
    "d_query": args.d_query,
    "chunk_size": args.chunk_size,
    "query_len": args.query_len,
    "compressed_chunk_size": args.compressed_chunk_size,
    "num_heads": config.num_attention_heads,
    "num_layers": config.num_hidden_layers,
    "memory_type": args.memory_type,
    "num_sink_tokens": 4,
    "temperature": args.temperature
}
setattr(config, 'mem_perceiver_config', mem_perceiver_config) 


# 3. 设置tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True, use_fast=False)

# 4. 加载数据集
max_train_len = 8192
eval_context_len = 0
# eval_predict_len = 4096
eval_predict_len = 16384
if 'llama' in args.llm_model:
    train_data_path = "/remote-home/share/personal/zyzeng/data/redpajama-15k-4k-llama/"
    # train_data_path = "/remote-home/share/personal/zyzeng/data/demo_1k/"
    eval_data_paths = ["./eval_datasets/github_65k_llama_tokenized", "./eval_datasets/arxiv_65k_llama_tokenized"]
else:
    train_data_path = "/remote-home/share/personal/wtshu/data/redpajama-15k-8k-internlm/train"
    eval_data_paths = ["/remote-home/share/personal/wtshu/data/redpajama-15k-8k-internlm/test/github", "/remote-home/share/personal/wtshu/data/redpajama-15k-8k-internlm/test/arxiv"]

def prepare_train_dataset(samples, num_train_data, max_seq_len):
    sub_samples = []
    for sample in samples:
        sub_samples.append(sample)
        if num_train_data is not None and len(sub_samples) == num_train_data:
            break
    dataset = CollieDatasetForTraining([
        {
            'tokens': torch.tensor(sample['input_ids'])[:max_seq_len],
            'output': torch.tensor(sample['labels'])[:max_seq_len],
            'attention_mask': torch.tensor(sample['attention_mask'])[:max_seq_len],
        } for sample in sub_samples
    ])
    return dataset

def prepare_eval_dataset(samples, num_eval_data, eval_context_len, eval_predict_len):
    sub_samples = []
    for sample in samples:
        sample['labels'] = deepcopy(sample['input_ids'])
        if eval_context_len > 0:
            sample['labels'][:eval_context_len] = [-100 for _ in range(eval_context_len)]
        assert len(sample['labels']) == len(sample['input_ids'])
        sub_samples.append(sample)
        if num_eval_data is not None and len(sub_samples) == num_eval_data:
            break
    dataset = CollieDatasetForTraining([
        {
            'tokens': torch.tensor(sample['input_ids'])[:eval_context_len+eval_predict_len],
            'output': torch.tensor(sample['labels'])[:eval_context_len+eval_predict_len],
            'attention_mask': torch.ones(len(sample['input_ids'])).long()[:eval_context_len+eval_predict_len],
        } for sample in sub_samples
    ])
    return dataset

print('loading training and eval data', flush=True)
train_dataset = prepare_train_dataset(datasets.load_from_disk(train_data_path), args.num_train_samples, max_train_len)
print('finish loading training examples', flush=True)
try:
    github_eval_dataset = prepare_eval_dataset(datasets.load_from_disk(eval_data_paths[0])['train'], args.num_eval_samples, eval_context_len, eval_predict_len)
    arxiv_eval_dataset = prepare_eval_dataset(datasets.load_from_disk(eval_data_paths[1])['train'], args.num_eval_samples, eval_context_len, eval_predict_len)
except Exception as e:
    github_eval_dataset = prepare_eval_dataset(datasets.load_from_disk(eval_data_paths[0]), args.num_eval_samples, eval_context_len, eval_predict_len)
    arxiv_eval_dataset = prepare_eval_dataset(datasets.load_from_disk(eval_data_paths[1]), args.num_eval_samples, eval_context_len, eval_predict_len)
    
# print('example data: {}'.format(tokenizer.decode(train_dataset[0]['input_ids'])))
print(f'training set size: {len(train_dataset)}, github eval set size: {len(github_eval_dataset)}, arxiv eval set size: {len(arxiv_eval_dataset)}')

# 5. 加载预训练模型
if args.pruner_type is not None:
    model_for_training = AutoPruner.from_pretrained(pruner_type=args.pruner_type, config=config, pretrained_model_name_or_path=pretrained_model, perceiver_path=args.perceiver_path)
elif args.fuser_type is not None:
    model_for_training = AutoFuser.from_pretrained(fuser_type=args.fuser_type, config=config, pretrained_model_name_or_path=pretrained_model, perceiver_path=args.perceiver_path)
else:
    raise NotImplementedError

# # 6. 设置优化器
if args.do_train:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_for_training.parameters()), lr=args.lr
    )
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
github_evaluator_ppl = EvaluatorForPerplexity(
    model = model_for_training,
    config = config,
    dataset = github_eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'github_ppl': PPLMetric()
    }
)
arxiv_evaluator_ppl = EvaluatorForPerplexity(
    model = model_for_training,
    config = config,
    dataset = arxiv_eval_dataset,
    monitors = [
        EvalMonitor(config)
    ],
    metrics = {
        'arxiv_ppl': PPLMetric()
    }
)

if args.pruner_type is not None:
    save_path = f"/remote-home/zyzeng/collie/ckpts/llm{args.llm_model}#pruner{args.pruner_type}#memory{args.memory_type}#lr{args.lr}#chunk{args.chunk_size}#temperature{args.temperature}"
elif args.fuser_type is not None:
    save_path = f"/remote-home/zyzeng/collie/ckpts/llm{args.llm_model}#fuser{args.fuser_type}#memory{args.memory_type}#lr{args.lr}#chunk{args.chunk_size}#temperature{args.temperature}"
else:
    raise RuntimeError("pruner type and fuser type can not be None at the same time")
callbacks = [
    CheckpointCallback(save_path,
        every_n_batches=1000, # 每个 epoch 保存一次
        every_n_epochs=1,
        model_only=False, # 仅保存模型权重，不保存optimzer、训练步数等断点重训信息
    )
]

# 9. 实例化trainer
trainer = Trainer(
    model = model_for_training,
    config = config,
    loss_fn = GPTLMLoss(-100),
    optimizer = optimizer,
    train_dataset = train_dataset,
    monitors = monitors,
    evaluators = [github_evaluator_ppl, arxiv_evaluator_ppl],
    callbacks=callbacks
) 

if args.do_train:
    # 10. 训练/验证
    trainer.train()

if args.do_eval:
    trainer.eval()

#  Command CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 finetune_moss_for_training.py