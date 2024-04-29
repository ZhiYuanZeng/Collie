import sys
sys.path.append('..')
from transformers import AutoTokenizer

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining

from collie.optim.lomo import Lomo

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import MemPerceiver, ParallelMemPerceiver, AutoPruner, AutoFuser

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
parser.add_argument("--pruner_type", type=str, choices=['parallel_fuse', 'pipeline_fuse', 'h2o', 'parallel_sparse', 'local_window', 'no_compress', 'streaming', 'random_prune', 'chunk_prefix', 'chunk_postfix', 'tova_pruner', None], default=None)
parser.add_argument("--fuser_type", type=str, choices=['sparse_fuser', None], default=None)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--perceiver_path", type=str, default=None)
parser.add_argument("--memory_type", type=str, default=None,
                    choices=[
                        'write_new_compressed_read_new_compressed',
                        'increment_compressed_read_all_compressed',
                        'update_incremental_compressed_read_all_compressed',
                        'increment_all_read_retrieved',
                        'increment_compressed_read_retrieved'
                    ])
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

# 1. 设置路径
# 1.1 预训练模型路径
pretrained_model = "/remote-home/share/personal/zyzeng/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"

# 2. 设置配置
# 2.1 加载配置
config = CollieConfig.from_pretrained(pretrained_model, trust_remote_code=True)
config.tp_size = 1
config.dp_size = 1
config.pp_size = 1
config.train_epochs = 3
config.eval_per_n_steps = 1000
config.eval_per_n_epochs = 1
config.train_micro_batch_size = 4
config.gradient_accumulation_steps = 1
config.eval_batch_size = 1
config.use_flash = True
config.ds_config = {
        "fp16": {
            "enabled": True
        },
        "tensorboard": {
            "enabled": True,
            "output_path": f"./ds_tb_logs/{args.pruner_type}",
            "job_name": f"lr{args.lr}"
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
pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
          'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 2048, 
          'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }
setattr(config.model_config, 'pe_config', pe_config)

chunk_size=512
d_model=config.hidden_size
d_query=config.hidden_size // 4
d_ffn=config.hidden_size // 2
num_heads=config.num_attention_heads
train_query_len=64
eval_query_len=64
num_layers=config.num_hidden_layers

mem_perceiver_config = {
    "d_model": d_model,
    "d_query": d_query,
    "d_ffn": d_ffn,
    "chunk_size": chunk_size,
    "query_len": train_query_len,
    "eval_query_len": eval_query_len,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "memory_type": args.memory_type,
    "num_sink_tokens": 4,
    "temperature": 0.1
}
setattr(config, 'mem_perceiver_config', mem_perceiver_config) 


# 3. 设置tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

# 4. 加载数据集
max_train_len = 8192
num_train_data = 15000 # 60M
num_eval_data = 128
eval_context_len = 0
# eval_predict_len = 4096
eval_predict_len = 16384
train_data_path = "/remote-home/share/personal/zyzeng/data/redpajama-15k-4k-llama/"
# train_data_path = "/remote-home/share/personal/zyzeng/data/demo_1k/"
eval_data_paths = ["./eval_datasets/github_65k_llama_tokenized", "./eval_datasets/arxiv_65k_llama_tokenized"]

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

print('loading training and eval data', flush=True)
train_dataset = prepare_train_dataset(datasets.load_from_disk(train_data_path), max_train_len)
print('finish loading training examples', flush=True)
github_eval_dataset = prepare_eval_dataset(datasets.load_from_disk(eval_data_paths[0])['train'], eval_context_len, eval_predict_len)
arxiv_eval_dataset = prepare_eval_dataset(datasets.load_from_disk(eval_data_paths[1])['train'], eval_context_len, eval_predict_len)

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
    perceiver_parameters = []
    for n,p in model_for_training.named_parameters():
        if 'model.' not in n:
            print(n)
            perceiver_parameters.append(p)
    optimizer = torch.optim.Adam(perceiver_parameters, lr=args.lr)
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

callbacks = [
    CheckpointCallback(f"/remote-home/zyzeng/collie/ckpts/{args.pruner_type}_lr{args.lr}_memory_{args.memory_type}",
        every_n_epochs=1, # 每个 epoch 保存一次
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