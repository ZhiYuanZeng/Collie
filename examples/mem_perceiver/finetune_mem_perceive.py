"""
一个使用CoLLie训练Moss的例子（使用LOMO优化器，开启ZeRO3）
"""
import sys
sys.path.append('..')
from transformers import AutoTokenizer

from collie.config import CollieConfig

from collie.data import CollieDatasetForTraining

from collie.optim.lomo import Lomo

from collie.controller.trainer import Trainer
from collie.controller.evaluator import EvaluatorForPerplexity, EvaluatorForGeneration

from collie.models import LlamaForCausalLM
from collie.models.mem_perceiver import MemPerceiver

from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, EvalMonitor
from collie.metrics import DecodeMetric, PPLMetric
from collie.module import GPTLMLoss
import datasets
from datasets import Dataset
import torch
import os

os.environ['MASTER_PORT'] = "12345"

# 1. 设置路径
# 1.1 预训练模型路径
pretrained_model = "/remote-home/share/models/open_llama_3b_v2/"

# 2. 设置配置
# 2.1 加载配置
config = CollieConfig.from_pretrained(pretrained_model, trust_remote_code=True)
config.tp_size = 1
config.dp_size = 1
config.pp_size = 1
config.train_epochs = 1
config.eval_per_n_steps = 0
config.eval_per_n_epochs = 1 
config.train_micro_batch_size = 1
config.gradient_accumulation_steps = 1
config.eval_batch_size = 1
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
config.checkpointing = False

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


# 3. 设置tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

# 4. 加载数据集
train_dataset = [
    {
        'tokens': torch.tensor(sample['input_ids']),
        'output': torch.tensor(sample['labels']),
        'attention_mask': torch.tensor(sample['attention_mask']),
    } for sample in datasets.load_from_disk("/remote-home/share/personal/zyzeng/data/demo_1k/")
]
train_dataset = CollieDatasetForTraining(train_dataset)
eval_dataset = train_dataset[:10]

# 5. 加载预训练模型
model = LlamaForCausalLM.from_pretrained(pretrained_model, config=config)
mem_perceiver = MemPerceiver.from_config(config, model)

# 6. 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


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
    model = model,
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
    model = model,
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
    model = model,
    config = config,
    loss_fn = GPTLMLoss(-100),
    optimizer = optimizer,
    train_dataset = train_dataset,
    monitors = monitors,
    evaluators = [evaluator_ppl]
)
torch.autograd.detect_anomaly(True)

# 10. 训练/验证
trainer.train()

#  Command CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29402 --nnodes=1 --nproc_per_node=4 finetune_moss_for_training.py