import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig
from collie.config import CollieConfig

from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType
import torch
import datasets
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams.update({'font.size': 15})

def prepare_data_for_eval(samples, bsz, seq_len):
    sub_samples = []
    for sample in samples:
        sub_samples.append(sample)
        if len(sub_samples) == bsz:
            break
    input_ids = torch.tensor(
        [sample['input_ids'][:seq_len] for sample in sub_samples]
    )
    return input_ids

# llm_name_or_path = "/remote-home/share/models/internlm2-7b-base-hf/"
model_name_dict = {
'llama2_7b':"/remote-home/share/models/llama_v2_hf/7b/",
'internlm2_7b':"/remote-home/share/models/models--internlm2-7b"
}
# llm_name_or_path = "/remote-home/share/storage/zyyin/moss2Huawei"

# test generating
data_path = "./eval_datasets/github_65k_llama_tokenized/"

class MeasureGPUTime():
    def __init__(self, norm=1) -> None:
        self.norm = norm

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()  # 记录开始时间

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()  # 记录结束时间
        torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        self.execution_time = self.start_event.elapsed_time(self.end_event) / 1000.0 / self.norm  # 计算执行时间（以秒为单位）
        print(f"函数的执行时间为: {self.execution_time:.6f} 秒")


def eval_speed(model_name, pruner_types, memory_type, chunk_size, memory_size_limit, input_len, generate_len):
    prune_res = {}
    for pruner_type in pruner_types:
        # config
        config = CollieConfig.from_pretrained(model_name_dict[model_name],
                trust_remote_code=True)
        config.checkpointing = True
        config.use_flash = True

        batch_size=1
        d_model=config.hidden_size // config.num_attention_heads * config.num_key_value_heads
        num_heads=config.num_key_value_heads
        num_layers=config.model_config.num_hidden_layers


        pe_config  = {'exp': False, '1d': False, 'imp': False, 'log': False, 
                'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 2048, 
                'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }
        setattr(config.model_config, 'pe_config', pe_config)
        mem_perceiver_config = {
                # llm config
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                # custom config
                "memory_type": memory_type,
                "chunk_size": chunk_size,
                "num_sink_tokens": 4,
                "memory_size_limit": memory_size_limit,
            }
        # print(mem_perceiver_config)
        setattr(config, 'mem_perceiver_config', mem_perceiver_config) 
        mem_perceiver = AutoPruner.from_pretrained(
            pruner_type=pruner_type,
            config=config,
            pretrained_model_name_or_path=model_name_dict[model_name])

        mem_perceiver = mem_perceiver.cuda()
        mem_perceiver.eval()

        mem_perceiver = mem_perceiver.to(torch.bfloat16)

        input_ids = prepare_data_for_eval(datasets.load_from_disk(data_path)['train'], batch_size, input_len).cuda()

        print(f'model_name: {model_name} | chunk_size: {chunk_size} | memory size limit: {ms} | \
            memory type: {memory_type} | pruner type: {pruner_type} | input {input_len} tokens | generate {generate_len} tokens'+'='*10, flush=True)
        
        gen_config = GenerationConfig(max_new_tokens=generate_len, min_new_tokens=generate_len, early_stopping=True, eos_token_id=2)
        model_outputs = mem_perceiver.generate(input_ids, generation_config=gen_config) # warmup GPU
        
        indices_counter = mem_perceiver.report_cached_indices()
        prune_res[pruner_type] = indices_counter
    
    # cached_indices = [c.view(-1)//chunk_size for c in cached_indices]
    visualize_prune(prune_res, 
                    figure_name=f"model{model_name}#pruner{pruner_type}#memory{memory_type}#chunk{chunk_size}#eval_len{input_len}#memorysize{memory_size_limit}.png",
                    seq_len = input_len, chunk_size = chunk_size, model_name=model_name)

def visualize_prune(counters, figure_name, seq_len, chunk_size, model_name):
    data_to_plot = {}
    matrix_data = {}
    for key,counter in counters.items():
        # chunk_size is used for down_sampling
        v_matrix = np.zeros([len(counter), seq_len // chunk_size])
        for i,layer_counter in enumerate(counter):
            for k,v in layer_counter.items():
                v_matrix[i, int(k)//chunk_size] += int(v)
        v_matrix = v_matrix/(v_matrix.sum(axis=-1, keepdims=True)+1e-6)
        matrix_data[key] = v_matrix
        data_to_plot[key] = v_matrix.mean(axis=0)
    
    np.savez(f'imgs/{model_name}.npz', **matrix_data)
    # print(v_matrix)
    # visualize_matrix(v_matrix, 'matrix@' + figure_name) # 去掉最后1列
    visual_vector(data_to_plot, 'vector@' + figure_name, model_name) # 去掉最后1列

def visual_vector(data_to_plot, figure_name, model_name):
    pruner_name_transform = {
        'conv': 'SnapKV',
        'streaming_llm': 'StreamingLLM'
    }
    # 创建x轴上的取值
    plt.grid(True)
    # 绘制柱状图
    for k,v in data_to_plot.items():
        x = np.arange(len(v))
        plt.bar(x, v, label=pruner_name_transform[k], alpha=0.7)
    plt.xlabel('Chunk')
    plt.ylabel('Proportion')
    plt.legend(loc='upper left')
    # plt.tight_layout()
    plt.title(f'Memory Distributions of {model_name}')
    plt.savefig(f'imgs/{figure_name}')
    plt.close()

    print(f'saving image to imgs/{figure_name}')

def visualize_matrix(matrix, figure_name):
    # matrix = np.flipud(matrix)

    # 创建图形和轴
    plt.figure(figsize=(10,6))
    ax = sns.heatmap(matrix, cmap="Blues", cbar_kws={'shrink': 1.0})
    # 使用 imshow 可视化矩阵
    cax = ax.imshow(matrix, cmap='Blues')

    # 添加颜色条
    fig.colorbar(cax)

    ax.set_xlabel('Chunk', fontsize=14)
    ax.set_ylabel('Layer', fontsize=14)

    # 设置刻度
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))

    # 设置刻度标签（可选）
    ax.set_xticklabels(np.arange(matrix.shape[1]))
    ax.set_yticklabels(np.arange(matrix.shape[0]))

    # 显示网格线（可选）
    ax.grid(False)

    # 显示数值（可选）
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         ax.text(j, i, matrix[i, j], ha='center', va='center', color='white')
    ax.invert_yaxis()

    # 显示图形
    plt.savefig(f'imgs/{figure_name}')
    plt.close()
    print(f'saving image to imgs/{figure_name}')

# pruner_types = [PrunerType.CONV]
# memory_types = [MemoryType.DYNAMIC_INCREMENTAL]
# chunk_sizes = [512]
# memory_size_limits=[512]
# input_lens = [16384]
# generate_lens = [1]
# for ms in memory_size_limits:
#     for chunk_size in chunk_sizes:
#         for memory_type in memory_types:
#             for pruner_type in pruner_types: 
#                 for input_len in input_lens:
#                     for generate_len in generate_lens:
#                         num_chunks = input_len // chunk_size
#                         ccs = ms // num_chunks
#                         eval_speed(pruner_type=pruner_type, memory_type=memory_type, chunk_size=chunk_size, 
#                                 compressed_chunk_size=ccs, memory_size_limit=ms, input_len=input_len, 
#                                 generate_len=generate_len)

pruner_types = [PrunerType.CONV, PrunerType.STREAMING]
memory_types = [MemoryType.CHUNK_STREAMING]
chunk_sizes = [1024]
memory_size_limits=[1024]
input_lens = [32768]
generate_lens = [1]
model_names = ["internlm2_7b", "llama2_7b"]
for model_name in model_names:
    for ms in memory_size_limits:
        for chunk_size in chunk_sizes:
            for memory_type in memory_types:
                for input_len in input_lens:
                    for generate_len in generate_lens:
                        eval_speed(model_name, pruner_types=pruner_types, memory_type=memory_type, chunk_size=chunk_size, 
                                memory_size_limit=ms, input_len=input_len, 
                                generate_len=generate_len)

