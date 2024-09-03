
import os
import math
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import transformers
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
import torch
from collie.models.mem_perceiver import AutoFuser, MemoryType, AutoPruner, AutoMemory, llm_dict, PrunerType
from collie import CollieConfig, env, Trainer
import json

data_path = '/remote-home/zyzeng/collie/PaulGrahamEssays.json'
with open(data_path, 'r') as f:
    data  = json.load(f)
    text_data = data['text']

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="PY007/LongMamba_16384_bs128_step400")
    parser.add_argument('--pruner_type', type=str, default="no_compress")
    parser.add_argument('--max_tokens', type=int, default=32768, help='maximum token length for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
    parser.add_argument('--all_review_times', nargs='+', type=int, default=[], help='number of review times')

    args = parser.parse_args()
    return args


def insert_multiple_times(base_string, insert_string, insertions=1):
    base_list = list(base_string)
    total_length = len(base_string)
    
    if insertions == 1:
        # 只插入一次到序列末尾
        position = total_length
        base_list[position:position] = insert_string
    else:
        # 将整个字符串均匀划分成 insertions 份
        interval = total_length / insertions
        
        positions = [round((i + 1) * interval) for i in range(insertions)]
        
        # 累计偏移量，用于调整插入位置
        offset = 0
        
        # 遍历每个插入位置，进行插入操作
        for position in positions:
            base_list[position + offset:position + offset] = insert_string
            offset += len(insert_string)
    
    new_string = ''.join(base_list)
    return new_string

def generate_prompt_landmark(n_garbage, seed, n_garbage_prefix, review_times=0, review_length=0):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    # garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    # garbage_inf = " ".join([garbage] * 5000)
    # assert len(garbage_inf) >= n_garbage
    # garbage_suffix = garbage_inf[:n_garbage_suffix]
    # garbage_prefix = garbage_inf[:n_garbage_prefix]
    assert len(text_data) >= n_garbage
    text_data_prefix = text_data[:n_garbage_prefix]
    text_data_suffix = text_data[n_garbage_prefix:n_garbage_prefix+n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."

    if review_times == 0:
        text_data_prefix = text_data_prefix + '\n' + information_line
    else:
        review_prefix = text_data_prefix[-review_length:]
        review_prefix = insert_multiple_times(review_prefix, information_line, review_times+1)
        text_data_prefix = text_data_prefix[:-review_length] + review_prefix
    final_question = "What is the pass key? The pass key is" 
    lines = [
        task_description,
        text_data_prefix,
        # information_line,
        text_data_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=60000, seed=666, review_times=0, review_length=0):
    prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix, review_times, review_length)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    generation_output = model.generate(
        input_ids=input_ids, max_length=answer_ids.shape[-1] + input_ids.shape[-1] + 1
    )
    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    model_answer = tokenizer.decode(model_answer).strip()
    gold_answer = tokenizer.decode(answer_ids[0]).strip()
    print(f'{model_answer=}, {gold_answer=}')
    is_correct = (model_answer == gold_answer)
    return is_correct, len_token


def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)


    # Load model and tokenizer
    llm_name_or_path = llm_dict[args.base_model]
    config = CollieConfig.from_pretrained(llm_name_or_path,
            trust_remote_code=True)

    chunk_size = 1024
    memory_type = MemoryType.CHUNK_STREAMING
    memory_size = 2048
    mem_perceiver_config = {
        "d_model": config.hidden_size,
        "chunk_size": chunk_size,
        "num_heads": config.num_key_value_heads,
        "num_sink_tokens": 4,
        "memory_type": memory_type,
        "num_layers": config.model_config.num_hidden_layers,
        "memory_size_limit": memory_size
    }
    setattr(config, 'mem_perceiver_config', mem_perceiver_config) 
    model = AutoMemory.from_pretrained(
        compresser_type = args.pruner_type,
        pretrained_model_name_or_path=llm_name_or_path,
        config=config,
        perceiver_path=None).cuda().to(torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name_or_path)

    all_accuries = []
    for review_times in args.all_review_times:
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * args.max_tokens // 1024 * 1024)
        # 10 diffierent n_garbage_prefix for each n_garbage that uniformly distributed
        avg_tokens = None
        for n_garbage_prefix in range(0, n_garbage, n_garbage // 10):
            # context被分成10份，分别测试不同深度
            passed_tests = 0
            total_tokens = 0
            for k in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=n_garbage, seed=k, review_times=review_times,
                                                                 review_length=n_garbage // 10) # review_length设置为1/10的context length，因为它随着context长度而线性变化
                passed_tests += is_correct
                total_tokens += len_tokens
            avg_tokens = total_tokens//args.num_tests if avg_tokens is None else avg_tokens
            accuracy = float(passed_tests)/args.num_tests
            depth = n_garbage_prefix/n_garbage
            print("accuracy on the token length %d, depth %f, is %f"%(avg_tokens,depth, accuracy))
            result = {"Review Times": review_times, "Document Depth": round(depth*100, -1),"Score": accuracy}
            all_accuries.append(result)
    df = pd.DataFrame(all_accuries)
    colors = [(0,"#F0496E"), (0.5, "#EBB839"), (1.0, "#0CD79F")]
    norm = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [(norm(v), c) for v, c in colors])

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Review Times'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Review Times", values="Score")
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        norm=norm
    )

    # More aesthetics
    plt.xlabel('Review Times')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    plt.savefig(f"imgs/heatmap_len{args.max_tokens}_review{args.all_review_times}_model{args.base_model}.png")
if __name__ == "__main__":
    args = parse_config()
    main(args)