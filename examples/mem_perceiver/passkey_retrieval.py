import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collie.models.mem_perceiver import AutoFuser, MemoryType, AutoPruner, AutoMemory, llm_dict, PrunerType
from collie import CollieConfig, env, Trainer
import json
from numpy import random


data_path = '/remote-home/zyzeng/collie/PaulGrahamEssays.json'
with open(data_path, 'r') as f:
    data  = json.load(f)
    text_data = data['text']
# print(text_data[:100])

def insert_multiple_times(base_string, insert_string, insertions):
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

def load_data(remember_len, forget_len):
    assert remember_len + forget_len <= len(text_data)
    return text_data[:remember_len], text_data[remember_len:remember_len + forget_len]

def generate_prompt_multiple_landmark(remember_len, forget_len, seed, insertions=1):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    remember_content, forget_content = load_data(remember_len, forget_len)
    pass_key = random.randint(1, 1000)
    information_line = f"\n{{The pass key is {pass_key}. Remember it. {pass_key} is the pass key.}}\n"
    remember_content_with_info = insert_multiple_times(remember_content, information_line, insertions=insertions)
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        remember_content_with_info,
        forget_content,
        final_question,
    ]
    # print(remember_content_with_info)
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)

def passkey_retrieval_test(model, tokenizer, remember_len, forget_len, seed=555, insertions=1):
    prompt, answer = generate_prompt_multiple_landmark(remember_len, forget_len, seed, insertions)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to('cuda')

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1
    )

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    is_correct = (model_answer == answer_ids[0]).all().item()
    print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct, input_ids.shape[-1]

def evaluate(model_name, compresser_type, chunk_size, memory_type, memory_size, remember_len, forget_len, review_times):
    llm_name_or_path = llm_dict[model_name]
    config = CollieConfig.from_pretrained(llm_name_or_path,
            trust_remote_code=True)

    config.checkpointing = True
    config.use_flash = True

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
        compresser_type = compresser_type,
        pretrained_model_name_or_path=llm_name_or_path,
        config=config,
        perceiver_path=None).cuda().to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)

    num_tests = 10
    passed_tests = 0
    print('-'* 120)
    print(f'{model_name=}, {compresser_type=}, {chunk_size=}, {memory_size=}, {memory_type=}, {forget_len=}, {review_times=}')
    print('-'* 120)
    for i in range(num_tests):
        is_correct, num_tokens = passkey_retrieval_test(model, tokenizer, remember_len, forget_len, seed=i, insertions=review_times)
        passed_tests += is_correct
    print(f"Prompt has {num_tokens} tokens")
    print(f"Accuracy is {passed_tests/num_tests}")

model_names = ['llama2-7b']
compresser_types = [PrunerType.CONV]
memory_types = [MemoryType.CHUNK_STREAMING]
decremental_chunk = False
chunk_sizes = [1024]
memory_size_limits=[2048]
remember_len = 4096
forget_lens = [4096] # token nums: 1k, 2k, 4k
review_intervals = [1024]

for model_name in model_names:
    for ms in memory_size_limits:
        for chunk_size in chunk_sizes:
            for compresser_type in compresser_types:
                for memory_type in memory_types:
                    for i,forget_len in enumerate(forget_lens):
                        for interval in review_intervals:
                            if interval is not None:
                                review_times = remember_len // interval
                            else:
                                review_times = 1
                            forget_len *= 4
                            remember_len *= 4
                            evaluate(model_name=model_name, compresser_type=compresser_type, chunk_size=chunk_size, memory_type=memory_type, memory_size=ms, remember_len=remember_len, forget_len=forget_len, review_times=review_times)

# for model_name in model_names:
#     for ms in memory_size_limits:
#         for chunk_size in chunk_sizes:
#             for compresser_type in compresser_types: 
#                 for memory_type in memory_types:
#                     for i,forget_len in enumerate(forget_lens):
#                         # if review:
#                         #     intervals = [interval_base * (2 ** j) for j in range(i+1)]
#                         # else:
#                         #     intervals = None
#                         review_times = 1
#                         evaluate(model_name=model_name, compresser_type=compresser_type, chunk_size=chunk_size, memory_type=memory_type, memory_size=ms, remember_len=remember_len, forget_len=forget_len, review_times=review_times)
