import sys
sys.path.append("../../../")
sys.path.append("/remote-home/zyzeng/LongMamba")

from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM
from collie.config import CollieConfig

from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType, llm_dict
from collie.models import LlamaForCausalLM
import torch
import datasets
import torch.nn.functional as F
import numpy as np
from modeling.mamba_lm import MambaLMHeadModel
import random
import json
import math

def prepare_mtob_data(data_path="./mtob_book.json", model_name=None, num_epochs=1, train_datasize=None, eval_datasize=0):
    random.seed(42)
    with open(data_path, 'r') as f:
        text = f.read()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_ids = tokenizer.encode(text)
    segment_size = 1024
    num_segments = len(input_ids) // segment_size
    num_train_segments  = train_datasize // segment_size
    num_eval_segments = eval_datasize // segment_size

    input_ids = torch.tensor(input_ids[:num_segments * segment_size]).reshape(num_segments, segment_size)
    shuffle_indices = list(range(num_segments))
    # random.shuffle(shuffle_indices)
    input_ids = input_ids[shuffle_indices]
    
    eval_segments = input_ids[:num_eval_segments].view(-1).tolist()
    train_segments = input_ids[num_eval_segments: num_eval_segments + num_train_segments].view(-1).tolist()
    
    train_input_ids = torch.tensor(
        [train_segments * num_epochs,]
    )
    valid_input_ids = torch.tensor(
        [eval_segments]
    )
    print(f"num epochs: {num_epochs}, the size of train data: {train_input_ids.shape}, the size of valid data: {valid_input_ids.shape}")
    return torch.cat([train_input_ids, valid_input_ids], dim=1).long(), None

def extract_span_indices(lst, start_tag='[start]', end_tag='[end]'):
    spans = []
    inside_span = False
    start_index = None

    for i, item in enumerate(lst):
        if item == start_tag:
            inside_span = True
            start_index = i
        elif item == end_tag:
            inside_span = False
            spans.append((start_index, i))
        elif inside_span:
            continue

    return spans

def prepare_orca_data(data_path="/remote-home/zyzeng/openorca.jsonl", model_name=None, train_datasize=None, eval_datasize=0, num_epochs=1):
    all_qa = []
    char_count = 0
    start_header = '<|start_header_id|>'
    end_header = '<|end_header_id|>'
    end_token = '<|eot_id|>'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    with open(data_path, 'r') as f:
        for l in f:
            d = json.loads(l)
            qa = 'user: ' + d['question'] + '\n' + start_header + 'assistant' + end_header + d['response'] + end_token
            if len(qa) > 1024 * 4:
                continue

            all_qa.append(qa)
            char_count += len(qa)

            if char_count >= 1024 * 1024 * 8:
                break
        
    random.seed(42)
    random.shuffle(all_qa)
    all_qa = '\n\n'.join(all_qa)
    input_ids = tokenizer.encode(all_qa)
    response_indices = extract_span_indices(input_ids, start_tag=tokenizer.encode(end_header)[0], end_tag=tokenizer.encode(end_token)[0])
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor([-100 for _ in range(len(input_ids))])
    for indices in response_indices:
        labels[indices[0]:indices[1]+1] = input_ids[indices[0]:indices[1]+1]
        # print(tokenizer.decode(labels[indices[0]:indices[1]+1]))
        # print('-'*50)
    # print(tokenizer.decode(input_ids))

    train_input_ids = input_ids[eval_datasize:train_datasize * num_epochs + eval_datasize].unsqueeze(dim=0)
    eval_input_ids = input_ids[:eval_datasize].unsqueeze(dim=0)
    train_labels = labels[eval_datasize:train_datasize * num_epochs + eval_datasize].unsqueeze(dim=0)
    eval_labels = labels[:eval_datasize].unsqueeze(dim=0)
    assert train_input_ids.shape == train_labels.shape
    assert eval_input_ids.shape == eval_labels.shape
    print(f'the size of training data: {train_input_ids.shape[1]}, the size of eval data: {eval_input_ids.shape[1]}')
    return torch.cat([train_input_ids, eval_input_ids], dim=1), torch.cat([train_labels, eval_labels], dim=1)

def preapre_gsm8k_data(data_path="/remote-home/zyzeng/gsm_8k.jsonl", model_name=None, train_datasize=None, eval_datasize=0, num_epochs=1):
    random.seed(42)
    all_qa = []
    char_count = 0 
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    with open(data_path, 'r') as f:
        for l in f:
            d = json.loads(l)
            qa = 'Question: ' + d['question'] + '\nResponse: ' + tokenizer.bos_token + d['answer'] + tokenizer.eos_token
            all_qa.append(qa)
            char_count += len(qa)
            if char_count >= (train_datasize * num_epochs + eval_datasize) * 8:
                break
    random.shuffle(all_qa)
    all_qa = '\n\n'.join(all_qa)
    input_ids = tokenizer.encode(all_qa)
    response_indices = extract_span_indices(input_ids, start_tag=tokenizer.bos_token_id, end_tag=tokenizer.eos_token_id)
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor([-100 for _ in range(len(input_ids))])
    for indices in response_indices:
        labels[indices[0]:indices[1]+1] = input_ids[indices[0]:indices[1]+1]
    input_ids = input_ids[:(train_datasize * num_epochs + eval_datasize)].unsqueeze(dim=0)
    labels = labels[:(train_datasize * num_epochs + eval_datasize)].unsqueeze(dim=0)
    
    print(f'the size of input_ids: {input_ids.shape}, the size of labels: {labels.shape}')
    return input_ids, labels
    
def prepare_zhihu_data(data_path="/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_raw/moss_data_split/cn_zhihu/aa.jsonl", model_name=None, train_datasize=None, eval_datasize=0, num_epochs=1):
    all_data = []
    char_count = 0
    with open(data_path, 'r') as f:
        for l in f:
            d = json.loads(l)
            all_data.append(d['text'])
            char_count += len(d['text'])
            if char_count >= (train_datasize * num_epochs + eval_datasize) * 8:
                break
    random.shuffle(all_data)
    all_data = '\n\n'.join(all_data)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_ids = tokenizer.encode(all_data)
    input_ids = torch.tensor(
        [input_ids[:(train_datasize * num_epochs + eval_datasize)]]
    )
    print(f'the size of data: {input_ids.shape}')
    return input_ids

def prepare_noisy_data_for_eval(data_path, model_name, num_epochs, datasize, noisesize):
    data_size = datasize + noisesize
    with open(data_path, 'r') as f:
        text = f.read()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_ids = tokenizer.encode(text)
    valid_data = input_ids[:data_size]
    noisy_data = input_ids[-noisesize:]
    input_ids = torch.tensor(
        [valid_data * num_epochs + noisy_data + valid_data]
    )
    print(f"the size of data: {len(data_size)}, the size of noisy data: {len(noisy_data)}")
    return input_ids
    

def test_scaling_law_of_fm(llm_path, pruner_type, memory_type, chunk_size, memory_size_limit, train_datasize=1024, num_epochs=1):
    # config
    config = CollieConfig.from_pretrained(llm_path,
            trust_remote_code=True)
    config.checkpointing = True
    config.use_flash = True

    d_model=config.hidden_size // config.num_attention_heads * config.num_key_value_heads
    num_heads=config.num_key_value_heads
    num_layers=config.model_config.num_hidden_layers

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
            "decremental_chunk": False,
            "review_scheduler": None,
        }
    # print(mem_perceiver_config)
    setattr(config, 'mem_perceiver_config', mem_perceiver_config)
    mem_perceiver = AutoPruner.from_pretrained(
        pruner_type=pruner_type,
        config=config,
        pretrained_model_name_or_path=llm_path)

    mem_perceiver = mem_perceiver.to(torch.bfloat16)
    mem_perceiver = mem_perceiver.cuda()
    mem_perceiver.eval()

    print('-'*100)
    print(f'llm: {llm_path} | chunk_size: {chunk_size} | memory size limit: {memory_size_limit} | \
        memory type: {memory_type} | pruner type: {pruner_type} | train_datasize: {train_datasize}'+'='*10, flush=True)
    print('-'*100)

    input_ids, labels = prepare_mtob_data(model_name=llm_path, train_datasize=train_datasize, num_epochs=num_epochs)
    input_ids = input_ids.cuda()
    if labels is not None:
        labels = labels.cuda()

    with torch.no_grad():
        model_outputs = mem_perceiver(input_ids, labels=labels)
        loss_list = model_outputs.loss
    token_count = (np.arange(len(loss_list)) + 1) * chunk_size
    loss_scaling  = dict((tc, l) for tc, l in zip(token_count, loss_list))
    print(loss_scaling)
    return loss_scaling
    

def test_kv_prune(llm_path, pruner_type, memory_type, chunk_size, memory_size_limit, num_epochs=1, train_datasize=1024, eval_datasize=1024):
    review_scheduler = None
    # config
    config = CollieConfig.from_pretrained(llm_path,
            trust_remote_code=True)

    d_model=config.hidden_size // config.num_attention_heads * config.num_key_value_heads
    num_heads=config.num_key_value_heads
    num_layers=config.model_config.num_hidden_layers

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
            "decremental_chunk": False,
            "review_scheduler": review_scheduler,
        }
    # print(mem_perceiver_config)
    setattr(config, 'mem_perceiver_config', mem_perceiver_config)
    mem_perceiver = AutoPruner.from_pretrained(
        pruner_type=pruner_type,
        config=config,
        pretrained_model_name_or_path=llm_path)

    mem_perceiver = mem_perceiver.to(torch.bfloat16)
    mem_perceiver = mem_perceiver.cuda()
    mem_perceiver.eval()

    print('-'*100)
    print(f'llm: {llm_path} | chunk_size: {chunk_size} | memory size limit: {memory_size_limit} | \
        memory type: {memory_type} | pruner type: {pruner_type} | epochs: {num_epochs} | train_datasize: {train_datasize} | eval_datasize: {eval_datasize}'+'='*10, flush=True)
    print('-'*100)

    train_losses = []
    eval_losses = []
    for step in range(num_epochs):
        input_ids, _ = prepare_mtob_data(data_path='./mtob_book.txt',model_name=llm_path, num_epochs=step+1, train_datasize=train_datasize, eval_datasize=eval_datasize)
        input_ids = input_ids.cuda()

        with torch.no_grad():
            model_outputs = mem_perceiver(input_ids)
            loss_list = model_outputs.loss
        num_train_chunks = train_datasize * (step+1) // chunk_size
        num_eval_chunks = eval_datasize // chunk_size
        assert num_train_chunks + num_eval_chunks == len(loss_list), f"num_train_chunks: {num_train_chunks}, num_eval_chunks: {num_eval_chunks}, loss_list: {len(loss_list)}"
        train_loss = loss_list[:num_train_chunks]
        train_epochs_loss = np.array(train_loss).reshape((step+1), -1).mean(axis=1)
        eval_loss = loss_list[num_train_chunks:]
        eval_loss = sum(eval_loss) / len(eval_loss)

        train_losses.append(train_epochs_loss.tolist())
        eval_losses.append(eval_loss)
    print('training loss of each epoch: {}'.format(train_losses[-1]))
    print('eval loss of each epoch : {}'.format(eval_losses))
    return train_epochs_loss, eval_loss

def test_hf_model(llm_name_or_path, num_epochs=8, train_datasize=None, eval_datasize=None):
    with torch.no_grad():
        if 'llama' in llm_name_or_path.lower():
            llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",).cuda()
        else:
            llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, trust_remote_code=True, ).to(torch.bfloat16).cuda()
        train_losses = []
        eval_losses = []
        for i in range(num_epochs):
            input_ids, labels = prepare_orca_data(model_name=llm_name_or_path, num_epochs=(i+1), train_datasize=train_datasize, eval_datasize=eval_datasize)
            # try:
            llm_outputs = llm(input_ids = input_ids.cuda(), use_cache=False)
            # except Exception as e:
            #     print(e)
            #     print(f'early exit at epoch {i+1}')
            #     break
            
            train_loss, eval_loss = estimate_loss(input_ids, llm_outputs.logits.cpu(), i+1, train_datasize, eval_datasize, labels=labels)
            train_losses.append(train_loss)
            eval_losses.append(eval_loss)
        print('training loss of each epoch: {}'.format(train_losses[-1]))
        print('eval loss of each epoch : {}'.format(eval_losses))

def test_long_mamba(num_epochs=8, datasize=None, train_datasize=None, eval_datasize=None, llm_name_or_path='/remote-home/zyzeng/mamba2-long'):
    with torch.no_grad():
        llm = MambaLMHeadModel.from_pretrained(llm_name_or_path).to(torch.bfloat16).cuda()
        train_losses = []
        eval_losses = []

        for i in range(num_epochs):
            input_ids, _ = prepare_mtob_data(model_name="EleutherAI/gpt-neox-20b", num_epochs=i+1, train_datasize=train_datasize, eval_datasize=eval_datasize)
            input_ids = input_ids.cuda()
            try:
                llm_outputs = llm(input_ids = input_ids)
            except Exception as e:
                print(e)
                print(f'early exit at epoch {i+1}')
                break
            train_loss, eval_loss = estimate_loss(input_ids, llm_outputs.logits, i+1, train_datasize, eval_datasize)
            train_losses.append(train_loss)
            eval_losses.append(eval_loss)
        
        print('training loss of each epoch: {}'.format(train_losses[-1]))
        print('eval loss of each epoch : {}'.format(eval_losses))

def divide_list_into_chunks(lst, chunk_size):
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def calculate_all_chunks_mean(lst, chunk_size):
    chunks = divide_list_into_chunks(lst, chunk_size)
    means = []
    for chunk in chunks:
        non_zero_num = sum([1 for x in chunk if x != 0])
        if non_zero_num == 0:
            print(f'{len(lst)=}, {chunk_size=}')
            assert False
        mean = sum(chunk) / non_zero_num
        means.append(mean)
    return means

def estimate_loss(input_ids, logits, num_epochs, train_len, eval_len, labels=None, window_size=4096):
    # 使用交叉熵损失函数计算损失
    batch_size, sequence_length, vocab_size = logits.shape

    # 将 input_ids 向左偏移一位，去掉第一个 token，最后一个 token 可以填充为 0 或者其他填充值
    if labels is None:
        labels = input_ids
    
    # 对 logits 进行相应的切片，去掉最后一个预测，因为它没有对应的标签
    shifted_logits = logits[:, :-1, :].contiguous()

    # 重新计算形状，以便于交叉熵损失的计算
    shifted_logits = shifted_logits.view(-1, vocab_size)
    
    if eval_len != 0:
        eval_labels_mask = labels[:, 1:].contiguous().clone()
        eval_labels_mask[:, :-eval_len] = -100
        eval_labels_mask = eval_labels_mask.view(-1)

        # 使用交叉熵损失函数计算损失
        eval_loss = F.cross_entropy(shifted_logits, eval_labels_mask).item()
    else:
        eval_loss = 0
    
    assert num_epochs * train_len + eval_len == labels.shape[1], f'{num_epochs=}, {train_len=}, {eval_len=}, {input_ids.shape[1]=}'
    shifted_input_ids = labels[:, 1:].contiguous().clone()
    train_label_mask = torch.full_like(shifted_input_ids, fill_value=-100)
    train_label_mask[:, :-eval_len] = shifted_input_ids[:, :-eval_len]

    train_label_mask = train_label_mask.view(-1)
    loss = F.cross_entropy(shifted_logits, train_label_mask, reduction='none')
    num_windows = train_len * num_epochs // window_size
    loss_list = loss[:window_size*num_windows-1].tolist()
    train_loss = calculate_all_chunks_mean(loss_list, window_size)
    
    return train_loss, eval_loss

if __name__ == '__main__':
    # llm = llm_dict['llama3-8b']
    llm = '/remote-home/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct/'
    # llm = '/remote-home/zyzeng/mamba2'
    # llm = '/remote-home/zyzeng/mamba2-long'

    for i in range(2, 7):
        data_size = (2**i) * 1024
        test_hf_model(llm, num_epochs=1, train_datasize=data_size, eval_datasize=4096)

    # for i in range(8):
    #     data_size = (2**8) * 1024
    #     test_long_mamba(num_eochs=1, train_datasize=data_size, eval_datasize=4*1024)

    # for i in range(8):
    #     data_size = (2**i) * 1024
    #     test_kv_prune(llm_path=llm, pruner_type=PrunerType.CONV, memory_type=MemoryType.DYNAMIC_INCREMENTAL, 
    #             chunk_size=1024, memory_size_limit=2048, num_epochs=1, train_datasize=data_size, eval_datasize=4*1024)

    # test_scaling_law_of_fm(llm_path=llm, pruner_type=PrunerType.CONV, memory_type=MemoryType.CHUNK_STREAMING, chunk_size=1024, memory_size_limit=2048, train_datasize= 256 * 1024, num_epochs=3)