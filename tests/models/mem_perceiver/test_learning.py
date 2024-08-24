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
from parallel_tokenizer import ParallelTokenizer


class Template:
    """
    A class for generating chat template for a specific model.
    """
    def apply(self, question, answer):
        raise NotImplementedError

    def get_answer_indices(self, input_ids):
        raise NotImplementedError
    
    @classmethod
    def get_template(cls, model_name, tokenizer):
        if model_name == 'llama3-1':
            return Llama3_1Template(tokenizer)
        elif model_name == 'falcon-mamba':
            return MambaTemplate(tokenizer)
        elif model_name == 'recurrent-gemma':
            return RecurrentGemmaTemplate(tokenizer)

class Llama3_1Template():
    def __init__(self, tokenizer):
        self.start_header = '<|start_header_id|'
        self.end_header = '<|end_header_id|'
        self.end_token = '<|eot_id|>'
        self.tokenizer = tokenizer
    
    def apply(self, question, answer):
        return 'user: ' + question + '\n' + self.start_header + 'assistant' + answer + self.end_token

    def get_answer_indices(self, input_ids):
        return extract_span_indices(input_ids, start_tag=self.tokenizer.encode(self.start_header)[0], end_tag=self.tokenizer.encode(self.end_token)[0])

class MambaTemplate():
    def __init__(self, tokenizer):
        self.start_token = '<|im_start|>'
        self.end_token = '<|im_end|>'
        self.tokenizer = tokenizer

    def apply(self, question, answer):
        return 'user: ' + question + '\n' + self.start_token + 'assistant: ' + answer + self.end_token
    
    def get_answer_indices(self, input_ids):
        return extract_span_indices(input_ids, start_tag=self.tokenizer.encode(self.start_token)[0], end_tag=self.tokenizer.encode(self.end_token)[0])

class RecurrentGemmaTemplate():
    def __init__(self, tokenizer):
        self.start_token = '<start_of_turn>'
        self.end_token = '<end_of_turn>'
        self.tokenizer = tokenizer
    
    def apply(self, question, answer):
        return 'user: ' + question + '\n' + self.start_token + 'assistant: ' + answer + self.end_token

    def get_answer_indices(self, input_ids):
        # print(self.tokenizer.encode(self.start_token)[1])
        # print(self.tokenizer.encode(self.end_token)[1])
        return extract_span_indices(input_ids, start_tag=self.tokenizer.encode(self.start_token)[1], end_tag=self.tokenizer.encode(self.end_token)[1])

class Dataset():
    def __init__(self, model_name, train_datasize, eval_datasize, num_epochs, data_path=None, data_name=None, seed=None):
        self.model_name = model_name
        self.train_datasize = train_datasize
        self.eval_datasize = eval_datasize
        self.num_epochs = num_epochs
        self.data_path = data_path
        self.data_name = data_name
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        parallel_tokenizer = ParallelTokenizer(
            tokenizer=hf_tokenizer,
            num_processes=64,
            overlap_length=128,
            concat_keys=["input_ids", "attention_mask"],
        )
        self.tokenizer = parallel_tokenizer
        self.seed = seed
    
    def load_data(self,):
        raise NotImplementedError

    def read_raw_data(self, max_length=1024 * 1024 * 8):
        raise NotImplementedError

    @classmethod
    def get_dataset(cls, model_name, train_datasize, eval_datasize, num_epochs, data_path=None, data_name=None, template_name=None, seed=None):
        if data_name == 'MTOB':
            return MTOBDataset(model_name, train_datasize, eval_datasize, num_epochs, data_path, data_name, seed)
        elif data_name == 'Zhihu':
            return ZhihuDataset(model_name, train_datasize, eval_datasize, num_epochs, data_path, data_name, seed)
        elif data_name == 'OpenOrca':
            return OpenOrcaDataset(model_name, train_datasize, eval_datasize, num_epochs, data_path, data_name, template_name, seed)
        elif data_name == 'GSM8k':
            return GSM8kDataset(model_name, train_datasize, eval_datasize, num_epochs, data_path, data_name, template_name, seed)
        else:
            raise NotImplementedError
        
class PretrainDataset(Dataset):
    def load_data(self,):
        text = self.read_raw_data()
        input_ids = self.tokenizer.encode(text)
        segment_size = 4096
        num_segments = len(input_ids) // segment_size
        num_train_segments  = self.train_datasize // segment_size
        num_eval_segments = self.eval_datasize // segment_size

        input_ids = torch.tensor(input_ids[:num_segments * segment_size]).reshape(num_segments, segment_size)
        shuffle_indices = list(range(num_segments))
        random.seed(self.seed)
        random.shuffle(shuffle_indices)
        input_ids = input_ids[shuffle_indices]
        
        eval_segments = input_ids[:num_eval_segments].view(-1).tolist()
        train_segments = input_ids[num_eval_segments: num_eval_segments + num_train_segments].view(-1).tolist()
        
        train_input_ids = torch.tensor(
            [train_segments * self.num_epochs,]
        )
        valid_input_ids = torch.tensor(
            [eval_segments]
        )
        print(f"num epochs: {self.num_epochs}, the size of train data: {train_input_ids.shape}, the size of valid data: {valid_input_ids.shape}")
        return torch.cat([train_input_ids, valid_input_ids], dim=1).long(), None

class SFTDataset(Dataset):
    def __init__(self, model_name, train_datasize, eval_datasize, num_epochs, data_path=None, data_name=None, template_name=None, seed=None):
        super().__init__(model_name, train_datasize, eval_datasize, num_epochs, data_path, data_name, seed)
        self.template = Template.get_template(template_name, self.tokenizer)

    def load_data(self):
        all_qa = []
        
        all_qa = self.read_raw_data() 
        random.seed(self.seed)
        random.shuffle(all_qa)
        all_qa = '\n\n'.join(all_qa)
        input_ids = self.tokenizer.encode(all_qa)
        print('finish tokenizing')
        response_indices = self.template.get_answer_indices(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor([-100 for _ in range(len(input_ids))])
        for indices in response_indices:
            labels[indices[0]:indices[1]+1] = input_ids[indices[0]:indices[1]+1]
            # print(tokenizer.decode(labels[indices[0]:indices[1]+1]))
            # print('-'*50)
        # print(tokenizer.decode(input_ids))

        train_input_ids = input_ids[self.eval_datasize:self.train_datasize * self.num_epochs + self.eval_datasize].unsqueeze(dim=0)
        eval_input_ids = input_ids[:self.eval_datasize].unsqueeze(dim=0)
        train_labels = labels[self.eval_datasize:self.train_datasize * self.num_epochs + self.eval_datasize].unsqueeze(dim=0)
        eval_labels = labels[:self.eval_datasize].unsqueeze(dim=0)
        assert train_input_ids.shape == train_labels.shape
        assert eval_input_ids.shape == eval_labels.shape
        print(f'the size of training data: {train_input_ids.shape[1]}, the size of eval data: {eval_input_ids.shape[1]}')
        return torch.cat([train_input_ids, eval_input_ids], dim=1), torch.cat([train_labels, eval_labels], dim=1)
        
class MTOBDataset(PretrainDataset):
    """"
    A book about a low-resouce language, the number of tokens: 1594k
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_path is None:
            self.data_path = "./mtob_book.json"
        self.data_name = 'MTOB'
    
    def read_raw_data(self, max_length=1024 * 1024 * 8):
        with open(self.data_path, 'r') as f:
            text = f.read()
        # print('num tokens of MTOB:', len(self.tokenizer.encode(text)))
        return text[:max_length]

class ZhihuDataset(PretrainDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_path is None:
            self.data_path = "/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/data_raw/moss_data_split/cn_zhihu/aa.jsonl"
        self.data_name = 'Zhihu'
    
    def read_raw_data(self, max_length=1024 * 1024 * 8):
        all_data = []
        char_count = 0
        with open(self.data_path, 'r') as f:
            for l in f:
                d = json.loads(l)
                all_data.append(d['text'])
                char_count += len(d['text'])
                if char_count >= max_length:
                    break
        return '\n\n'.join(all_data)

class OpenOrcaDataset(SFTDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_path is None:
            self.data_path = "/remote-home/zyzeng/data/openorca.jsonl"
        self.data_name = 'OpenOrca'

    def read_raw_data(self, max_length=1024 * 1024 * 64):
        all_qa = []
        char_count = 0
        with open(self.data_path, 'r') as f:
            for l in f:
                d = json.loads(l)
                qa = self.template.apply(d['question'], d['response'])
                if len(qa) > 1024 * 4:
                    continue

                all_qa.append(qa)
                char_count += len(qa)

                if char_count >= max_length:
                    break
        return all_qa

class GSM8kDataset(SFTDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_path is None:
            self.data_path = "/remote-home/zyzeng/gsm_8k.jsonl"
        self.data_name = 'gsm-8k'
    
    def read_raw_data(self, max_length=1024 * 1024 * 8):
        all_qa = []
        char_count = 0
        with open(self.data_path, 'r') as f:
            for l in f:
                d = json.loads(l)
                qa = self.template.apply(d['question'], d['answer'])
                if len(qa) > 1024 * 4:
                    continue

                all_qa.append(qa)
                char_count += len(qa)

                if char_count >= max_length:
                    break
        return all_qa

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
        
def estimate_nonzero_avg(elements):
    nonzero_num = sum([1 for x in elements])
    return sum(elements) / (nonzero_num + 1e-6)

def test_kv_prune(llm_name_or_path, pruner_type, memory_type, chunk_size, memory_size_limit, num_epochs=1, train_datasize=1024, eval_datasize=1024, dataset_name=None, template_name=None, seed=None):
    # config
    config = CollieConfig.from_pretrained(llm_name_or_path,
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
            "review_scheduler": None,
        }
    # print(mem_perceiver_config)
    setattr(config, 'mem_perceiver_config', mem_perceiver_config)
    mem_perceiver = AutoPruner.from_pretrained(
        pruner_type=pruner_type,
        config=config,
        pretrained_model_name_or_path=llm_name_or_path)

    mem_perceiver = mem_perceiver.to(torch.bfloat16)
    mem_perceiver = mem_perceiver.cuda()
    mem_perceiver.eval()

    print('-'*100)
    print(f'llm: {llm_name_or_path} | chunk_size: {chunk_size} | memory size limit: {memory_size_limit} | \
        memory type: {memory_type} | pruner type: {pruner_type} | epochs: {num_epochs} | train_datasize: {train_datasize} | eval_datasize: {eval_datasize} | seed: {seed}'+'='*10, flush=True)
    print('-'*100)

    eval_losses = []
    
    for step in range(num_epochs):
        dataset = Dataset.get_dataset(model_name=llm_name_or_path, data_name=dataset_name, train_datasize=train_datasize, eval_datasize=eval_datasize, num_epochs=step+1, template_name=template_name, seed=seed)
        input_ids, labels = dataset.load_data()
        input_ids = input_ids.cuda()
        if labels is not None:
            labels = labels.cuda()
        with torch.no_grad():
            model_outputs = mem_perceiver(input_ids=input_ids, labels=labels)
            loss_list = model_outputs.loss
        
        # num_train_chunks = (input_ids.shape[1] - eval_datasize) // chunk_size
        num_eval_chunks = eval_datasize // chunk_size
        # assert num_train_chunks + num_eval_chunks == len(loss_list), f"num_train_chunks: {num_train_chunks}, num_eval_chunks: {num_eval_chunks}, loss_list: {len(loss_list)}"
        train_loss = loss_list[:-num_eval_chunks]
        # train_epochs_loss = np.array(train_loss).reshape((step+1), -1).mean(axis=1)
        eval_loss = loss_list[-num_eval_chunks:]
        eval_loss = estimate_nonzero_avg(eval_loss)

        train_loss = calculate_all_chunks_mean(train_loss, chunk_size=4)
        train_epoch_loss = np.array(train_loss).reshape(step+1, -1).mean(axis=-1).reshape(-1).tolist()
        eval_losses.append(eval_loss)
    print('training loss (interval: 4096 tokens): {}'.format(train_loss))
    print('training loss of all epochs: {}'.format(train_epoch_loss))
    print('eval loss of each epoch : {}'.format(eval_losses))
    return train_loss, train_epoch_loss, eval_losses

def test_hf_model(llm_name_or_path, num_epochs=1, train_datasize=None, eval_datasize=None, dataset_name=None, template_name=None, seed=None):
    with torch.no_grad():
        try:
            llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",).cuda()
        except Exception as e:
            print(e)
            llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
                
        eval_losses = []
        for i in range(num_epochs):
            dataset = Dataset.get_dataset(model_name=llm_name_or_path, data_name=dataset_name, train_datasize=train_datasize, eval_datasize=eval_datasize, num_epochs=i+1, template_name=template_name, seed=seed)
            input_ids, labels = dataset.load_data()
            llm_outputs = llm(input_ids = input_ids.cuda(), use_cache=False)
            
            train_loss, train_epoch_loss, eval_loss = estimate_loss(input_ids, llm_outputs.logits.cpu(), i+1, train_datasize, eval_datasize, labels=labels)
            eval_losses.append(eval_loss)
            del llm_outputs
        print('training loss of all steps (interval: 4096 tokens): {}'.format(train_loss))
        print('training loss of all epochs: {}'.format(train_epoch_loss))
        print('eval loss of each epoch : {}'.format(eval_losses))
    return train_loss, train_epoch_loss, eval_losses

def test_long_mamba(llm_name_or_path, num_epochs=8, train_datasize=None, eval_datasize=None, dataset_name=None, seed=None, template_name=None):
    with torch.no_grad():
        llm = MambaLMHeadModel.from_pretrained(llm_name_or_path).to(torch.bfloat16).cuda()
        eval_losses = []

        for i in range(num_epochs):
            dataset = Dataset.get_dataset(model_name='EleutherAI/gpt-neox-20b', data_name=dataset_name, train_datasize=train_datasize, eval_datasize=eval_datasize, num_epochs=i+1, seed=seed)
            input_ids, labels = dataset.load_data()
            llm_outputs = llm(input_ids = input_ids.cuda())
            train_loss, train_epoch_loss, eval_loss = estimate_loss(input_ids, llm_outputs.logits.cpu(), i+1, train_datasize, eval_datasize, labels=labels)
            eval_losses.append(eval_loss)
        
        print('training loss of all steps (interval 4096 tokens): {}'.format(train_loss))
        print('training loss of all epochs: {}'.format(train_epoch_loss))
        print('eval loss of each epoch : {}'.format(eval_losses))
        return train_loss, train_epoch_loss, eval_losses

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
            print(f'{len(lst)=}, {chunk_size=}, {len(chunk)=}')
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
    train_epoch_loss = np.array(train_loss).reshape(num_epochs, -1).mean(axis=-1).reshape(-1).tolist()
    return train_loss, train_epoch_loss, eval_loss

class Runer:
    def __init__(self, model_name, num_epochs, min_train_data_size, max_train_data_size, eval_data_size, dataset_name, pruner_type=None, memory_type=None, chunk_size=None, memory_size_limit=None):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.min_train_data_size = min_train_data_size
        self.max_train_data_size = max_train_data_size
        self.eval_data_size = eval_data_size
        self.dataset_name = dataset_name
        self.pruner_type = pruner_type
        self.memory_type = memory_type
        self.chunk_size = chunk_size
        self.memory_size_limit = memory_size_limit

        self.llama3_1_instruct = '/remote-home/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct/'
        self.llama3_1_instruct_70B = '/remote-home/share/models/llama3_1_hf/Meta-Llama-3.1-70B-Instruct/'
        self.llama3_1 = '/remote-home/share/models/llama3_1_hf/Meta-Llama-3.1-8B'
        self.falcon_mamba = '/remote-home/zyzeng/models/falcon-mamba'
        self.falcon_mamba_instruct = '/remote-home/zyzeng/models/falcon-mamba-instruct'
        self.mamba = '/remote-home/zyzeng/models/mamba2'
        self.long_mamba = '/remote-home/zyzeng/models/mamba2-long'
        self.recurrent_gemma_instruct = '/remote-home/zyzeng/collie/models/recurrent_gemma'


    def run(self,):
        assert self.max_train_data_size >= self.min_train_data_size
        eval_loss_dict ={}
        train_loss_dict = {}
        train_epoch_loss_dict = {}
        for seed in (40, 41, 42):
            eval_loss_list = []
            train_loss_list = []
            train_epoch_loss_list = []
            data_size = self.min_train_data_size
            while data_size <= self.max_train_data_size:
                train_loss, train_epoch_loss, eval_loss = self._run(data_size, seed)
                data_size = data_size * 2

                eval_loss_list.append(eval_loss)
                train_loss_list.append(train_loss)
                train_epoch_loss_list.append(train_epoch_loss)
                
            eval_loss_dict[seed] = eval_loss_list
            train_epoch_loss_dict[seed] = train_epoch_loss_list
            train_loss_dict[seed] = train_loss_list
        print('=' * 50)
        print('train loss of all seeds:')
        print(train_loss_dict)
        print('=' * 50)
        print('train epoch loss of all seeds:')
        print(train_epoch_loss_dict)
        print('=' * 50)
        print('eval loss of all seeds:')
        print(eval_loss_dict)
        return train_loss_dict, train_epoch_loss_dict, eval_loss_dict
    
    def _run(self, data_size, seed):
        if self.model_name == 'llama3_1_instruct':
            llm_name_or_path = self.llama3_1_instruct
            template_name = 'llama3-1'
        elif self.model_name == 'llama3_1_instruct_70B':
            llm_name_or_path = self.llama3_1_instruct_70B
            template_name = 'llama3-1'
        # elif self.model_name == 'llama3_1':
        #     llm_name_or_path = self.llama3_1
        #     template_name = None
        elif self.model_name == 'long_mamba':
            llm_name_or_path = self.long_mamba
            template_name = None
        # elif self.model_name == 'falcon_mamba':
        #     llm_name_or_path = self.falcon_mamba
        #     template_name = None # only instruct model has the chat template for sft
        elif self.model_name == 'falcon_mamba_instruct':
            llm_name_or_path = self.falcon_mamba_instruct
            template_name = 'falcon-mamba'
        elif self.model_name == 'recurrent_gemma_instruct':
            llm_name_or_path = self.recurrent_gemma_instruct
            template_name = 'recurrent-gemma'
        else:
            raise NotImplementedError 
        
        if self.pruner_type is not None and self.memory_type is not None and self.chunk_size is not None and self.memory_size_limit is not None:
            return test_kv_prune(llm_name_or_path=llm_name_or_path, num_epochs=self.num_epochs, train_datasize=data_size, 
                                 eval_datasize=self.eval_data_size, dataset_name=self.dataset_name, pruner_type=self.pruner_type, 
                                 memory_type=self.memory_type, chunk_size=self.chunk_size, memory_size_limit=self.memory_size_limit, 
                                 template_name=template_name, seed=seed)
        elif self.model_name == 'long_mamba':
            return test_long_mamba(llm_name_or_path=llm_name_or_path, num_epochs=self.num_epochs, train_datasize=data_size, eval_datasize=self.eval_data_size, 
                                   dataset_name=self.dataset_name, seed=seed, template_name=template_name)
        elif self.model_name in ('falcon_mamba_instruct', 'llama3_1_instruct', 'recurrent_gemma_instruct'):
            return test_hf_model(llm_name_or_path=llm_name_or_path, num_epochs=self.num_epochs, train_datasize=data_size, eval_datasize=self.eval_data_size,
                                 dataset_name=self.dataset_name, seed=seed, template_name=template_name)
        else:  
            raise NotImplementedError 
  
if __name__ == '__main__':
    # run full attention
    # runner = Runer(model_name='llama3_1_instruct', num_epochs=1, min_train_data_size=4*1024, max_train_data_size=64*1024, eval_data_size=4*1024, dataset_name='OpenOrca')
    # res = runner.run() 

    # run kv-cache pruning
    # runner = Runer(model_name='llama3_1_instruct', num_epochs=4, min_train_data_size=8*1024, max_train_data_size=8*1024, eval_data_size=4*1024, dataset_name='MTOB', pruner_type=PrunerType.CONV, memory_type=MemoryType.CHUNK_STREAMING, chunk_size=1024, memory_size_limit=2048)
    # res = runner.run() 

    all_res = {}
    datasets = ('OpenOrca', 'MTOB')
    pruner_types = (PrunerType.STREAMING, PrunerType.RANDOM)
    for ds in datasets:
        all_res[ds] = {}
        for pt in pruner_types:
            all_res[ds][pt] = {}
            for memory_size in (2048,):
                print(pt, ds)
                runner = Runer(model_name='llama3_1_instruct', num_epochs=1, min_train_data_size=4*1024, max_train_data_size=8*1024*1024, eval_data_size=4*1024, dataset_name=ds, pruner_type=pt, memory_type=MemoryType.CHUNK_STREAMING, chunk_size=1024, memory_size_limit=memory_size)
                res = runner.run() 
                all_res[ds][pt][memory_size] = res
    print(all_res) 

    # runner = Runer(model_name='long_mamba', num_epochs=1, min_train_data_size=4*1024, max_train_data_size=64*1024, eval_data_size=4*1024, dataset_name='MTOB')
    # res = runner.run() 