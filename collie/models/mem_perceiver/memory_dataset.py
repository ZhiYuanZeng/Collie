import torch
import json
import random
from transformers import AutoTokenizer
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
        elif model_name == 'llama2':
            return Llama2Template(tokenizer)
        elif model_name == 'falcon-mamba':
            return MambaTemplate(tokenizer)

class Llama3_1Template():
    def __init__(self, tokenizer):
        self.start_header = '<|begin_of_text|'
        self.end_header = '<|end_header_id|'
        self.end_token = '<|eot_id|>'
        self.tokenizer = tokenizer
    
    def apply(self, question, answer):
        question = self.start_header + 'user' + self.end_header + '\n\n' + question + self.end_token
        answer = self.start_header + 'assistant' + self.end_header + '\n\n' + answer + self.end_token
        return question + answer

    def get_answer_indices(self, input_ids):
        indices = extract_span_indices(input_ids, start_tag=self.tokenizer.encode(self.start_header)[0], end_tag=self.tokenizer.encode(self.end_token)[0])
        return indices[1::2]

class  Llama2Template():
    def __init__(self, tokenizer):
        self.start_question_token = '[INST]'
        self.end_question_token = '[/INST]'
        self.start_conversation_token = '<s>'
        self.end_conversation_token = '</s>'
        self.tokenizer = tokenizer
    
    def apply(self, question, answer):
        question = self.start_conversation_token + self.start_question_token + question + self.end_question_token
        answer = answer + self.end_conversation_token
        return question + answer

    def get_answer_indices(self, input_ids):
        print(input_ids[:1000])
        print(self.tokenizer.encode(self.end_question_token)[1:], self.tokenizer.encode(self.end_conversation_token)[1:])
        indices = extract_span_indices(input_ids, start_tags=self.tokenizer.encode(self.end_question_token)[1:], end_tags=self.tokenizer.encode(self.end_conversation_token)[1:])
        return indices
 
class MambaTemplate():
    def __init__(self, tokenizer):
        self.start_token = '<|im_start|>'
        self.end_token = '<|im_end|>'
        self.tokenizer = tokenizer

    def apply(self, question, answer):
        return 'user: ' + question + '\n' + self.start_token + 'assistant: ' + answer + self.end_token
    
    def get_answer_indices(self, input_ids):
        return extract_span_indices(input_ids, start_tags=self.tokenizer.encode(self.start_token), end_tags=self.tokenizer.encode(self.end_token))

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
    def load_data(self, concat_train_eval=True):
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
        if concat_train_eval:
            return torch.cat([train_input_ids, valid_input_ids], dim=1).long(), None
        else:
            return train_input_ids, train_input_ids, valid_input_ids, valid_input_ids

class SFTDataset(Dataset):
    def __init__(self, model_name, train_datasize, eval_datasize, num_epochs, data_path=None, data_name=None, template_name=None, seed=None):
        super().__init__(model_name, train_datasize, eval_datasize, num_epochs, data_path, data_name, seed)
        self.template = Template.get_template(template_name, self.tokenizer)

    def load_data(self, concat_train_eval=True):
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
            # print(self.tokenizer.decode(labels[indices[0]:indices[1]+1]))
            # print('-'*50)
        # print(tokenizer.decode(input_ids))

        train_input_ids = input_ids[self.eval_datasize : self.train_datasize + self.eval_datasize].unsqueeze(dim=0)
        train_input_ids = torch.repeat_interleave(train_input_ids, repeats=self.num_epochs, dim=1)
        eval_input_ids = input_ids[:self.eval_datasize].unsqueeze(dim=0)
        train_labels = labels[self.eval_datasize : self.train_datasize + self.eval_datasize].unsqueeze(dim=0)
        train_labels = torch.repeat_interleave(train_labels, repeats=self.num_epochs, dim=1)
        eval_labels = labels[:self.eval_datasize].unsqueeze(dim=0)
        assert train_input_ids.shape == train_labels.shape
        assert eval_input_ids.shape == eval_labels.shape
        print(f'the size of training data: {train_input_ids.shape[1]}, the size of eval data: {eval_input_ids.shape[1]}')
        if concat_train_eval:
            return torch.cat([train_input_ids, eval_input_ids], dim=1), torch.cat([train_labels, eval_labels], dim=1)
        else:
            return train_input_ids, train_labels, eval_input_ids, eval_labels
        
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