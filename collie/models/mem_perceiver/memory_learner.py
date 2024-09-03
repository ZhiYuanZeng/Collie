import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collie.models.mem_perceiver.memory_dataset import Dataset, calculate_all_chunks_mean
import torch.nn.functional as F
import os
from tqdm import tqdm
from tqdm.contrib import tzip
from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType
from collie.config import CollieConfig

llm_path_dict = dict(
    llama3_1_instruct = '/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B-Instruct/',
    llama3_1 = '/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B',
    falcon_mamba = '/remote-home1/zyzeng/models/falcon-mamba',
    falcon_mamba_instruct = '/remote-home1/zyzeng/models/falcon-mamba-instruct',
    mamba = '/remote-home1/zyzeng/mamba2',
    long_mamba = '/remote-home1/zyzeng/models/mamba2-long',
)

class BaseLearner(torch.nn.Module):
    def __init__(self, llm_name_or_path, kv_path=None, template_name=None) -> None:
        super().__init__()
        self.llm_name_or_path = llm_name_or_path
        self.kv_path = kv_path
        self.model = self.load_model()
        self.template_name=template_name
    
    def load_model(self):
        raise NotImplementedError
    
    def load_kv(self, kv_path):
        if kv_path is not None and os.path.exists(kv_path) and os.path.isfile(kv_path):
            print(f'loading kv-cache from {kv_path}')
            self.kv = torch.load(kv_path)

    def save_kv(self, kv_path):
        torch.save(self.kv, kv_path)
    
    def train(self, *args, **kwargs):
        raise NotImplementedError
    
    def test(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError
    
    def clear_cache(self,):
        self.kv = None
    
    @property
    def device(self, ):
        return self.model.device
    
    def expand_tensor_along_dim0(self,tensor, n):
        # 检查第0维度的大小是否为1
        if tensor.shape[0] != 1:
            raise ValueError(f"The size of the first dimension must be 1 for expansion. The shape of tensor: {tensor.shape}")

        # 使用expand方法扩展第0个维度
        expanded_tensor = tensor.expand(n, *tensor.shape[1:])
        
        return expanded_tensor

    def forward(self, *args, **kwargs):
        if self.kv is None:
            print("Warning! The leaner has not been trained!")
        else:
            # the batch size of kv-cache may not align with the inputs
            if 'input_ids' in kwargs:
                input_ids = kwargs['input_ids']
            else:
                input_ids = args[0]
            batch_size = input_ids.shape[0]
            if self.kv[0][0].shape[0] != batch_size:
                expanded_kv = [[self.expand_tensor_along_dim0(layer_kv[0], batch_size), self.expand_tensor_along_dim0(layer_kv[1], batch_size)] for layer_kv in self.kv]
                kwargs['past_key_values'] = expanded_kv
            else:
                kwargs['past_key_values'] = self.kv

        # input_ids = args[0]
        # print('model device: {}'.format(self.device))
        # print(f'data device: {input_ids.device}')

        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if self.kv is None:
            print("Warning! The leaner has not been trained!")
        kwargs['past_key_values'] = self.kv
        return self.model.generate(*args, **kwargs)

    def update_kv(self, new_kv):
        # new_kv = torch.stack([torch.stack(item, dim=0) for item in new_kv], dim=0)
        # print(f'the shape of {new_kv[0][0].shape=}')
        # if self.kv is None:
        #     self.kv = new_kv
        # else:
            # past_key_values are tuple of tuple
        self.kv = new_kv # the new kv contains the old kv

class FullAttentionLearner(BaseLearner):
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.llm_name_or_path, trust_remote_code=True, 
                        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map='auto')
        return model

    @torch.no_grad
    def train(self, input_ids, labels):
        model_outputs = self.model(input_ids.cuda(), labels=labels.cuda(), past_key_values=self.kv)
        new_kv_cache = model_outputs.past_key_values
        self.update_kv(new_kv_cache)
        return model_outputs.loss.item()

    @torch.no_grad()
    def test(self, input_ids, labels):
        if self.kv is None:
            print("Warning! The leaner has not been trained!")
        if labels is None:
            labels = input_ids
        llm_outputs = self.model(input_ids.cuda(), labels=labels.cuda(), past_key_values=self.kv)
        return llm_outputs.loss.item()

class UseFlash():
    def __init__(self, model):
        self.model = model
        self.use_flash = None

    def __enter__(self):
        self.use_flash = self.model.model.model.layers[0].config.use_flash

        for layer in self.model.model.model.layers: # force model to not use flash attention
            layer.config.use_flash = True
    
    def __exit__(self, exc_type, exc_value, traceback):
        for layer in self.model.model.model.layers: # force model to not use flash attention
            layer.config.use_flash = self.use_flash

class CompressKVLearner(BaseLearner):
    def __init__(self, llm_name_or_path, kv_path=None, template_name=None, load_kv=False, pruner_type=None, memory_type=None, chunk_size=None, memory_size=None):
        self.pruner_type = pruner_type
        self.memory_type = memory_type
        self.chunk_size = chunk_size
        self.memory_size_limit = memory_size
        self.template_name = template_name
        self.load_kv = load_kv
        super().__init__(llm_name_or_path=llm_name_or_path, kv_path=kv_path, template_name=template_name, load_kv=load_kv)

    def load_model(self):
        config = CollieConfig.from_pretrained(self.llm_name_or_path,
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
                "memory_type": self.memory_type,
                "chunk_size": self.chunk_size,
                "num_sink_tokens": 4,
                "memory_size_limit": self.memory_size_limit,
                "decremental_chunk": False,
                "review_scheduler": None,
            }
        # print(mem_perceiver_config)
        setattr(config, 'mem_perceiver_config', mem_perceiver_config)
        mem_perceiver = AutoPruner.from_pretrained(
            pruner_type=self.pruner_type,
            config=config,
            pretrained_model_name_or_path=self.llm_name_or_path)

        mem_perceiver = mem_perceiver.to(torch.bfloat16)
        mem_perceiver = mem_perceiver.cuda()
        mem_perceiver.eval()
        return mem_perceiver

    def non_zero_avg(self, ls):
        return sum(ls) / sum([1 for x in ls if x!= 0])

    @torch.no_grad()
    def train(self, input_ids, labels):
        llm_outputs = self.model(input_ids.cuda(), labels=labels.cuda(), update_memory=True, past_key_values = self.kv)
        self.update_kv(llm_outputs.past_key_values)
        return self.non_zero_avg(llm_outputs.loss) # a list
    
    @torch.no_grad()
    def test(self, input_ids, labels):
        with UseFlash(self.model):
            if self.kv is None:
                print("Warning! The leaner has not been trained!")
            llm_outputs = self.model.model(input_ids.cuda(), past_key_values = self.kv)
            loss = self.model.estimate_loss(input_ids=input_ids.cuda(), 
                                            logits=llm_outputs.logits, 
                                            labels=labels.cuda())
            return loss.item()

    def forward(self, *args, **kwargs):
        with UseFlash(self.model):
            if self.kv is None:
                print("Warning! The leaner has not been trained!")
            else:
                # the batch size of kv-cache may not align with the inputs
                if 'input_ids' in kwargs:
                    input_ids = kwargs['input_ids']
                else:
                    input_ids = args[0]
                batch_size = input_ids.shape[0]
                if self.kv[0][0].shape[0] != batch_size:
                    expanded_kv = [[self.expand_tensor_along_dim0(layer_kv[0], batch_size), self.expand_tensor_along_dim0(layer_kv[1], batch_size)] for layer_kv in self.kv]
                    kwargs['past_key_values'] = expanded_kv
                else:
                    kwargs['past_key_values'] = self.kv
            # input_ids = args[0]
            # print('model device: {}'.format(self.device))
            # print(f'data device: {input_ids.device}')

            return self.model.model(*args, **kwargs) # we do not compress the inputs at the test time
    
    def generate(self, *args, **kwargs):
        with UseFlash(self.model):
            if self.kv is None:
                print("Warning! The leaner has not been trained!")
            kwargs['past_key_values'] = self.kv
            return self.model.model.generate(*args, **kwargs)

def estimate_avg_loss(loss_data, num_epoch):
    avg_loss = {}
    for k,v in loss_data.items():
        chunk_size = len(v) // num_epoch
        avg_loss[k] = calculate_all_chunks_mean(v, chunk_size)
    return avg_loss

def train(learner, dataset_name='OpenOrca', train_configs=None):
    train_len = train_configs['train_len']
    batch_size = train_configs['batch_size']
    save_interval = train_configs['save_interval']
    eval_len = train_configs['eval_len']
    num_epochs = train_configs['num_epochs']
    seeds = train_configs['seeds']
    dataset_name = train_configs['dataset_name']
    load_kv = train_configs['load_kv']
    all_train_loss = {}
    all_eval_loss = {}

    for seed in seeds:
        if load_kv:
            learner.load_kv()
        dataset = Dataset.get_dataset(model_name=learner.llm_name_or_path, data_name=dataset_name, train_datasize=train_len, eval_datasize=eval_len, num_epochs=1, template_name=learner.template_name, seed=seed)
        train_input_ids, train_labels, eval_input_ids, eval_labels = dataset.load_data(concat_train_eval=False)
        assert train_input_ids.shape[-1] % batch_size == 0
        batched_input_ids = train_input_ids.view(-1, 1, batch_size)
        batched_labels = train_labels.view(-1, 1, batch_size)

        all_train_loss[seed] = []
        all_eval_loss[seed] = []
        num_tokens_trained = 0
        for epoch in tqdm(range(1, num_epochs+1)):
            # evaluate without training
            for input_ids, labels in tzip(batched_input_ids, batched_labels):
                train_loss = learner.train(input_ids, labels)
                all_train_loss[seed].append(train_loss)

                # evaluate after training
                eval_loss = learner.test(eval_input_ids, eval_labels)
                all_eval_loss[seed].append(eval_loss)
                num_tokens_trained += batch_size
                if num_tokens_trained % save_interval == 0 and learner.kv_path is not None:
                    checkpoint_path = learner.kv_path + f'dataset-{dataset}-epoch-{epoch}-token{num_tokens_trained}-seed{seed}.pt'
                    print(f'saving checkpoint to {checkpoint_path}')
                    learner.save_kv(checkpoint_path)
        learner.clear_cache()

    print(f'training loss of all steps: {all_train_loss}')
    print(f'training loss of all epochs: {estimate_avg_loss(all_train_loss, num_epochs)}')
    print('-'*40)
    print(f'eval loss of all steps: {all_eval_loss}')
    print(f'eval loss of all epochs: {estimate_avg_loss(all_eval_loss, num_epochs)}')

def train_compress_kv(model_configs, train_configs):
    template_name = model_configs['template_name']
    llm_name_or_path = model_configs['llm_name_or_path']
    memory_type=model_configs['memory_type']
    chunk_size=model_configs['chunk_size']
    memory_sizes = model_configs['memory_sizes']
    pruner_types = model_configs['pruner_types']
    kv_path = model_configs['kv_path']
    
    dataset_name = train_configs['dataset_name']
    for pruner_type in pruner_types:
        for memsize in memory_sizes:
            kv_path = kv_path.format(pruner_type=pruner_type, memory_type=memory_type, chunk_size=chunk_size, memsize=memsize)
            kv_cache_prune_learner = CompressKVLearner(llm_name_or_path=llm_name_or_path, 
                                                        kv_path=kv_path,
                                                        pruner_type=pruner_type, 
                                                        memory_type=memory_type,
                                                        chunk_size=chunk_size,
                                                        memory_size=memsize, 
                                                        template_name=template_name)
            print(f'pruner type: {pruner_type}, memory size: {memsize}' + '=' * 20)
            train(kv_cache_prune_learner, dataset_name=dataset_name, train_configs=train_configs)

def train_full_attention(model_configs, train_configs):
    template_name = model_configs['template_name']
    llm_name_or_path = model_configs['llm_name_or_path']
    kv_path = model_configs['kv_path']

    full_attention_learner = FullAttentionLearner(llm_name_or_path=llm_name_or_path, kv_path=kv_path, template_name=template_name)
    dataset_name = 'OpenOrca'
    train(full_attention_learner, dataset_name, train_configs=train_configs)

if __name__ == '__main__':
    train_configs = dict(
        train_len = 128 * 1024,
        batch_size = 16 * 1024,
        save_interval = 16 * 1024,
        eval_len = 4 * 1024,
        num_epochs = 1,
        seeds = [40, 41, 42],
        dataset_name = 'OpenOrca',
        load_kv=False
    )

    compress_kv_configs = dict(
        template_name = 'llama3-1',
        llm_name_or_path = llm_path_dict['llama3_1_instruct'],
        memory_type=MemoryType.CHUNK_STREAMING,
        chunk_size=1024,
        memory_sizes = [2048],
        pruner_types = [PrunerType.CONV],
        kv_path = './kv_states/compress_kv/pruner-{pruner_type}-memory-{memory_type}-chunksize-{chunk_size}-memsize-{memsize}',
    )
    
    full_attention_config = dict(
        template_name = 'llama3-1',
        llm_name_or_path = llm_path_dict['llama3_1_instruct'],
        kv_path = './kv_states/full_attention/llama3_1_8b',
    )

    # train_full_attention(compress_kv_configs, train_configs)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    train_compress_kv(compress_kv_configs, train_configs)
    # test_mamba_attention(mamba_config, train_configs)