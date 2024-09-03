import sys
sys.path.append("../../../")

from transformers import AutoTokenizer, GenerationConfig
from collie.config import CollieConfig

from collie.models.mem_perceiver import AutoPruner, PrunerType, MemoryType, llm_dict
import torch
import datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class LongDataset(Dataset):
    def __init__(self, data_path, seq_len, eval_num) -> None:
        super().__init__()
        self.samples = datasets.load_from_disk(data_path)['train']
        self.seq_len = seq_len
        self.eval_num = eval_num
    
    def __getitem__(self, index):
        return torch.tensor(self.samples['input_ids'][index][:self.seq_len]).cuda()
    
    def __len__(self):
        return min(len(self.samples), self.eval_num)

def save_kv_rank(llm_name_or_path, data_path, kv_path, seq_len=16384, batch_size=1, eval_num=32):
    # config
    config = CollieConfig.from_pretrained(llm_name_or_path,
            trust_remote_code=True)
    config.use_flash = True

    model = AutoPruner.from_pretrained(
        pruner_type="no_compress",
        config=config,
        pretrained_model_name_or_path=llm_name_or_path)

    model = model.to(torch.bfloat16)
    model = model.cuda()
    model.eval()


    dataset = LongDataset(data_path, seq_len, eval_num)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    move_avg_of_rank = None
    move_avg_of_sim = None
    count = None
    with torch.no_grad():
        for input_ids in tqdm(dataloader):
            model_outputs = model(input_ids)
            kv_cache = model_outputs.past_key_values
            kv_cache = [torch.stack([kv[0], kv[1]], dim=0) for kv in kv_cache]
            kv_cache = torch.stack(kv_cache, dim=0) # (layer, 2, bsz, ...)
            num_layers, _, bsz, seq, num_heads, head_dim = kv_cache.shape
            kv_cache = kv_cache.reshape(num_layers, 2, bsz, seq, -1)
            
            for i in range(num_layers):
                for j in range(i+1, num_layers):
                    if move_avg_of_rank is None:
                        move_avg_of_rank = torch.zeros(num_layers, num_layers, 2).to(kv_cache.device)
                        move_avg_of_sim = torch.zeros(num_layers, num_layers, 2).to(kv_cache.device)
                        count  = torch.zeros(num_layers, num_layers, 2).to(kv_cache.device)
                    rank = torch.linalg.matrix_rank((kv_cache[j] - kv_cache[i]).float()).float().mean(dim=-1)
                    cos_sim = torch.cosine_similarity(kv_cache[j], kv_cache[i], dim=-1).mean(dim=-1).mean(dim=-1)
                    print(f'{i=}, {j=}, {rank=}, {cos_sim=}')
                    move_avg_of_rank[i][j] = (move_avg_of_rank[i][j] * count[i][j] + rank) / (count[i][j]+1)
                    move_avg_of_sim[i][j] = (move_avg_of_sim[i][j] * count[i][j] + cos_sim) / (count[i][j]+1)
                    count[i][j] += 1    
    
    states = {
       'rank': move_avg_of_rank,
       'cos_sim': move_avg_of_sim,
    }
    torch.save(states, kv_path)
    return move_avg_of_rank

github_path = "./eval_datasets/github_65k_llama_tokenized"
arxiv_path = "./eval_datasets/arxiv_65k_llama_tokenized"

save_kv_rank(llm_name_or_path=llm_dict['llama2-7b'], data_path=arxiv_path, kv_path='/remote-home/zyzeng/collie/arxiv_kv.pt', batch_size=1, eval_num=32)