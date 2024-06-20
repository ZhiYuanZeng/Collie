from typing import Tuple, Optional
from torch import Tensor
import torch.nn as nn
import torch
import math
from torch.nn.modules.activation import _arg_requires_grad, _check_arg_device, _is_make_fx_tracing
from torch.nn.functional import scaled_dot_product_attention, linear, pad, softmax, _in_projection_packed, _none_or_dtype, _canonical_mask, has_torch_function, handle_torch_function, dropout, _mha_shape_check
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np

def gradient_hook(grad, param_name="UnKnown"):
    print(f"Param_name: [{param_name}] Gradient: {grad}", flush=True)

def visualize_prune(counters, figure_name, seq_len, chunk_size):
    # chunk_size is used for down_sampling
    v_matrix = np.zeros([len(counters), seq_len // chunk_size])
    for i,layer_counter in enumerate(counters):
        for k,v in layer_counter.items():
            v_matrix[i, int(k)//chunk_size] += int(v)
    print(v_matrix.shape)
    v_matrix = v_matrix/(v_matrix.sum(axis=-1, keepdims=True)+1e-6)
    print(v_matrix)
    visualize_matrix(v_matrix, figure_name)

def visualize_matrix(matrix, figure_name):
    # 创建图形和轴
    fig, ax = plt.subplots()

    # 使用 imshow 可视化矩阵
    cax = ax.imshow(matrix, cmap='viridis')

    # 添加颜色条
    fig.colorbar(cax)

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

    # 显示图形
    plt.savefig(f'imgs/{figure_name}')
    plt.close()

class IncrementalType:
    LINEAR="linear"
    SQUARE="square"
    SQRT="sqrt"
    INVERSE_CONVEX="inverse_convex"
    INVERSE_CONCAVE="inverse_concave"
    all_types = (LINEAR, SQUARE, SQRT, INVERSE_CONVEX, INVERSE_CONCAVE)

class LayerwiseIncrementalType:
    SQUARE_SQRT="square_sqrt"
    DOUBLE_INVERSE="double_inverse"
    ADAPTIVE="adaptive"
    SMOOTH_ADAPTIVE="smooth_adaptive"
    all_types = (SQUARE_SQRT, DOUBLE_INVERSE, ADAPTIVE, SMOOTH_ADAPTIVE)
    

class MemSizeScheduler:
    def __init__(self, chunk_size, max_mem_size, incremental_type=None, num_layers=None) -> None:
        self.chunk_size = chunk_size
        self.max_mem_size = max_mem_size
        self.incremental_type = incremental_type
        self.num_layers = num_layers
    
    def _get_num_chunks(self, seq_len):
        if seq_len % self.chunk_size == 0:
            num_chunks = seq_len // self.chunk_size
        else:
            num_chunks = seq_len // self.chunk_size + 1
        return num_chunks
    
    def get_incremental_memsize(self, step, seq_len, incremental_type=None):
        if incremental_type is None:
            incremental_type = self.incremental_type
        num_chunks = self._get_num_chunks(seq_len=seq_len)
        assert self.max_mem_size is not None
        if step >= (num_chunks-1):
            return self.max_mem_size

        avg_mem_size_per_chunk = self.max_mem_size // num_chunks
        if incremental_type == IncrementalType.LINEAR:
            incremental_len = avg_mem_size_per_chunk * (step+1)
        elif incremental_type == IncrementalType.SQUARE:
            incremental_len = (step ** 2) * ((self.max_mem_size-avg_mem_size_per_chunk) / (num_chunks-1) ** 2) + avg_mem_size_per_chunk
        elif incremental_type == IncrementalType.SQRT:
            incremental_len = (step ** 0.5) * ((self.max_mem_size-avg_mem_size_per_chunk) / (num_chunks-1) ** 0.5) + avg_mem_size_per_chunk
        elif incremental_type == IncrementalType.INVERSE_CONVEX:
            incremental_len = self.max_mem_size / (num_chunks -  step)
        elif incremental_type == IncrementalType.INVERSE_CONCAVE:
            incremental_len = self.max_mem_size + avg_mem_size_per_chunk - self.max_mem_size / (step + 1)
        else:
            raise NotImplementedError(incremental_type)
        
        return int(incremental_len)
    
    def get_layerwise_incremental_memsize(self, step, seq_len, layer_idx, keep_memory_rate):
        num_chunks = self._get_num_chunks(seq_len=seq_len)
        avg_mem_size_per_chunk = self.max_mem_size // num_chunks

        if step+1 == num_chunks:
            return self.max_mem_size
        assert self.num_layers is not None and layer_idx < self.num_layers, f'{self.num_layers=}, {layer_idx=}'
        if self.incremental_type == LayerwiseIncrementalType.SQUARE_SQRT:
            if layer_idx < self.num_layers // 2:
                return self.get_incremental_memsize(step, seq_len, incremental_type=IncrementalType.SQUARE)
            else:
                return self.get_incremental_memsize(step, seq_len, incremental_type=IncrementalType.SQRT)
        elif self.incremental_type == LayerwiseIncrementalType.DOUBLE_INVERSE:
            if layer_idx < self.num_layers // 2:
                return self.get_incremental_memsize(step, seq_len, incremental_type=IncrementalType.INVERSE_CONVEX)
            else:
                return self.get_incremental_memsize(step, seq_len, incremental_type=IncrementalType.INVERSE_CONCAVE)
        elif self.incremental_type == LayerwiseIncrementalType.ADAPTIVE:
            assert keep_memory_rate is not None
            if step == 0:
                incremental_len = avg_mem_size_per_chunk
            elif step == 1:
                incremental_len = 2 * avg_mem_size_per_chunk
            else:
                keep_rate = [r['avg'] for r in keep_memory_rate] 
                normalized_keep_rate = torch.tensor(keep_rate).softmax(dim=-1)
                # print(keep_rate, normalized_keep_rate, flush=True)
                incremental_len = (step+1) * avg_mem_size_per_chunk * self.num_layers * normalized_keep_rate[layer_idx].item()
                incremental_len = min(incremental_len, self.max_mem_size)
            return int(incremental_len)
    
    def get_fix_memsize(self):
        return self.max_mem_size
    
    def get_fix_incremental_memsize(self, step, seq_len):
        def split_list_lengths(n, k):
            base_length = n // k
            extra = n % k
            
            lengths = [base_length + 1 if i < extra else base_length for i in range(k)]
            return lengths
        
        num_chunks = self._get_num_chunks(seq_len)
        memsizes = split_list_lengths(self.max_mem_size, num_chunks)
        assert len(memsizes) == num_chunks
        if step >= num_chunks:
            return memsizes[-1]
        else:
            return memsizes[step]

    def get_memsize(self, step=None, seq_len=None, layer_idx=None, keep_memory_rate=None, min_chunk_size=1024, min_mem_size=256):
        if self.incremental_type is None:
            return self.get_fix_memsize()
        elif self.incremental_type in IncrementalType.all_types:
            return self.get_incremental_memsize(step, seq_len)
        elif self.incremental_type in LayerwiseIncrementalType.all_types:
            return self.get_layerwise_incremental_memsize(step, seq_len, layer_idx, keep_memory_rate)
        else:
            raise NotImplementedError
        # elif self.incremental_type == 'decremental_chunk':
        #     return self.get_increment_memsize_according_to_max_and_mean(seq_len, step, min_chunk_size, min_mem_size)

class ChunkSizeScheduler:
    def __init__(self, ref_chunk_size) -> None:
        self.ref_chunk_size = ref_chunk_size

    def divide_list_max_length(self, n, k):
        # 计算总共需要多少个子列表
        num_sublists = (n + k - 1) // k
        
        # 创建每个子列表的长度
        sublist_lengths = [k] * num_sublists
        
        # 处理最后一个子列表的长度
        if n % k != 0:
            sublist_lengths[-1] = n % k
        
        return sublist_lengths 

    def get_std_chunk_sizes(self, seq_len):
        return self.divide_list_max_length(seq_len, self.ref_chunk_size)

    def get_decremental_chunk_sizes(self, seq_len, mem_size_shechuler:MemSizeScheduler):
        num_chunks = mem_size_shechuler._get_num_chunks(seq_len)
        avg_chunk_size = seq_len // num_chunks
        if num_chunks == 1:
            return avg_chunk_size
        assert self.ref_chunk_size == mem_size_shechuler.chunk_size
        mem_sizes = [mem_size_shechuler.get_memsize(i, seq_len) for i in range(num_chunks)]
        avg_mem_size = sum(mem_sizes[:-1]) // (len(mem_sizes)-1)
        avg_chunk_size = seq_len // num_chunks
        draft_chunk_sizes = [avg_chunk_size,] + [avg_chunk_size + avg_mem_size - mem_sizes[i-1] for i in range(1, num_chunks)]
        draft_chunk_sizes[-1] = seq_len-sum(draft_chunk_sizes[:-1])
        # print(mem_sizes, flush=True)
        # print(draft_chunk_sizes, flush=True)
        return draft_chunk_sizes

class ReviewScheduler():
    def __init__(self, scheduler = None, review_times=0):
        self.scheduler = scheduler
        self.review_times = review_times
    
    def repeat_first_chunk(self, lst, positions):
        first_chunk = lst[0]
        new_lst = lst[:]
        for pos in sorted(positions, reverse=True):
            if pos <= len(new_lst):
                new_lst.insert(pos, first_chunk)
            else:
                new_lst.append(first_chunk)
        return new_lst

    def track_insert_positions(self, lst_length):
        positions = self.get_review_positions(lst_length)
        insert_positions = []
        
        for i, pos in enumerate(sorted(positions)):
            actual_position = pos + i
            if actual_position <= lst_length:
                insert_positions.append(actual_position)
            else:
                insert_positions.append(lst_length - 1)
        # print(f'{positions=}, {insert_positions=}, {lst_length=}', flush=True)

        return insert_positions

    @staticmethod
    def uniform_positions(lst_length, num_elements):
        """
        确定均匀插入元素的位置
        :param lst_length: 原始列表的长度
        :param num_elements: 需要插入的元素数量
        :return: 插入位置的列表
        """
        interval = lst_length / (num_elements + 1)
        positions = [int(interval * (i + 1)) for i in range(num_elements)]
    
        return positions

    @staticmethod
    def exp_positions(lst_length, num_elements):
        """
        确定插入点之间的间隔/插入点和开始结尾的间隔呈指数增长的位置
        :param lst_length: 原始列表的长度
        :param num_elements: 需要插入的元素数量
        :return: 插入位置的列表
        """
        # 确定指数增长的基数
        base = np.exp(np.log(lst_length - 1) / num_elements)

        positions = [0] * num_elements
        for i in range(num_elements):
            positions[i] = int(base ** (i + 1))
            if positions[i] >= lst_length:
                positions[i] = lst_length - 1
        
        # 去除重复和超出范围的插入点
        positions = sorted(set(positions))
        
        return positions
    
    def get_review_positions(self, num_chunks):
        assert self.review_times > 0 or num_chunks > self.review_times

        if self.scheduler == 'uniform':
            return ReviewScheduler.uniform_positions(num_chunks, self.review_times)
        elif self.scheduler == 'exp':
            return ReviewScheduler.exp_positions(num_chunks, self.review_times)
        else:
            raise NotImplementedError

    def review(self, chunks):
        if self.review_times == 0 or self.scheduler is None:
            return chunks
        positions = self.get_review_positions(len(chunks))
        new_chunks = self.repeat_first_chunk(chunks, positions)
        print("before review, the number of chunks: {} => {}".format(len(chunks), len(new_chunks)))
        return new_chunks

class MemoryStateManager():
    def __init__(self, num_layers, chunk_size, num_sink_tokens, review_scheduler) -> None:
        self.chunk_size = chunk_size
        self.num_sink_tokens = num_sink_tokens
        self.num_layers = num_layers
        self.states = {
            'memory_rention': MemoryRention(chunk_size),
            'memory_distrbution': MemoryDistribution(chunk_size),
            'memory_forgive': MemoryForgive(chunk_size, review_scheduler),
            'memory_usage': MemoryUsage(chunk_size)
        }
        self.memory_indices = [None for _ in range(self.num_layers)]
        self.review_scheduler = review_scheduler

    @property
    def memory_rention(self):
        if 'memory_rention' in self.states:
            states = self.states['memory_rention'].states
            return self.states['memory_rention'].extract_output(states)
        else:
            return None

    def update_memory_indices(self, seq_len, topk_indices, chunk_step, layer_idx):
        bsz, num_heads, _ = topk_indices.shape
        if self.memory_indices[layer_idx] is not None:
            new_chunk_size = seq_len - self.memory_indices[layer_idx].shape[-1]
            new_chunk_indices = self.chunk_size * chunk_step + \
                torch.arange(new_chunk_size, device=topk_indices.device).view(1,1,new_chunk_size).expand(bsz, num_heads, new_chunk_size)
            global_indices = torch.cat([self.memory_indices[layer_idx], new_chunk_indices], dim=-1)
        else:
            new_chunk_indices = self.num_sink_tokens + torch.arange(seq_len, device=topk_indices.device).view(1,1,seq_len).expand(bsz, num_heads, seq_len)
            global_indices = new_chunk_indices
        assert global_indices.shape[-1] == seq_len

        self.memory_indices[layer_idx] = torch.gather(global_indices, dim=-1, index=topk_indices)
    
    def update_state(self, seq_len, prefill_len, topk_indices, chunk_step, layer_idx, attention_scores):
        if chunk_step == 0:
            self.memory_indices = [None for _ in range(self.num_layers)]
        self.update_memory_indices(seq_len, topk_indices, chunk_step, layer_idx)
        for k, state in self.states.items():
            state.update(self.memory_indices[layer_idx], seq_len, prefill_len, topk_indices, chunk_step, layer_idx, attention_scores)

    def report_state(self):
        for k, state in self.states.items():
            num_placehold = 20
            print('#'* num_placehold + 'memory retention ratio' + '#'*num_placehold)
            state.report()

    def clear_state(self):
        raise NotImplementedError        


class MemoryState:
    def __init__(self, chunk_size) -> None:
        self.states = defaultdict(lambda: None)
        self.chunk_size = chunk_size
    
    def update(self, memory_indices, seq_len, prefill_len, topk_indices, chunk_step, layer_idx, attention_scores):
        raise NotImplementedError
    
    def report(self):
        raise NotImplementedError
    
    def _update_move_avg(self, avg_states, k, v):
        if avg_states.get(k, None) is None:
            avg_states[k] = {'avg': 0, 'count': 0}

        # 更新计数和总和
        avg_states[k]['avg'] = (avg_states[k]['avg'] * avg_states[k]['count'] + v) / (avg_states[k]['count'] + 1)
        avg_states[k]['count'] += 1
    
    def extract_output(self, values):
        assert isinstance(values, dict)
        return dict([(k,v['avg']) for k,v in values.items()])
    
    def _get_num_chunks(self, seq_len):
        if seq_len % self.chunk_size == 0:
            num_chunks = seq_len // self.chunk_size
        else:
            num_chunks = seq_len // self.chunk_size + 1
        return num_chunks



class MemoryRention(MemoryState):
    def update(self, memory_indices, seq_len, prefill_len, topk_indices, chunk_step, layer_idx, attention_scores):
        if memory_indices is None:
            return
        else:
            mem_size = memory_indices.shape[-1]
            temp_retention_ratio = torch.sum(topk_indices < mem_size) / torch.numel(topk_indices)
            temp_retention_ratio = temp_retention_ratio.item()
            self._update_move_avg(self.states, f'{layer_idx=}', temp_retention_ratio) # average on setp
            self._update_move_avg(self.states, f'{chunk_step=}', temp_retention_ratio) # average on layer                              

    def report(self):
        layer_memory_retention_ratio = {}
        step_memory_retention_ratio = {}
        for k,v in self.states.items():
            if 'layer' in k:
                layer_memory_retention_ratio[k] = v
        for k,v in self.states.items():
            if 'step' in k:
                step_memory_retention_ratio[k] = v
        print('average keep memory rate of layers:', self.extract_output(layer_memory_retention_ratio))
        print('average keep memory rate of steps:', self.extract_output(step_memory_retention_ratio))

class MemoryUsage(MemoryState):
    def update(self, memory_indices, seq_len, prefill_len, topk_indices, chunk_step, layer_idx, attention_scores):
        if memory_indices is None:
            return
        memory_size = memory_indices.shape[-1]
        # attention shape: bsz, num_heads, seq_len, seq_len
        if attention_scores is None:
            return
        chunk_size = seq_len - memory_size
        valid_attention_scores = attention_scores[:, :, -chunk_size:]
        avg_attention_scores = valid_attention_scores.mean(dim=2) # (bsz, num_heads, seq_len)
        memory_attention_scores = avg_attention_scores[:, :, :memory_size]
        num_valid_memory_kv = torch.sum(memory_attention_scores > attention_scores.mean())/memory_attention_scores.numel()
        valid_memory_score = memory_attention_scores.mean()
        self._update_move_avg(self.states, f'num_{chunk_step}', num_valid_memory_kv.item())
        self._update_move_avg(self.states, f'score_{chunk_step}', valid_memory_score.item())
    
    def report(self):
        valid_num = {}
        valid_value = {}
        for k,v in self.states.items():
            if 'num' in k:
                valid_num[k] = v
        for k,v in self.states.items():
            if 'score' in k:
                valid_value[k] = v
        print('memory usage (num valid):', self.extract_output(valid_num))
        print('memory usage (mean attn score):', self.extract_output(valid_value))

    
class MemoryForgive(MemoryState):
    def __init__(self, chunk_size, review_scheduler:ReviewScheduler) -> None:
        super().__init__(chunk_size)
        self.states = {}
        self.review_scheduler = review_scheduler

    def update(self, memory_indices, seq_len, prefill_len, topk_indices, chunk_step, layer_idx, attention_scores):
        if memory_indices is not None:
            # report the number of kv of the first chunk remained in the memory while iteration
            num_chunks = self._get_num_chunks(prefill_len)
            review_positions = self.review_scheduler.track_insert_positions(num_chunks)
            review_positions = [0,] + review_positions
            # print(f'{review_positions=}', flush=True)
            # print(torch.unique(memory_indices // self.chunk_size))
            # print(review_positions)
            is_kv_from_first_chunk = torch.isin(memory_indices // self.chunk_size, torch.tensor(review_positions).type_as(memory_indices))
            
            first_chunk_keep_rate = is_kv_from_first_chunk.float().mean()
            self._update_move_avg(self.states, k=f'{chunk_step=}', v=first_chunk_keep_rate.item())
    
    def report(self):
        print('forgive ratio of first chunks and review chunks:', self.extract_output(self.states))

class MemoryDistribution(MemoryState):
    def update(self, memory_indices, seq_len, prefill_len, topk_indices, chunk_step, layer_idx, attention_scores):
        layer_key = f'layer_{layer_idx}'
        step_key = f'step_{chunk_step}'
        num_chunks = self._get_num_chunks(prefill_len)
        if self.states[layer_key] is None:
            self.states[layer_key] = Counter(memory_indices.view(-1).tolist())
        elif chunk_step == num_chunks-1:
            # 平均所有steps的memory distribution好像不合理，好像应该取最后一个step的才合理
            self.states[layer_key] += Counter(memory_indices.view(-1).tolist())
        
        if self.states[step_key] is None:
            self.states[step_key] = Counter(memory_indices.view(-1).tolist())
        else:
            self.states[step_key] += Counter(memory_indices.view(-1).tolist())

    def report(self):
        def down_sampling_distribution(dist):
            sampled_dist = defaultdict(lambda: 0)
            for k,v in dist.items():
                sampled_dist[k//self.chunk_size] += v
            return dict(sorted(sampled_dist.items()))
        
        layer_memory_distribution = {}
        step_memory_distribution = {}
        for k,v in self.states.items():
            v = down_sampling_distribution(v)
            if 'step' in k:
                step_memory_distribution[k] = v
            elif 'layer' in k:
                layer_memory_distribution[k] = v
        print('memory distribution of all layers: ', layer_memory_distribution)
        print('memory distribution of all steps: ', step_memory_distribution)

    # def update_temp_freqs(self, seq_len, topk_indices, chunk_step, layer_idx):
    #     bsz, num_heads, target_len = topk_indices.shape
    #     if self.temo_frequences[layer_idx] is None:
    #         self.temo_frequences[layer_idx] = torch.ones(target_len, device=topk_indices.device).view(1,1,-1).expand(bsz, num_heads, target_len)
    #     else:
    #         self.temo_frequences[layer_idx] = self.temo_frequences[layer_idx] + 1
    #         new_len = seq_len-self.temo_frequences[layer_idx].shape[-1]
    #         new_frequences = torch.ones(new_len, device=topk_indices.device).view(1,1,-1).expand(bsz, num_heads, new_len)
    #         new_frequences = torch.cat([self.temo_frequences[layer_idx], new_frequences], dim=-1)
    #         self.temo_frequences[layer_idx] = torch.gather(new_frequences, index=topk_indices, dim=-1)


