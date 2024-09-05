from concurrent.futures import ProcessPoolExecutor
import time

from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

class ParallelTokenizer:
    def __init__(self, tokenizer, num_processes=128):
        """
        初始化 ParallelTokenizer 类
        :param tokenizer_name: Hugging Face 预训练模型的名称或路径
        :param num_processes: 并行处理的进程数
        """
        self.tokenizer = tokenizer
        self.num_processes = num_processes

    def _encode_batch(self, texts):
        """
        私有方法，用于批量编码文本并返回 list 类型的结果
        :param texts: 文本列表
        :return: 批量编码的结果，保持为 list 格式
        """
        return self.tokenizer.batch_encode_plus(texts, padding=False, truncation=False, return_tensors=None, add_special_tokens=False)

    @timing_decorator
    def parallel_encode(self, texts):
        """
        并行处理文本的编码，返回 list 格式的结果
        :param texts: 输入的文本列表
        :return: 并行处理后合并的编码结果（list 格式）
        """
        # 将输入数据划分为多个子集，以便并行处理
        chunk_size = len(texts) // self.num_processes
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

        # 使用 ProcessPoolExecutor 进行多进程编码
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(self._encode_batch, chunks))

        # 合并所有进程的结果，保持 list 格式
        final_result = {key: sum([result[key] for result in results], []) for key in results[0].keys()}
        return final_result
    
    def decode(self, token_ids, **kwargs):
        """
        解码 token_ids 为文本，调用内部 tokenizer 的 decode 方法
        :param token_ids: 需要解码的 token ids
        :param kwargs: 其他参数
        :return: 解码后的文本
        """
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)
