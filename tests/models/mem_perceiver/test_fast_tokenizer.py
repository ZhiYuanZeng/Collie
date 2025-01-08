from transformers import BertTokenizer

from concurrent.futures import ProcessPoolExecutor
import time
from contextlib import contextmanager

from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

class ParallelTokenizer:
    def __init__(self, tokenizer_name='bert-base-uncased', num_processes=4):
        """
        初始化 ParallelTokenizer 类
        :param tokenizer_name: Hugging Face 预训练模型的名称或路径
        :param num_processes: 并行处理的进程数
        """
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.num_processes = num_processes

    def _encode_batch(self, texts):
        """
        私有方法，用于批量编码文本并返回 list 类型的结果
        :param texts: 文本列表
        :return: 批量编码的结果，保持为 list 格式
        """
        return self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors=None)

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

@contextmanager
def time_it():
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")

# 示例用法
if __name__ == "__main__":
    texts = ["Hello world" * 200] * 10000

    # 初始化 ParallelTokenizer，使用 4 个进程
    parallel_tokenizer = ParallelTokenizer(num_processes=64)

    # 并行编码
    with time_it():
        encoded_inputs = parallel_tokenizer.parallel_encode(texts)

    # 打印编码后的结果
    # print(encoded_inputs['input_ids'])
