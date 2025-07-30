from functools import partial
import os
import time
from typing import Any, Dict, List, Union
from data_juicer.ops.base_op import Filter, Mapper
from data_juicer.utils.constant import Fields, StatsKeys
import ray
import torch
import pyarrow


from loguru import logger

def filter_batch(batch, filter_func):
    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)

@ray.remote(num_gpus=0.0) 
class Actor:
    def __init__(self, op, rank=None):
        """
        现在直接传入整个op对象，更灵活控制流程
        """
        self.op = op
        self._model_loaded = False  # 标记模型是否已加载
        self.rank = rank
        self.model = None
        self.processor = None
    
    def load_model(self):

        if self.op.use_cuda() and not self._model_loaded:
            self.model, self.processor = self.op.load_model(rank=self.rank)
            self._model_loaded = True

    def mapper_cuda(self, data):
        if not self._model_loaded:
            self.load_model()  
        data = self.op.process_single(data, self.model, self.processor)
        return data
    
    def mapper_cuda_batched(self, data):
        if not self._model_loaded:
            self.load_model() 
        data = self.op.process_batched(data, self.model, self.processor)
        return data
    
    def mapper_cpu(self, data):
        processed_data = self.op.process_single(data)
        return processed_data
    
    def filter_cuda(self, data):
        if not self._model_loaded:
            self.load_model()
        data = self.op.compute_stats_single(data, self.model, self.processor)
        keep = self.op.process_single(data)
        if keep:
            return data
        else:
            return None
    
    def filter_cpu(self, data):
        data = self.op.compute_stats_single(data)
        keep = self.op.process_single(data)
        if keep:
            return data
        else:
            return None

    
    def export_stats(self, data, export_path):
        return data.write_json(export_path, force_ascii=False)
    
