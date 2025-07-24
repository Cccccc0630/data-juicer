from __future__ import annotations

from collections import defaultdict
from concurrent import futures
import json
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import Namespace
from functools import partial
import subprocess
import time
from typing import Any, Dict, List, Literal, Optional, Union
import uuid
os.environ["RAY_DEDUP_LOGS"] = "1"
from data_juicer.core.ray_actor import ComputeActor, filter_batch
import pyarrow
import pyarrow as pa    
import ray.data as rd
import ray.data.read_api as ds
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.core.data.schema import Schema
from data_juicer.ops import Deduplicator, Filter, Mapper
from data_juicer.ops.base_op import TAGGING_OPS
from data_juicer.utils.constant import Fields
from data_juicer.utils.process_utils import calculate_np

import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


def get_abs_path(path, dataset_dir):
    full_path = os.path.abspath(os.path.join(dataset_dir, path))
    if os.path.exists(full_path):
        return full_path
    else:
        return path


def convert_to_absolute_paths(samples, dataset_dir, path_keys):
    samples = samples.to_pydict()
    for key in path_keys:
        for idx in range(len(samples[key])):
            paths = samples[key][idx]
            if isinstance(paths, str):
                samples[key][idx] = get_abs_path(paths, dataset_dir)
            elif isinstance(paths, list):
                samples[key][idx] = [
                    get_abs_path(item, dataset_dir) for item in paths
                ]
    return pyarrow.Table.from_pydict(samples)


# TODO: check path for nestdataset
def set_dataset_to_absolute_path(dataset, dataset_path, cfg):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    path_keys = []
    columns = dataset.columns()
    for key in [cfg.video_key, cfg.image_key, cfg.audio_key]:
        if key in columns:
            path_keys.append(key)
    if len(path_keys) > 0:
        dataset_dir = os.path.dirname(dataset_path)
        logger.error(f'dataset_dir: {dataset_dir}')
        dataset = dataset.map_batches(partial(convert_to_absolute_paths,
                                              dataset_dir=dataset_dir,
                                              path_keys=path_keys),
                                      batch_format='pyarrow',
                                      zero_copy_batch=True)
    return dataset


def preprocess_dataset(dataset: rd.Dataset, dataset_path, cfg) -> rd.Dataset:
    if dataset_path:
        dataset = set_dataset_to_absolute_path(dataset, dataset_path, cfg)
    return dataset


def get_num_gpus(op, op_proc):
    if not op.use_cuda():
        return 0
    proc_per_gpu = op_proc / cuda_device_count()
    return 1.0 / proc_per_gpu


# def filter_batch(batch, filter_func):
#     mask = pyarrow.array(filter_func(batch.to_pydict()))
#     return batch.filter(mask)




class RayDataset(DJDataset):

    def __init__(self,
                dataset: rd.Dataset,
                dataset_path: str = None,
                cfg: Optional[Namespace] = None) -> None:
        self.data = preprocess_dataset(dataset, dataset_path, cfg)
        self.num_proc = getattr(cfg, 'np', getattr(cfg, 'num_proc', None)) if cfg else None
        self.gpu_pg = placement_group([{"CPU": 16, "GPU": 2}], strategy="STRICT_SPREAD")
        ray.get(self.gpu_pg.ready())
        
        # 初始化MPS管理器
        # self.mps_manager = MPSManager()
        # self.mps_manager.init_mps()

    def schema(self) -> Schema:
        """Get dataset schema.

        Returns:
            Schema: Dataset schema containing column names and types
        """
        if self.data is None or self.data.columns() is None:
            raise ValueError('Dataset is empty or not initialized')

        # Get schema from Ray dataset
        _schema = self.data.schema()

        # convert schema to proper list and dict
        column_types = {
            k: Schema.map_ray_type_to_python(v)
            for k, v in zip(_schema.names, _schema.types)
        }
        return Schema(column_types=column_types, columns=column_types.keys())

    def get(self, k: int) -> List[Dict[str, Any]]:
        """Get k rows from the dataset."""
        if k < 0:
            raise ValueError(f'k must be non-negative, got {k}')

        if k == 0:
            return []

        k = min(k, self.data.count())
        return list(self.data.limit(k).take())

    def get_column(self, column: str, k: Optional[int] = None) -> List[Any]:
        """Get column values from Ray dataset.

        Args:
            column: Name of the column to retrieve
            k: Optional number of rows to return. If None, returns all rows

        Returns:
            List of values from the specified column

        Raises:
            KeyError: If column doesn't exist
            ValueError: If k is negative
        """
        if (self.data is None or self.data.columns() is None
                or column not in self.data.columns()):
            raise KeyError(f"Column '{column}' not found in dataset")

        if k is not None:
            if k < 0:
                raise ValueError(f'k must be non-negative, got {k}')
            if k == 0:
                return []
            k = min(k, self.data.count())
            return [row[column] for row in self.data.limit(k).take()]

        return [row[column] for row in self.data.take()]
    
    def process1(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]
        for op in operators:
            self._run_single_op(op)
            self.data = self.data.materialize()
        return self

    def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        from ray.data import from_arrow

        actors = {}
        actor_allocate = [0, 1, 1]  # 每个operator分配的actor数量
        # actors = []
        
        # if self.data is not None:
        #     logger.info("data type after clip:",self.data)
        
        # 初始化所有actor
        for idx, op in enumerate(operators):
            actors[op._name] = []
            
            # 数据预处理（保持不变）
            columns = self.data.columns()
            if isinstance(op, Filter) and Fields.stats not in columns:
                def process_batch_arrow(table: pyarrow.Table):
                    new_column_data = [{} for _ in range(len(table))]
                    new_table = table.append_column(Fields.stats, [new_column_data])
                    return new_table
                self.data = self.data.map_batches(process_batch_arrow, batch_format='pyarrow')
            
            if idx == 0 :
                # split
                logger.info("Video cliping")
                self.data = self.data.map_batches(
                            op.process, batch_size=1, batch_format="pyarrow", num_gpus=0
                        )
                continue
            
            # 设置batch size
            batch_size = 15 if op.is_batched_op() else 1
            
            # 资源分配
            gpu_allocate = 1 if op.use_cuda() else 0
            cpu_allocate = 1 if idx == 0 else 4
            bundle_allocate = [0, 0, 1]
            
            # 创建多个actor
            for actor_num in range(actor_allocate[idx]):
                actor = ComputeActor.options(
                    name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
                    num_gpus=gpu_allocate,
                    num_cpus=cpu_allocate,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.gpu_pg,
                        placement_group_capture_child_tasks=True
                    ),
                    placement_group_bundle_index=bundle_allocate[idx],
                    runtime_env={
                        "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"} if op.use_cuda() else {},
                    }
                ).remote(op, batch_size)
                actors[op._name].append(actor)
        operators = operators[-2:]
        # 打印所有actor信息
        for op._name, actor_list in actors.items():
            logger.info(f"Operator {op._name} 有以下actors:")
            for i, actor in enumerate(actor_list):
                logger.info(f"  Actor {i}: {actor._ray_actor_id.hex()[:6]}")


        op_indices = {op_name: 0 for op_name in actors.keys()}

        def get_next_actor(op_name):
            """轮询获取下一个actor"""
            actor_list = actors[op_name]
            idx = op_indices[op_name]
            op_indices[op_name] = (idx + 1) % len(actor_list)
            return actor_list[idx]
        
        # 准备数据批次
        input_batches = list(self.data.iter_batches(batch_size=batch_size, batch_format="pyarrow"))
        logger.info(f"Total batches generated: {len(input_batches)}")
        logger.info(f"input batches: {input_batches}")  # List of pyarrow.Table

        futures = []
        buckets = defaultdict(list)
        op_batch_counters = defaultdict(int)

        logger.info(f"初始化: futures={len(futures)} items, buckets={dict(buckets)}")

        for batch_idx, batch in enumerate(input_batches):
            logger.info(f"\n=== 处理第 {batch_idx} 个 batch ===")
            
            # 转换 batch 为 records
            if isinstance(batch, pyarrow.Table):
                records = batch.to_pylist()
                logger.info(f"转换 pyarrow.Table -> {len(records)} 条 records")
            elif isinstance(batch, list):
                records = batch
                logger.info(f"直接使用 list batch -> {len(records)} 条 records")
            else:
                raise ValueError("Unsupported batch format")

            for record_idx, record in enumerate(records):
                logger.info(f"\n--- 处理第 {record_idx} 条 record ---")
                current = record
                logger.debug(f"初始 current: {type(current)} {current}")

                for op_idx, op in enumerate(operators):
                    op_name = op._name
                    actor = get_next_actor(op_name)  # 修改点：从字典轮询获取actor
                    logger.info(f"操作 {op_idx} ({op_name}) - 输入: {type(current)}")

                    if op.is_batched_op():
                        logger.info(f"添加到 bucket[{op_idx}] (当前大小: {len(buckets[op_idx])})")
                        buckets[op_idx].append(current)

                        if len(buckets[op_idx]) >= batch_size:
                            logger.info(f"触发 flush bucket[{op_idx}] (达到 batch_size={batch_size})")
                            # 修改点：传递单个actor而不是整个列表
                            refs = self.flush_bucket(op_idx, operators, {op_name: [actor]}, buckets, op_batch_counters)
                            logger.info(f"flush 返回 {len(refs)} 个 refs: {[type(r) for r in refs]}")

                            if op_idx + 1 < len(operators):
                                next_op = operators[op_idx + 1]
                                next_op_name = next_op._name
                                if next_op.is_batched_op():
                                    logger.info(f"传递到下一个 batched op {op_idx + 1}")
                                    for ref in refs:
                                        table = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
                                        pylist = table.to_pylist() if isinstance(table, pyarrow.Table) else table
                                        logger.info(f"解包得到 {len(pylist)} 条记录")
                                        for r in pylist:
                                            buckets[op_idx + 1].append(r)
                                            if len(buckets[op_idx + 1]) >= batch_size:
                                                logger.info(f"下级 bucket 满，触发 flush")
                                                next_actor = get_next_actor(next_op_name)
                                                new_refs = self.flush_bucket(
                                                    op_idx + 1, 
                                                    operators, 
                                                    {next_op_name: [next_actor]}, 
                                                    buckets, 
                                                    op_batch_counters
                                                )
                                                futures.extend(new_refs)
                                                logger.info(f"添加 {len(new_refs)} 个 futures (来自下级 flush), 现在总数: {len(futures)}")
                                    break
                                else:
                                    current = ray.get(refs[0])
                                    logger.info(f"传递到 single op, 新 current: {type(current)}")
                            else:
                                futures.extend(refs)
                                logger.info(f"最后一级操作，添加 {len(refs)} 个 futures, 现在总数: {len(futures)}")
                                break
                        else:
                            current = None
                            logger.info("bucket 未满，跳过后续处理")
                            break
                    else:
                        logger.info("执行 single op 处理")
                        table = pyarrow.Table.from_pylist([current])
                        if isinstance(op, Filter):
                            result_table = ray.get(actor.process_and_filter.remote(table))
                        elif isinstance(op, Mapper):
                            result_table = ray.get(actor.mapper_single.remote(table))
                        else:
                            raise NotImplementedError(f"Unsupported single op: {type(op)}")
                        
                        current = result_table[0] if isinstance(result_table, list) else result_table
                        logger.info(f"single op 输出: {type(current)}")

                if current is not None and not any(op.is_batched_op() for op in operators):
                    futures.append(current)

        # 处理未满的 buckets (同样需要修改为使用字典结构)
        logger.info("\n=== 处理未满的 buckets ===")
        for op_idx in range(len(operators)):
            if buckets[op_idx]:
                op_name = operators[op_idx]._name
                actor = get_next_actor(op_name)
                logger.info(f"flush bucket[{op_idx}] (大小: {len(buckets[op_idx])})")
                refs = self.flush_bucket(op_idx, operators, {op_name: [actor]}, buckets, op_batch_counters)
                logger.info(f"得到 {len(refs)} 个 refs: {[type(r) for r in refs]}")

                for ref in refs:
                    current = ray.get(ref)
                    logger.info(f"解包 ref 得到: {type(current)}")
                    current = current.to_pylist() if isinstance(current, pyarrow.Table) else [current]
                    logger.info(f"转换为 list (长度: {len(current)})")

                    for sample_idx, sample in enumerate(current):
                        logger.info(f"处理第 {sample_idx} 个 sample")
                        temp = sample
                        for next_op_idx in range(op_idx + 1, len(operators)):
                            next_op = operators[next_op_idx]
                            next_op_name = next_op._name
                            next_actor = get_next_actor(next_op_name)
                            if next_op.is_batched_op():
                                logger.info(f"添加到 bucket[{next_op_idx}]")
                                buckets[next_op_idx].append(temp)
                                if len(buckets[next_op_idx]) >= batch_size:
                                    logger.info(f"触发 flush bucket[{next_op_idx}]")
                                    new_refs = self.flush_bucket(
                                        next_op_idx, 
                                        operators, 
                                        {next_op_name: [next_actor]}, 
                                        buckets, 
                                        op_batch_counters
                                    )
                                    futures.extend(new_refs)
                                    logger.info(f"添加 {len(new_refs)} 个 futures, 现在总数: {len(futures)}")
                                break
                            else:
                                logger.info(f"执行 single op {next_op_idx}")
                                temp = ray.get(
                                    next_actor.mapper_single.remote(
                                        pyarrow.Table.from_pylist([temp])
                                    )
                                )[0]
                                logger.info(f"single op 输出: {type(temp)}")
                        else:
                            logger.info(f"添加到 futures (类型: {type(temp)})")
                            futures.append(temp)
                            logger.info(f"futures 总数: {len(futures)}")

        # 最终统计
        logger.info(f"\n=== 最终结果 ===")
        logger.info(f"总 futures 数量: {len(futures)}")
        for i, future in enumerate(futures):
            logger.debug(f"future[{i}]: {type(future)}")
            if isinstance(future, ray.ObjectRef):
                try:
                    content = ray.get(future)
                    logger.debug(f"  content: {type(content)} (size: {len(content) if hasattr(content, '__len__') else 1})")
                except:
                    logger.debug("  (无法获取内容)")
        results = []
        for future in futures:
            try:
                if isinstance(future, ray.ObjectRef):
                    data = ray.get(future)
                    # 处理可能的各种返回类型
                    if isinstance(data, pyarrow.Table):
                        results.extend(data.to_pylist())
                    elif isinstance(data, (list, dict)):
                        results.extend(data) if isinstance(data, list) else results.append(data)
                    else:
                        results.append(data)
                else:
                    # 处理非ObjectRef的结果
                    if isinstance(future, pyarrow.Table):
                        results.extend(future.to_pylist())
                    else:
                        results.append(future)
            except Exception as e:
                print(f"Error processing future: {e}")

        # 确保所有记录都是可序列化的Python原生类型
        def ensure_serializable(obj):
            if isinstance(obj, pyarrow.Table):
                return obj.to_pylist()
            elif isinstance(obj, pyarrow.RecordBatch):
                return obj.to_pydict()
            elif hasattr(obj, '__dict__'):
                return vars(obj)
            return obj

        # 写入JSONL文件
        output_path = "output.jsonl"
        start = time.time()
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in results:
                try:
                    # 处理可能的嵌套结构
                    if isinstance(record, (list, tuple)):
                        record = [ensure_serializable(item) for item in record]
                    elif isinstance(record, dict):
                        record = {k: ensure_serializable(v) for k, v in record.items()}
                    else:
                        record = ensure_serializable(record)
                    
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Error writing record: {e}\nRecord: {record}")

        print(f"成功写入 {len(results)} 条数据到 {output_path}")
        print(f"写入耗时: {time.time() - start:.3f} s.")

        return self

    
    def flush_bucket(self, op_idx, operators, actors_dict, buckets, op_batch_counters):
        """将 bucket 中的数据拼成 pyarrow 表，并送入对应 actor 执行
        
        Args:
            op_idx: 操作索引
            operators: 所有操作列表
            actors_dict: 字典结构 {op_name: [actor1, actor2,...]}
            buckets: 数据桶
            op_batch_counters: 操作批计数器
        """
        op = operators[op_idx]
        op_name = op._name
        
        # 从字典中获取对应op的actors列表
        if op_name not in actors_dict:
            raise ValueError(f"No actors found for operator {op_name}")
        
        # 使用字典中该op的第一个actor（因为外部已经做了轮询选择）
        actor = actors_dict[op_name][0]
        
        bucket = buckets[op_idx]
        if not bucket:
            return []
        
        op_batch_counters[op_idx] += 1
        print(f"[DEBUG] → Flushing {op_name} (index {op_idx}) | Batch #{op_batch_counters[op_idx]} | Size: {len(bucket)}")

        # 合并 pylist 为 pyarrow.Table
        flattened_bucket = []
        for item in bucket:
            if isinstance(item, pyarrow.Table):
                flattened_bucket.extend(item.to_pylist())
            elif isinstance(item, dict):
                flattened_bucket.append(item)
            else:
                raise TypeError(f"[flush_bucket] Unsupported item type in bucket: {type(item)}")

        if not flattened_bucket:
            return []

        table = pyarrow.Table.from_pylist(flattened_bucket)
        logger.info("table type  :", type(table))
        logger.info("table :",table)
        # 根据操作类型调用不同的actor方法
        if isinstance(op, Filter):
            
            ref = actor.process_and_filter.remote(table)
        elif isinstance(op, Mapper):
            ref = actor.mapper.remote(table)
        else:
            raise NotImplementedError(f"Unsupported batched op type: {type(op)}")
        
        # 清空 bucket
        buckets[op_idx] = []
        return [ref]

    
    def merge_tables_to_one_row(self, all_batches: list[pa.Table]) -> pa.Table:
        # 把每个 Table 转为 List[Dict]
        tables_as_dicts = [tbl.to_pylist() for tbl in all_batches]

        # 构造单条记录，key用 mapper idx 标识
        merged_record = {
            f"mapper_{i+1}_results": tables_as_dicts[i]
            for i in range(len(tables_as_dicts))
        }

        # 转成只有一条记录的列表
        merged_list = [merged_record]

        # 生成 pyarrow.Table 返回
        return pa.Table.from_pylist(merged_list)
    
    def merge_records_by_video(self, records):
        merged = defaultdict(lambda: {
            "videos": None,
            "texts": [],
            "aesthetics_scores": []
        })

        for rec in records:
            # 关键点：确保 video_key 是 hashable 类型
            video_key = rec['videos']
            if isinstance(video_key, list):
                while isinstance(video_key, list):
                    video_key = video_key[0] if video_key else None
            video_key = str(video_key)

            if merged[video_key]["videos"] is None:
                merged[video_key]["videos"] = rec["videos"]

            merged[video_key]["texts"].append(rec.get("text"))
            merged[video_key]["aesthetics_scores"].append(rec.get("aesthetics_score"))

        merged_list = []
        for video_key, data in merged.items():
            merged_list.append({
                "videos": data["videos"],
                "texts": data["texts"],
                "aesthetics_scores": data["aesthetics_scores"]
            })
        return merged_list

    # def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
    #     if operators is None:
    #         return self
    #     if not isinstance(operators, list):
    #         operators = [operators]

    #     # 预先创建所有operator的actor
    #     actors = []
    #     for op in operators:
    #         columns = self.data.columns()
    #         if Fields.stats not in columns:

    #             def process_batch_arrow(table: pyarrow.Table):
    #                 new_column_data = [{} for _ in range(len(table))]
    #                 new_table = table.append_column(
    #                     Fields.stats, [new_column_data])
    #                 return new_table

    #             self.data = self.data.map_batches(process_batch_arrow,
    #                                                 batch_format='pyarrow')
    #         batch_size = getattr(op, 'batch_size', 1) if op.is_batched_op() else 1
    #         # 为每个operator创建actor
    #         actor = ComputeActor.options(
    #             name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
    #             num_gpus=0.5 if op.use_cuda() else 0,
    #             # num_gpus=0,
    #             scheduling_strategy=PlacementGroupSchedulingStrategy(
    #                 placement_group=self.gpu_pg,
    #                 placement_group_capture_child_tasks=True
    #             ),
    #             runtime_env={
    #                 "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"} if op.use_cuda() else {}
    #             }
    #         ).remote(op, batch_size)
    #         actors.append(actor)
        
    #     # 初始化流水线
    #     current_data = self.data
    #     result_refs = []
        
    #     # 启动异步流水线处理
    #     for i, (op, actor) in enumerate(zip(operators, actors)):
    #         # 如果是第一个operator，处理原始数据
    #         if i == 0:
    #             input_data = current_data
    #         else:
    #             # 否则等待前一个operator的结果
    #             input_data = ray.get(result_refs[i-1])
            
    #         # 提交当前operator的处理任务
    #         ref = actor.process_and_filter.remote(input_data)
    #         result_refs.append(ref)
            
    #         # 如果是最后一个operator，等待结果并处理输出
    #         if i == len(operators) - 1:
    #             final_result = ray.get(ref)
    #             if op.stats_export_path:
    #                 final_result.write_json(op.stats_export_path, force_ascii=False)
    #             self.data = final_result
        
    #     return self
    
    def _run_single_op(self, op):
        op_proc = calculate_np(op._name, op.mem_required, op.cpu_required, self.num_proc, op.use_cuda())
        num_gpus = get_num_gpus(op, op_proc)

        if op._name in TAGGING_OPS.modules and Fields.meta not in self.data.columns():

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.meta, [new_column_data])
                return new_table

            self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")

        try:
            batch_size = getattr(op, "batch_size", 1) if op.is_batched_op() else 1
            if isinstance(op, Mapper):
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_gpus=num_gpus,
                        concurrency=op_proc,
                        batch_format="pyarrow",
                    )
                else:
                    logger.info(f"{op._name} map_batches with {num_gpus} gpus ~")
                    self.data = self.data.map_batches(
                        op.process, batch_size=batch_size, batch_format="pyarrow", num_gpus=num_gpus
                    )
                    logger.info(f"type : {type(self.data)}")
                    logger.info("self.data:", self.data)
            elif isinstance(op, Filter):
                columns = self.data.columns()
                if Fields.stats not in columns:

                    def process_batch_arrow(table: pyarrow.Table):
                        new_column_data = [{} for _ in range(len(table))]
                        new_talbe = table.append_column(Fields.stats, [new_column_data])
                        return new_talbe

                    self.data = self.data.map_batches(process_batch_arrow, batch_format="pyarrow")
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_gpus=num_gpus,
                        concurrency=op_proc,
                        batch_format="pyarrow",
                    )
                else:
                    self.data = self.data.map_batches(
                        op.compute_stats, batch_size=batch_size, batch_format="pyarrow", num_gpus=num_gpus
                    )
                if op.stats_export_path is not None:
                    self.data.write_json(op.stats_export_path, force_ascii=False)
                if op.is_batched_op():
                    self.data = self.data.map_batches(
                        partial(filter_batch, filter_func=op.process),
                        batch_format="pyarrow",
                        batch_size=batch_size,
                        num_gpus=num_gpus,
                        zero_copy_batch=True,
                    )
                else:
                    self.data = self.data.filter(op.process)
            elif isinstance(op, Deduplicator):
                self.data = op.run(self.data)
            else:
                logger.error("Ray executor only support Filter and Mapper OPs for now")
                raise NotImplementedError
        except:  # noqa: E722
            logger.error(f"An error occurred during Op [{op._name}].")
            import traceback

            traceback.print_exc()
            exit(1)


    @classmethod
    def read(cls, data_format: str, paths: Union[str, List[str]]) -> RayDataset:
        if data_format in {"json", "jsonl"}:
            return RayDataset.read_json(paths)
        elif data_format == "webdataset":
            return RayDataset.read_webdataset(paths)
        elif data_format in {
            "parquet",
            "images",
            "parquet_bulk",
            "csv",
            "text",
            "avro",
            "numpy",
            "tfrecords",
            "binary_files",
            "lance",
        }:
            return getattr(ray.data, f"read_{data_format}")(paths)

    @classmethod
    def read_json(cls, paths: Union[str, List[str]]) -> RayDataset:
        # Note: a temp solution for reading json stream
        # TODO: replace with ray.data.read_json_stream once it is available
        import pyarrow.json as js

        try:
            js.open_json
            return read_json_stream(paths)
        except AttributeError:
            return ray.data.read_json(paths)

    @classmethod
    def read_webdataset(cls, paths: Union[str, List[str]]) -> RayDataset:
        return ray.data.read_webdataset(paths, decoder=partial(_custom_default_decoder, format="PIL"))

    def to_list(self) -> list:
        return self.data.to_pandas().to_dict(orient="records")


class JSONStreamDatasource(ray.data.read_api.JSONDatasource):
    """
    A temp Datasource for reading json stream.

    Note:

        Depends on a customized `pyarrow` with `open_json` method.
    """

    def _read_stream(self, f: "pyarrow.NativeFile", path: str):
        from pyarrow.json import open_json

        try:
            reader = open_json(
                f,
                read_options=self.read_options,
                **self.arrow_json_args,
            )
            schema = None
            while True:
                try:
                    batch = reader.read_next_batch()
                    table = pyarrow.Table.from_batches([batch], schema=schema)
                    if schema is None:
                        schema = table.schema
                    yield table
                except StopIteration:
                    return
        except pyarrow.lib.ArrowInvalid as e:
            raise ValueError(f"Failed to read JSON file: {path}.") from e


def read_json_stream(
    paths: Union[str, List[str]],
    *,
    filesystem: Optional["pyarrow.fs.FileSystem"] = None,
    parallelism: int = -1,
    ray_remote_args: Dict[str, Any] = None,
    arrow_open_stream_args: Optional[Dict[str, Any]] = None,
    meta_provider=None,
    partition_filter=None,
    partitioning=ray.data.read_api.Partitioning("hive"),
    include_paths: bool = False,
    ignore_missing_paths: bool = False,
    shuffle: Union[Literal["files"], None] = None,
    file_extensions: Optional[List[str]] = ["json", "jsonl"],
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
    **arrow_json_args,
) -> ray.data.Dataset:
    if meta_provider is None:
        meta_provider = ray.data.read_api.DefaultFileMetadataProvider()

    datasource = JSONStreamDatasource(
        paths,
        arrow_json_args=arrow_json_args,
        filesystem=filesystem,
        open_stream_args=arrow_open_stream_args,
        meta_provider=meta_provider,
        partition_filter=partition_filter,
        partitioning=partitioning,
        ignore_missing_paths=ignore_missing_paths,
        shuffle=shuffle,
        include_paths=include_paths,
        file_extensions=file_extensions,
    )
    return ray.data.read_datasource(
        datasource,
        parallelism=parallelism,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
    )