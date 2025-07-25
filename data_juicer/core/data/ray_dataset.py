from __future__ import annotations
import warnings
# 忽略所有 Matplotlib 字体相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
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
from data_juicer.core.ray_actor import ComputeActor, ProcessActor, filter_batch
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
import pytz
from datetime import datetime
beijing_tz = pytz.timezone('Asia/Singapore')
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
        actor_allocate = [0, 1, 1]
        model_loading_tasks = []
        cpu_op = []
        batch_size_list = [1, 6, 6]
        for idx, op in enumerate(operators):

            columns = self.data.columns()
            if isinstance(op, Filter) and Fields.stats not in columns:
                def process_batch_arrow(table: pyarrow.Table):
                    new_column_data = [{} for _ in range(len(table))]
                    new_table = table.append_column(Fields.stats, [new_column_data])
                    return new_table
                self.data = self.data.map_batches(process_batch_arrow, batch_format='pyarrow')
            actors[op._name] = []

             # 设置 batch_size（op2, op3, ...）
            batch_size = batch_size_list[idx]

            # 资源分配
            gpu_allocate = 1
            cpu_allocate = 4
            bundle_allocate = [0, 0, 1]
            # 如果是不需要 CUDA（即不需要加载模型）则直接提交数据处理任务
            if op.use_cuda() == False:
                cpu_op.append(op)
                # 为这个不需要模型的 OP 创建 Actor
                actor = ProcessActor.options(
                    name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
                    num_gpus=0,  
                    num_cpus=4,  
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.gpu_pg,
                        placement_group_capture_child_tasks=True
                    ),
                    placement_group_bundle_index=0, 
                    runtime_env={},
                ).remote(op, batch_size)  
                if idx == 0:
                    self.data=ray.get(actor.process.remote(self.data))  # 直接提交 map_batches,和其他actor模型加载并行
                
            else:
                actor = ComputeActor.options(
                    name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
                    num_gpus=gpu_allocate,
                    num_cpus=cpu_allocate,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.gpu_pg,
                        placement_group_capture_child_tasks=True
                    ),
                    placement_group_bundle_index=bundle_allocate[idx],
                ).remote(op, batch_size)
                actor.load_model.remote()
            actors[op._name].append(actor)
            
        # operators = operators[-2:]
        # 打印所有actor信息
        for op._name, actor_list in actors.items():
            logger.info(f"Operator {op._name} 有以下actors:")
            for i, actor in enumerate(actor_list):
                logger.info(f"  Actor {i}: {actor._ray_actor_id.hex()[:6]}")


        def flatten_videos_field(videos_field):
            while isinstance(videos_field, list) and len(videos_field) == 1:
                videos_field = videos_field[0]
            return videos_field

        # 准备数据批次
        input_batches = list(self.data.iter_batches(batch_size=batch_size, batch_format="pyarrow"))
        input_batches = self.process_and_flatten_tables(input_batches) # List of Dict
        # logger.info(f"Total batches generated: {len(input_batches)}")
        # logger.info(f"input batches: {input_batches}")  # List of Dict
        # futures = []
        # buckets = defaultdict(list)
        # op_batch_counters = defaultdict(int)

        operators=operators[1:]
        bucket_size = 6  # 阈值
        op_buckets = {op._name: [] for op in operators}
        op_buckets[operators[0]._name] = input_batches  # 初始数据放第一个op bucket

        result_map = {}  # 用于存储所有op处理过的中间结果，格式 {videos: {...}}

        for op_idx, op in enumerate(operators):
            op_name = op._name
            actor = actors[op_name][0]

            while len(op_buckets[op_name]) >= bucket_size:
                bucket_data = op_buckets[op_name]
                to_process = bucket_data[:bucket_size]
                op_buckets[op_name] = bucket_data[bucket_size:]

                batch = pyarrow.Table.from_pylist(to_process)

                if isinstance(op, Filter):
                    result = ray.get(actor.process_and_filter.remote(batch))
                else:
                    result = ray.get(actor.mapper.remote(batch))

                result_list = result.to_pylist()

                # 把本次处理结果加入 result_map
                for item in result_list:
                    videos_field = item["videos"]
                    # 转成字符串或元组作为 key
                    key = flatten_videos_field(videos_field)
                    value = {k: v for k, v in item.items() if k != "videos"}
                    result_map[key] = value

                # 继续传递给下一个 op 的 bucket
                if op_idx + 1 < len(operators):
                    next_op_name = operators[op_idx + 1]._name
                    op_buckets[next_op_name].extend(result_list)

        # flush 每个 op 剩余数据，顺便把结果加到 result_map
        for op_idx, op in enumerate(operators):
            op_name = op._name
            actor = actors[op_name][0]
            bucket = op_buckets[op_name]

            if bucket:
                batch = pyarrow.Table.from_pylist(bucket)
                if isinstance(op, Filter):
                    result = ray.get(actor.process_and_filter.remote(batch))
                else:
                    result = ray.get(actor.mapper.remote(batch))

                result_list = result.to_pylist()
                for item in result_list:
                    key = item["videos"]
                    value = {k: v for k, v in item.items() if k != "videos"}
                    result_map[key] = value

        # 输出最终的 result_map
        print(f"最终中间结果共有 {len(result_map)} 条记录")
        

        with open("result_map.jsonl", "w", encoding="utf-8") as f:
            for key, value in result_map.items():
                json_line = {
                    "key": key,
                    "value": value
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        return

        
    def process_and_flatten_tables(self, input_batches: List[pa.Table]) -> List[Dict[str, Any]]:
        """处理并扁平化 PyArrow Table 列表，最终转换为 pylist
        
        参数:
            input_batches: 包含嵌套列表的 PyArrow Table 列表
            
        返回:
            扁平化后的记录列表 (pylist)
        """
        all_records = []
        
        for batch_idx, batch in enumerate(input_batches):
            logger.info(f"\n=== 处理第 {batch_idx} 个 batch ===")
            logger.info(f"输入 batch 类型: {type(batch)}")
            
            if not isinstance(batch, pa.Table):
                logger.warning(f"跳过非 PyArrow Table 类型的 batch: {type(batch)}")
                continue
                
            # 打印原始 batch 信息
            logger.debug(f"原始 batch:\n{batch}")
            
            # 扁平化处理
            flattened_rows = []
            for row_idx in range(batch.num_rows):
                row = batch.slice(row_idx, 1).to_pydict()
                
                # 提取行数据
                videos_nested = row["videos"][0]
                text = row["text"][0]
                sources_nested = row["__dj__source_file__"][0]
                stats = row["__dj__stats__"][0]
                
                # 处理嵌套的视频和源文件路径
                videos = videos_nested[0] if (isinstance(videos_nested, list) 
                                            and isinstance(videos_nested[0], list)) else videos_nested
                sources = sources_nested[0] if (isinstance(sources_nested, list) 
                                            and isinstance(sources_nested[0], list)) else sources_nested
                
                # 为每个视频创建一行
                for video, source in zip(videos, sources):
                    flattened_rows.append({
                        "videos": [video],
                        "text": text,
                        "__dj__source_file__": [source],
                        "__dj__stats__": stats
                    })
            
            # 转换为 PyArrow Table 并记录日志
            flattened_table = pa.Table.from_pylist(flattened_rows)
            logger.info(f"扁平化后得到 {len(flattened_rows)} 行数据")
            logger.debug(f"扁平化后的 table:\n{flattened_table}")
            
            # 转换为 pylist 并添加到总结果
            batch_records = flattened_table.to_pylist()
            logger.info(f"转换为 {len(batch_records)} 条 records")
            all_records.extend(batch_records)
        
        logger.info(f"\n=== 处理完成 ===")
        logger.info(f"总共生成 {len(all_records)} 条 records")
        return all_records
    
    
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