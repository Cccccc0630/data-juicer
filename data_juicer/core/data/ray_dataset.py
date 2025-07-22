from __future__ import annotations

from collections import defaultdict
from concurrent import futures
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

class MPSManager:
    """MPS资源管理器"""
    
    def __init__(self, target_gpu_id="0"):
        self.target_gpu_id = target_gpu_id
        self.current_percentage = 100
        self.op_allocations = {
            'video_aesthetics_filter': 50,  
            # 'video_captioning_from_frames_mapper': 10,  
        }  # 存储每个op的资源分配
        
    def init_mps(self):
        """初始化 MPS 服务器，确保完全清理旧状态"""
        try:
            subprocess.run(["pkill", "-9", "nvidia-cuda-mps"], check=False)
            subprocess.run(["pkill", "-9", "nvidia-cuda-mps-control"], check=False)
            time.sleep(1)  # 等待进程终止


            subprocess.run(
                ["nvidia-smi", "-i", str(self.target_gpu_id), "--compute-mode=0"],
                check=True,
            )


            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.target_gpu_id), "-q", "-d", "PIDS"],
                capture_output=True,
                text=True,
            )
            if 'MPS Server' in result.stdout:
                logger.info(f"MPS Server already running on GPU {self.target_gpu_id}")
            else:
                logger.info(f"Starting MPS Server on GPU {self.target_gpu_id}")
                subprocess.run(['nvidia-cuda-mps-control', '-d'], check=True)
                logger.info(f"MPS Server started on GPU {self.target_gpu_id}")


            self.set_mps_percentage(100)
            logger.info(f"MPS initialized on GPU {self.target_gpu_id}")

        except Exception as e:
            logger.error(f"Failed to initialize MPS: {e}")
            raise
    
    def set_mps_percentage(self, percentage):
        """设置MPS活跃线程百分比"""
        try:
            # 设置独占模式
            subprocess.run(['nvidia-smi', '-i', str(self.target_gpu_id), 
                        '--compute-mode=EXCLUSIVE_PROCESS'], check=True)
            
            # 直接设置环境变量并确保后续进程继承
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.target_gpu_id)
            os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percentage)
            
            # 显式启动一个设置百分比的MPS控制进程
            subprocess.run(
                f'echo "set_default_active_thread_percentage {percentage}" | nvidia-cuda-mps-control',
                shell=True,
                check=True
            )
            
            self.current_percentage = percentage
            logger.info(f"Set MPS percentage to {percentage}% on GPU {self.target_gpu_id}")
            
        except Exception as e:
            logger.warning(f"Failed to set MPS percentage: {e}")
            raise
    
    def allocate_resources_for_ops(self, operators):
        """为操作符分配MPS资源"""
        if not operators:
            return {}
        
        # 检查每个op的分配是否在合理范围内
        for op in operators:
            if op._name in self.op_allocations:
                allocation = self.op_allocations[op._name]
                if not isinstance(allocation, int) or allocation < 0 or allocation > 100:
                    logger.warning(f"Invalid allocation for op {op._name}: {allocation}. "
                                   f"Allocations should be integers between 0 and 100.")
                    return {}
            else:
                logger.warning(f"No allocation set for op {op._name}.")
        
        logger.info(f"MPS resource allocation: {self.op_allocations}")
        return self.op_allocations
    
    def get_runtime_env_for_op(self, op_name, base_env=None):
        """获取特定操作符的运行时环境"""
        runtime_env = base_env or {}
        
        if 'env_vars' not in runtime_env:
            runtime_env['env_vars'] = {}
        
        runtime_env['env_vars']['CUDA_VISIBLE_DEVICES'] = str(self.target_gpu_id)
        
        # 设置该op的MPS百分比
        percentage = self.op_allocations.get(op_name, 50)  # 默认50%
        runtime_env['env_vars']['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percentage)
        
        # 添加Nsight配置
        # if 'nsight' not in runtime_env:
        #     runtime_env['nsight'] = {
        #         "t": "cuda,nvtx,osrt",
        #         "cuda-memory-usage": "true",
        #         "stats": "true",
        #         "cuda-graph-trace": "graph",
        #     }
        
        return runtime_env


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
    
    # def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
    #     if operators is None:
    #         return self
    #     if not isinstance(operators, list):
    #         operators = [operators]
    #     for op in operators:
    #         self._run_single_op(op)
    #     return self
    def process(self, operators, *, exporter=None, checkpointer=None, tracer=None) -> DJDataset:
        if operators is None:
            return self
        if not isinstance(operators, list):
            operators = [operators]

        from ray.data import from_arrow

        actors = []

        # 初始化所有 actor
        for idx, op in enumerate(operators):
            columns = self.data.columns()
            if isinstance(op, Filter) and Fields.stats not in columns:
                def process_batch_arrow(table: pyarrow.Table):
                    new_column_data = [{} for _ in range(len(table))]
                    new_table = table.append_column(Fields.stats, [new_column_data])
                    return new_table
                self.data = self.data.map_batches(process_batch_arrow, batch_format='pyarrow')

            # batch_size = getattr(op, 'batch_size', 1) if op.is_batched_op() else 1
            # batch_size = 3
            if op.is_batched_op():
                batch_size = 10
            else:
                batch_size = 1

            gpu_allocate = 1 if op.use_cuda() else 0

            actor = ComputeActor.options(
                name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",
                # num_gpus=0.5 if op.use_cuda() else 0,
                num_gpus=gpu_allocate,
                num_cpus=4,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.gpu_pg,
                    placement_group_capture_child_tasks=True
                ),
                placement_group_bundle_index=idx,
                runtime_env={
                    "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"} if op.use_cuda() else {},
                    "nsight": {
                        "t": "cuda,nvtx,osrt",
                        "cuda-memory-usage": "true",
                        "stats": "true",
                        "cuda-graph-trace": "graph",
                    }
                }
            ).remote(op, batch_size)
            actors.append(actor)
        

        # 将数据按 batch 拆分处理，形成异步流水线
        # input_batches = self.data.iter_batches(batch_size=batch_size, batch_format="pyarrow")
        input_batches = list(self.data.iter_batches(batch_size=batch_size, batch_format="pyarrow"))
        print(f"Total batches generated: {len(input_batches)}")
        # print(input_batches)
        # final result refs for each batch
        futures = []

        for i, batch in enumerate(input_batches):
            if all(op.is_batched_op() for op in operators):
                # 全 batched 模式：直接链式传输
                ref = batch
                for op, actor in zip(operators, actors):
                    if isinstance(op, Filter):
                        ref = actor.process_and_filter.remote(ref)
                    elif isinstance(op, Mapper):
                        ref = actor.mapper.remote(ref)
                    else:
                        raise NotImplementedError(f"Unsupported op type: {type(op)}")
                futures.append(ref)
            
            else:
                # 混合模式
                current_refs = [batch]

                for op_idx, (op, actor) in enumerate(zip(operators, actors)):
                    new_refs = []

                    for ref in current_refs:
                        if op.is_batched_op():
                            if isinstance(ref, ray.ObjectRef):
                                ref = ray.get(ref)

                            if isinstance(ref, pyarrow.Table):
                                ref = ref.to_pylist()

                            new_ref = (
                                actor.process_and_filter.remote(ref) if isinstance(op, Filter)
                                else actor.mapper.remote(ref)
                            )
                            print(f"\nnew_ref: {new_ref}")
                            new_refs.append(new_ref)

                        else:
                            single_records = pyarrow.Table.to_pylist(ref)
                            for record in single_records:
                                single_table = pyarrow.Table.from_pylist([record])
                                ref_single = (
                                    actor.process_and_filter.remote(single_table)
                                    if isinstance(op, Filter)
                                    else actor.mapper_single.remote(single_table)
                                )
                                new_refs.append(ref_single)

                    # 更新 current_refs 用于下一轮
                    current_refs = new_refs

                # ✅ 所有 operators 处理完之后再 append！
                futures.append(current_refs)


        print(f"Number of futures: {len(futures)}")

        # Aggregate all results
        try:
            if futures:
                print("=" * 60)
                print("🔍 Step 1: Futures Submission Summary")
                print(f"→ Number of futures: {len(futures)}")
                print(futures)
                print("→ Operator chain:", [op.__class__.__name__ for op in operators])
                print("→ futures[0] type:", type(futures[0]))

                is_batched = all(op.is_batched_op() for op in operators)
                print(f"→ is_batched_op(): {is_batched}")

                flat_futures = (
                    [ref for sublist in futures for ref in sublist] if isinstance(futures[0], list) else futures
                )

                print("→ Total ObjectRefs to resolve:", len(flat_futures))

                results = ray.get(flat_futures)
                print("✅ Step 3: Results Retrieved from ray.get()")
                print(f"→ Number of result items: {len(results)}")
                print(f"→ Sample result type: {type(results[0])}")

                if is_batched and isinstance(results[0], list):
                    all_batches = [item for sublist in results for item in sublist]
                    print(f"→ Flattened batched results: {len(all_batches)}")
                else:
                    all_batches = results

                # 过滤空结果
                all_batches = [b for b in all_batches if b is not None]
                print(f"→ Filtered non-null batches: {len(all_batches)}")
                for i, b in enumerate(all_batches[:3]):
                    if isinstance(b, pyarrow.Table):
                        print(f"  Batch[{i}] rows: {b.num_rows}, schema: {b.schema}")
                    else:
                        print(f"  Batch[{i}] is not pyarrow.Table → {type(b)}")

                if all_batches:
                    # Step A: 合并多个表为一个大表（多行）
                    merged_table = pyarrow.concat_tables(all_batches)
                    print("✅ Step A: Tables Concatenated")
                    print(f"→ Merged rows: {merged_table.num_rows}")
                    print(f"→ Merged schema: {merged_table.schema}")

                    # Step B: 转成列表合并相同视频数据（按需）
                    records = merged_table.to_pylist()

                    # 调用按视频聚合函数，示例内定义
                    merged_records = self.merge_records_by_video(records)
                    print(f"✅ Step B: Records merged by video, total {len(merged_records)} records")

                    # 转回 pyarrow.Table
                    final_table = pyarrow.Table.from_pylist(merged_records)

                    # try:
                    #     df = final_table.to_pandas()
                    #     print("→ Preview merged result:\n", df.head(5))
                    #     dup = df[df.duplicated()]
                    #     print(f"→ Duplicate rows detected: {len(dup)}")
                    # except Exception as e:
                    #     print(f"⚠️ Failed to convert to pandas: {e}")

                    self.data = from_arrow(final_table)
                else:
                    print("⚠️ No valid batch data after filtering.")
                    self.data = ray.data.from_items([])
            else:
                print("⚠️ No futures to process.")
                self.data = ray.data.from_items([])

        except Exception as e:
            import traceback
            print("Ray Task Error:", type(e), str(e))
            traceback.print_exc()
            raise e

        return self


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
        op_proc = calculate_np(op._name, op.mem_required, op.cpu_required,
                               self.num_proc, op.use_cuda())
        op_proc = 1
        num_gpus = get_num_gpus(op, op_proc)
        num_gpus = 0.5

        if (op._name in TAGGING_OPS.modules
                and Fields.meta not in self.data.columns()):

            def process_batch_arrow(table: pyarrow.Table):
                new_column_data = [{} for _ in range(len(table))]
                new_table = table.append_column(Fields.meta, [new_column_data])
                return new_table

            self.data = self.data.map_batches(process_batch_arrow,
                                              batch_format='pyarrow')

        try:
            batch_size = getattr(op, 'batch_size', 1) if op.is_batched_op() else 1
            
            # 获取该操作符的运行时环境配置
            runtime_env = self.mps_manager.get_runtime_env_for_op(op._name)
            
            if isinstance(op, Mapper):
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    logger.info(f'Running Mapper {op._name} with MPS allocation: '
                              f'{self.mps_manager.op_allocations.get(op._name, "default")}%')
                    
                    self.data = self.data.map_batches(
                        op.__class__,
                        fn_args=None,
                        fn_kwargs=None,
                        fn_constructor_args=None,
                        fn_constructor_kwargs=op_kwargs,
                        batch_size=batch_size,
                        num_gpus=1,
                        concurrency=1,
                        batch_format='pyarrow',
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=self.gpu_pg,
                            placement_group_capture_child_tasks=True
                        ),
                        runtime_env=runtime_env)

                else:
                    self.data = self.data.map_batches(op.process,
                                                      batch_size=batch_size,
                                                      batch_format='pyarrow',
                                                      num_gpus=num_gpus)
                                                      
            elif isinstance(op, Filter):
                columns = self.data.columns()
                if Fields.stats not in columns:

                    def process_batch_arrow(table: pyarrow.Table):
                        new_column_data = [{} for _ in range(len(table))]
                        new_table = table.append_column(
                            Fields.stats, [new_column_data])
                        return new_table

                    self.data = self.data.map_batches(process_batch_arrow,
                                                      batch_format='pyarrow')
                                                      
                if op.use_cuda():
                    op_kwargs = op._op_cfg[op._name]
                    logger.info(f'Running Filter {op._name} with MPS allocation: '
                              f'{self.mps_manager.op_allocations.get(op._name, "default")}%')
                    
                    try:
                        # 创建Actor（资源参数全部在options中设置）
                        actor = ComputeActor.options(
                            name=f"actor_{op._name}_{uuid.uuid4().hex[:4]}",  # 唯一名称
                            num_gpus=0.5,      # 共享GPU
                            scheduling_strategy=PlacementGroupSchedulingStrategy(
                                placement_group=self.gpu_pg,
                                placement_group_capture_child_tasks=True
                            ),
                            runtime_env={
                                "env_vars": {"CUDA_VISIBLE_DEVICES": "0"}  # 明确指定GPU
                            }
                        ).remote(op, batch_size)

                        # 执行并获取结果
                        result_ref = actor.process_and_filter.remote(self.data)
                        dataset = ray.get(result_ref)  # 同步获取结果

                        # 处理输出
                        if op.stats_export_path:
                            dataset.write_json(op.stats_export_path, force_ascii=False)
                        self.data = dataset

                    except ray.exceptions.RayActorError as e:
                        print(f"Actor failed, falling back to CPU: {e}")
                        # CPU回退逻辑
                        self.data = self.data.map_batches(
                            partial(filter_batch, filter_func=op.process),
                            batch_size=batch_size,
                            batch_format="pyarrow"
                        )
                else:
                    # CPU模式直接调用（无需Actor）
                    if op.stats_export_path is not None:
                        self.data = self.data.map_batches(
                            op.compute_stats,
                            batch_size=batch_size,
                            batch_format="pyarrow"
                        ).write_json(op.stats_export_path, force_ascii=False)
                    
                    if op.is_batched_op():
                        self.data = self.data.map_batches(
                            partial(filter_batch, filter_func=op.process),
                            batch_size=batch_size,
                            batch_format="pyarrow",
                            zero_copy_batch=True
                        )
                    
            elif isinstance(op, Deduplicator):
                self.data = op.run(self.data)
            else:
                logger.error(
                    'Ray executor only support Filter and Mapper OPs for now')
                raise NotImplementedError
                
        except Exception as e:
            logger.error(f'An error occurred during Op [{op._name}]: {e}')
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