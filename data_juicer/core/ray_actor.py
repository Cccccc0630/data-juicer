from functools import partial
import os
import time
from typing import Any, Dict, List, Union
from unittest import result
from data_juicer.ops.base_op import Filter, Mapper
from data_juicer.utils.constant import Fields, StatsKeys
import ray
import torch
import pyarrow
import pyarrow as pa 

from loguru import logger


# @ray.remote(num_gpus=0.0, runtime_env={ "nsight": "default" })  # 注意：Ray不占GPU配额，由你手动控制
def filter_batch(batch, filter_func):
    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    keys = list_of_dicts[0].keys()
    dict_of_lists = {key: [] for key in keys}
    for d in list_of_dicts:
        for key in keys:
            dict_of_lists[key].append(d[key])
    return dict_of_lists

def dict_of_lists_to_list_of_dicts(data: Dict[str, List]) -> List[Dict]:
    return [
        {k: v[i] for k, v in data.items()}
        for i in range(min(len(v) for v in data.values()))
    ]


def flatten_records_with_text(records: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    flat_videos = []
    for v in records.get("videos", []):
        # 支持 [[[...]]] / [[...]] / [...] 各种层次
        if isinstance(v, list):
            while isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                v = v[0]
        flat_videos.append(v)

    # text字段直接保留，不变
    return {
        "videos": flat_videos,
        "text": records.get("text", []),
        "__dj__stats__": records.get("__dj__stats__", []),
        "__dj__source_file__": records.get("__dj__source_file__", [])
    }


def convert_record(batch: pyarrow.Table, root_path: str = "/home/xcy") -> dict:
    """
    安全转换PyArrow Table，兼容：
    - 三层嵌套 [[[path]]]
    - 直接字符串 "path"
    """
    # 解嵌套视频路径
    videos = []
    for video_col in batch.column('videos'):
        video = video_col.as_py()
        if isinstance(video, list):
            # 处理三层嵌套 [[[path]]]
            if video and isinstance(video[0], list):
                videos.append(str(video[0][0]))
            # 处理两层嵌套 [[path]]
            else:
                videos.append(str(video[0]))
        else:
            # 处理直接字符串
            videos.append(str(video))
    
    # 解嵌套文本
    texts = []
    for text_col in batch.column('text'):
        text = text_col.as_py()
        if isinstance(text, list):
            texts.append(str(text[0]))
        else:
            texts.append(str(text))
    
    # 构建结果字典
    result = {
        "videos": videos,
        "text": texts
    }
    
    # 处理统计信息（如存在）
    if '__dj__stats__' in batch.schema.names:
        result['__dj__stats__'] = [
            stats.as_py() for stats in batch.column('__dj__stats__')
        ]
    
    return result


def flatten_records(records):
    flattened = {}
    for k, v in records.items():
        if k == 'text' and isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
            flattened[k] = v[0]
        else:
            flattened[k] = v
    return flattened


@ray.remote(num_gpus=0.0) 
class ComputeActor:
    def __init__(self, op, batch_size, rank=None):
        """
        现在直接传入整个op对象，更灵活控制流程
        """
        self.op = op
        self.batch_size = batch_size
        self.call_count = 0
        # print(f"[Actor] Initialized with op: {self.op._name}, batch_size: {self.batch_size}")
        if op.use_cuda():
            start = time.time()
            self.op.load_model(rank=rank)
            print(f"[Actor] {self.op._name} Model loaded in {time.time() - start:.3f} seconds from {start} to {time.time()}")

    def process_and_filter(self, batch: pyarrow.Table) -> pyarrow.Table:
        self.call_count += 1
        logger.info(f"[Actor] Processing batch with {len(batch)} rows (call #{self.call_count})")

        try:
            # 1. 转换表格为记录列表
            records = batch.to_pylist()
            logger.debug(f"原始记录数: {len(records)}")

            # 2. 标准化记录结构
            processed_records = []
            for record in records:
                # 深度解包视频路径（处理三层嵌套）
                if 'videos' in record:
                    videos = []
                    for video_group in record['videos']:  # 第一层
                        if isinstance(video_group, list):
                            for video_list in video_group:  # 第二层
                                if isinstance(video_list, list):
                                    videos.extend(video_list)  # 第三层
                                else:
                                    videos.append(video_list)
                        else:
                            videos.append(video_group)
                    record['videos'] = videos

                # 处理文本字段
                if isinstance(record.get('text'), list):
                    record['text'] = record['text'][0] if record['text'] else ""

                processed_records.append(record)

            logger.debug(f"处理后记录数: {len(processed_records)}")
            logger.debug(f"首条记录视频路径: {processed_records[0].get('videos', [])[:1]}...")

            # 3. 过滤记录
            filtered_records = [
                record for record in processed_records
                if self.op.process_single(record)
            ]

            # 4. 重建表格（保持原始结构）
            schema = pyarrow.schema([
                ("videos", pyarrow.list_(pyarrow.list_(pyarrow.string()))),
                ("text", pyarrow.string()),
                ("__dj__stats__", pyarrow.map_(pyarrow.string(), pyarrow.string())),
                ("__dj__source_file__", pyarrow.list_(pyarrow.string()))
            ])

            # 将单层视频列表重新包装为三层结构
            final_records = []
            for record in filtered_records:
                final_record = {
                    "videos": [[record['videos']]] if record.get('videos') else [[]],
                    "text": record.get('text', ''),
                    "__dj__stats__": record.get('__dj__stats__', {}),
                    "__dj__source_file__": record.get('__dj__source_file__', [])
                }
                final_records.append(final_record)

            return pyarrow.Table.from_pylist(final_records, schema=schema)

        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            # 保存调试信息
            debug_data = {
                "input_sample": batch.slice(0, 1).to_pylist()[0] if len(batch) > 0 else None,
                "error": str(e)
            }
            with open("filter_debug.json", "w") as f:
                json.dump(debug_data, f, indent=2)
            raise
    
    def mapper(self, batch: Union[pa.Table, List[Dict]]) -> pa.Table:
        try:
            # === Step 1: 统一成 list[dict] ===
            if isinstance(batch, pa.Table):
                records = batch.to_pylist()
            elif isinstance(batch, list):
                records = []
                for item in batch:
                    if isinstance(item, pa.Table):
                        records.extend(item.to_pylist())
                    elif isinstance(item, dict):
                        records.append(item)
                    else:
                        print(f"[Mapper] Skipping item of type: {type(item)}")
            else:
                raise TypeError(f"[Mapper] Unsupported batch type: {type(batch)}")

            if not records:
                return None

            # === Step 2: 标准化每个样本（flatten videos, unwrap list fields） ===
            normalized_records = []
            for record in records:
                norm = {}

                # flatten videos field if present
                if "videos" in record:
                    v = record["videos"]
                    while isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                        v = v[0]
                    norm["videos"] = v
                else:
                    norm["videos"] = []

                # text: ensure it's a string
                text_val = record.get("text", "")
                norm["text"] = text_val[0] if isinstance(text_val, list) else text_val

                # pass through other fields (flatten list-of-one)
                for key, val in record.items():
                    if key in ("videos", "text"):
                        continue
                    if isinstance(val, list) and len(val) == 1:
                        norm[key] = val[0]
                    else:
                        norm[key] = val

                normalized_records.append(norm)

            if not normalized_records:
                return None

            # === Step 3: list[dict] → dict[list] for batched input ===
            batched_input = list_of_dicts_to_dict_of_lists(normalized_records)

            # === Step 4: align length to shortest list (防止字段不齐) ===
            min_len = min(len(v) for v in batched_input.values() if isinstance(v, list))
            for k in batched_input:
                if isinstance(batched_input[k], list):
                    batched_input[k] = batched_input[k][:min_len]

            # print(f"[Mapper] Final batched input keys: {list(batched_input.keys())}")
            # print(f"[Mapper] Length: {min_len}")

            # Step 5: 执行 batched operator
            output = self.op.process_batched(batched_input)

            if not output:
                return None

            # 如果是 dict of lists → 转为 list of dicts
            if isinstance(output, dict):
                output_records = dict_of_lists_to_list_of_dicts(output)
            else:
                output_records = output  # assume already List[Dict]
            result = pa.Table.from_pylist(output_records)
            # print("Mapper records schema:", result.schema)
            result = self.unify_mapper_schema(result)

            # print(f"Unified Mapper records schema: {result.schema}")
            # print(result)
            return result


        except Exception as e:
            print(f"[Actor(Mapper)] Failed: {e}")
            raise



    def mapper_single(self, batch: pyarrow.Table) -> pyarrow.Table:
        try:
            # print(batch)
            # 1. 原始转换
            records = convert_record(batch, root_path="/home/xcy")
            # print("\n\n")
            # print("records after convert:", records)
            records = flatten_records(records)
            # 2. 处理记录
            records = self.op.process_single(records)
            # print("records2 after mapper_single:", records)
            
            if not records:
                return None

            # 3. 标准化records格式（关键修改）
            standardized_records = []
            if isinstance(records, dict):
                # 处理单条记录（字典转列表）
                num_segments = len(records['videos'])
                standardized_records = [{
                    'videos': [[records['videos'][i]]],  # 双层嵌套
                    'text': [records['text']],           # 单层嵌套
                    '__dj__source_file__': [records['__dj__source_file__'][i]]
                } for i in range(num_segments)]
            elif isinstance(records, list):
                # 已经是列表格式则直接使用
                standardized_records = records
            

            result = pyarrow.Table.from_pylist(standardized_records)
            # print("Split records schema:", result.schema)
            # # 4. 转换为PyArrow Table
            # print(result)
            return result

        except Exception as e:
            print(f"[Actor(Mapper)] Failed: {e}")
            raise

    def unify_mapper_schema(self, table: pa.Table) -> pa.Table:
        columns = {}

        # === 1. 统一 videos 为 list<list<string>> ===
        if 'videos' in table.column_names:
            orig_videos = table['videos'].to_pylist()
            new_videos = []
            for v in orig_videos:
                if isinstance(v, list):
                    if len(v) > 0 and isinstance(v[0], list):
                        new_videos.append(v)
                    else:
                        new_videos.append([v])  # 转成 list<list<string>>
                else:
                    new_videos.append([[]])  # 空值统一为 [[]]
            columns['videos'] = pa.array(new_videos, type=pa.list_(pa.list_(pa.string())))

        # === 2. 保持 text 不变 ===
        if 'text' in table.column_names:
            columns['text'] = table['text']

        # === 3. 统一 aesthetics_score 为 list<item: double> ===
        if 'aesthetics_score' in table.column_names:
            orig_scores = table['aesthetics_score'].to_pylist()
            new_scores = []
            for score in orig_scores:
                if isinstance(score, list):
                    # 确保是 list<double> 格式
                    new_scores.append(score)
                else:
                    new_scores.append([score])  # 单个数值转换成 list<double>
            columns['aesthetics_score'] = pa.array(new_scores, type=pa.list_(pa.float64()))

        # 返回格式统一后的表
        return pa.table(columns)