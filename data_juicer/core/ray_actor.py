from functools import partial
import os
import time
from typing import Any, Dict, List, Union
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
        # print(f"\nbatch!!!!!: {batch}\n")
        # print(f"[Actor Debug] Processing batch with {len(batch)} rows")
        self.call_count += 1
        # print(f"[Actor] Call count: {self.call_count}, Rows: {len(batch)}")

        try:
            # 1. 先判断batch类型，获得records列表（dict列表）
            if isinstance(batch, list):
                # 如果list中的元素是Table，需要进一步处理
                records = []
                for item in batch:
                    if isinstance(item, pa.Table):
                        # 将Table转换为字典列表并扩展到records中
                        table_records = item.to_pylist()
                        records.extend(table_records)
                    elif isinstance(item, dict):
                        records.append(item)
                    else:
                        print(f"Warning: Unexpected item type in batch: {type(item)}")
                        
            elif isinstance(batch, pa.Table):
                records = batch.to_pylist()
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            # 2. 转换并标准化输入数据
            std_records = []
            for i, record in enumerate(records):
                print(f"Processing record {i}: type={type(record)}")
                # print("\nrecord:  ", record)
                # 确保record是字典
                if isinstance(record, pa.Table):
                    # 如果单个record还是Table，转换它
                    print("still Table:", record)
                    table_records = record.to_pylist()
                    for table_record in table_records:
                        if isinstance(table_record, dict):
                            std_records.append(table_record)
                elif isinstance(record, dict):
                    std_records.append(record)
                else:
                    print(f"Warning: Skipping record {i}, unexpected type: {type(record)}")
            print(f"Total records after conversion: {len(std_records)}")
            new_std_records = []
            # 现在使用std_records进行后续处理
            for record in std_records:
                # print("\nrecord:  ", record)
                if record.get('videos'):
                    # 兼容三层和两层嵌套
                    if isinstance(record['videos'][0][0], list):
                        videos = [v[0][0] for v in record['videos']]
                    else:
                        videos = [record['videos'][0][0]]

                text = record['text'][0] if isinstance(record['text'], list) else record['text']

                new_std_record = {
                    "videos": videos,
                    "text": text,
                    "__dj__stats__": record.get("__dj__stats__", {}),
                    "__dj__source_file__": record.get("__dj__source_file__", "")
                }
                new_std_records.append(new_std_record)

            # 3. 准备批量处理数据
            samples = {
                "videos": [r["videos"] for r in new_std_records],
                "text": [r["text"] for r in new_std_records],
                "__dj__stats__": [r["__dj__stats__"] for r in new_std_records],
                "__dj__source_file__": [r["__dj__source_file__"] for r in new_std_records]
            }

            # 4. 执行统计计算
            stats_records = self.op.compute_stats_batched(samples)
            print(f"\n\nComputed stats: {stats_records.keys()}")

            # 5. 构建过滤用的数据结构
            converted_batch = {
                Fields.stats: [
                    {
                        StatsKeys.video_frames_aesthetics_score: score,
                        StatsKeys.video_duration: duration
                    }
                    for score, duration in zip(
                        stats_records[StatsKeys.video_frames_aesthetics_score],
                        stats_records.get(StatsKeys.video_duration, [0]*len(stats_records[StatsKeys.video_frames_aesthetics_score]))
                    )
                ]
            }

            # 6. 执行过滤
            keep_flags = self.op.process_batched(converted_batch)

            filtered_records = [
                {
                    "videos": [[stats_records["videos"][i]]],
                    "text": stats_records["text"][i],
                    "aesthetics_score": stats_records[StatsKeys.video_frames_aesthetics_score][i],
                    # "video_duration": stats_records.get(StatsKeys.video_duration, [0]*len(stats_records[StatsKeys.video_frames_aesthetics_score]))[i],
                    # "__dj__stats__": stats_records["__dj__stats__"][i],
                    # "__dj__source_file__": stats_records["__dj__source_file__"][i]
                }
                for i, keep in enumerate(keep_flags) if keep
            ]
            # 7. 返回结果
            if not filtered_records:
                print("No records passed the filter")
                return None

            return pyarrow.Table.from_pylist(filtered_records)

        except Exception as e:
            print(f"[Actor] Processing failed: {e}")
            print(f"Last processed records: {std_records[-1] if std_records else 'None'}")
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

            print(f"[Mapper] Final batched input keys: {list(batched_input.keys())}")
            print(f"[Mapper] Length: {min_len}")

            # Step 5: 执行 batched operator
            output = self.op.process_batched(batched_input)

            if not output:
                return None

            # 如果是 dict of lists → 转为 list of dicts
            if isinstance(output, dict):
                output_records = dict_of_lists_to_list_of_dicts(output)
            else:
                output_records = output  # assume already List[Dict]

            return pa.Table.from_pylist(output_records)


        except Exception as e:
            print(f"[Actor(Mapper)] Failed: {e}")
            raise



    def mapper_single(self, batch: pyarrow.Table) -> pyarrow.Table:
        try:
            print(batch)
            # 1. 原始转换
            records = convert_record(batch, root_path="/home/xcy")
            print("\n\n")
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

            # 4. 转换为PyArrow Table
            return pyarrow.Table.from_pylist(standardized_records)
            
        except Exception as e:
            print(f"[Actor(Mapper)] Failed: {e}")
            raise


