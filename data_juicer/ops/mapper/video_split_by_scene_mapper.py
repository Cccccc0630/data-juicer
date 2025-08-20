from collections import defaultdict
import math
import os
import re
from itertools import chain
import time  # 导入time模块

from pydantic import NonNegativeFloat, NonNegativeInt
from scenedetect.frame_timecode import FrameTimecode
import cv2
from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import add_suffix_to_filename, transfer_filename
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens

from ..base_op import OPERATORS, Mapper
scenedetect = LazyLoader("scenedetect")

OP_NAME = "video_split_by_scene_mapper"


def replace_func(match, scene_counts_iter):
    try:
        count = next(scene_counts_iter)
        return SpecialTokens.video * count
    except StopIteration:
        return match.group(0)


@OPERATORS.register_module(OP_NAME)
class VideoSplitBySceneMapper(Mapper):
    """Mapper to cut videos into scene clips."""

    # Define shared detector keys and their properties
    avaliable_detectors = {
        "ContentDetector": ["weights", "luma_only", "kernel_size"],
        "AdaptiveDetector": [
            "window_width",
            "min_content_val",
            "weights",
            "luma_only",
            "kernel_size",
            "video_manager",
            "min_delta_hsv",
        ],
        "ThresholdDetector": ["fade_bias", "add_final_scene", "method", "block_size"],
    }

    def __init__(
        self,
        detector: str = "ContentDetector",
        threshold: NonNegativeFloat = 27.0,
        min_scene_len: NonNegativeInt = 15,
        show_progress: bool = False,
        save_dir: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param detector: Algorithm from `scenedetect.detectors`. Should be one
            of ['ContentDetector', 'ThresholdDetector', 'AdaptiveDetector`].
        :param threshold: Threshold passed to the detector.
        :param min_scene_len: Minimum length of any scene.
        :param show_progress: Whether to show progress from scenedetect.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)
        if detector not in self.avaliable_detectors:
            raise ValueError(
                f"Scene detector {detector} is not supported. "
                f"Can only be one of {list(self.avaliable_detectors.keys())}."
            )

        self.detector = detector
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.show_progress = show_progress
        self.save_dir = save_dir
        # prepare detector args
        avaliable_kwargs = self.avaliable_detectors[self.detector]
        self.detector_class = getattr(scenedetect.detectors, self.detector)
        self.detector_kwargs = {key: kwargs[key] for key in avaliable_kwargs if key in kwargs}

    def process_single(self, sample, context=False):
        # 打开log.txt文件进行写入日志
        with open("log.txt", "a") as log_file:
            # 记录开始时间
            start_time = time.time()
            start_ctime = time.ctime()

            # there is no video in this sample
            if self.video_key not in sample or not sample[self.video_key]:
                sample[Fields.source_file] = []
                log_file.write(f"[{time.ctime()}] No video found in sample.\n")
                return sample

            # load videos
            loaded_video_keys = sample[self.video_key]
            output_video_keys = {}
            scene_counts = {}

            for video_key in loaded_video_keys:
                # skip duplicate
                if video_key in output_video_keys:
                    continue

                redirected_video_key = transfer_filename(video_key, OP_NAME, **self._init_parameters)
                output_template = add_suffix_to_filename(redirected_video_key, "_$SCENE_NUMBER")

                # detect scenes
                detector = self.detector_class(self.threshold, self.min_scene_len, **self.detector_kwargs)
                scene_list = scenedetect.detect(video_key, detector, show_progress=self.show_progress, start_in_scene=True)
                scene_counts[video_key] = len(scene_list)
                print("Scene list for video '{}': {}".format(video_key, scene_list))
                # 记录视频处理的起始时间和结束时间
                if len(scene_list) > 1:
                    scene_num_format = f"%0{max(3, math.floor(math.log(len(scene_list), 10)) + 1)}d"
                    output_video_keys[video_key] = [
                        output_template.replace("$SCENE_NUMBER", scene_num_format % (i + 1)) for i in range(len(scene_list))
                    ]
                    # split video into clips
                    scenedetect.split_video_ffmpeg(
                        input_video_path=video_key,
                        scene_list=scene_list,
                        output_file_template=output_template,
                        show_progress=self.show_progress,
                    )

                    log_file.write(f"[{time.ctime()}] Video '{video_key}' processed, {len(scene_list)} scenes detected.\n")
                else:
                    output_video_keys[video_key] = [video_key]
                    log_file.write(f"[{time.ctime()}] Video '{video_key}' processed, 1 scene detected.\n")

            # replace split video tokens
            if self.text_key in sample:
                scene_counts_iter = iter([scene_counts[key] for key in loaded_video_keys])
                updated_text = re.sub(
                    re.escape(SpecialTokens.video),
                    lambda match: replace_func(match, scene_counts_iter),
                    sample[self.text_key],
                )
                sample[self.text_key] = updated_text

            # when the file is modified, its source file needs to be updated.
            sample[Fields.source_file] = []
            for value in loaded_video_keys:
                sample[Fields.source_file].extend([value] * len(output_video_keys[value]))

            sample[self.video_key] = list(chain.from_iterable([output_video_keys[key] for key in loaded_video_keys]))

            # 记录处理结束时间和耗时
            end_time = time.time()
            elapsed_time = end_time - start_time
            log_file.write(f"[{time.ctime()}] Video processing for {', '.join(loaded_video_keys)} completed. Time taken: {elapsed_time:.2f} seconds. Start at {start_ctime} \n\n")

        return sample
    
    def detect(self, sample, context=False):
        # 如果没有视频，返回原样
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        # 加载视频
        loaded_video_keys = sample[self.video_key]
        output_video_keys = {}
        scene_counts = {}
        time_pairs = []  # 存储所有需要切分的time_pair

        for video_key in loaded_video_keys:
            # 跳过重复的视频
            if video_key in output_video_keys:
                continue

            # 创建场景检测器
            detector = self.detector_class(self.threshold, self.min_scene_len, **self.detector_kwargs)
            scene_list = scenedetect.detect(video_key, detector, show_progress=self.show_progress, start_in_scene=True)
            scene_counts[video_key] = len(scene_list)

            # 记录视频处理的起始时间和结束时间（FrameTime对象）
            if len(scene_list) > 1:
                for scene in scene_list:
                    time_pairs.append({
                        'video_path': video_key,
                        'start_time': scene[0],  # FrameTime 对象
                        'end_time': scene[1],    # FrameTime 对象
                    })
                
                # 模拟输出文件名（实际不生成文件）
                scene_num_format = f"%0{max(3, math.floor(math.log(len(scene_list), 10)) + 1)}d"
                output_video_keys[video_key] = [
                    f"scene_{scene_num_format % (i + 1)}.mp4" for i in range(len(scene_list))
                ]
            else:
                # 如果没有检测到切分点，将整个视频的开头和结尾作为时间对
                video_cap = cv2.VideoCapture(video_key)
                frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video_cap.get(cv2.CAP_PROP_FPS)
                video_cap.release()
                
                if frame_count > 0 and fps > 0:
                    time_pairs.append({
                        'video_path': video_key,
                        'start_time': FrameTimecode(0, fps),  # 视频开头
                        'end_time': FrameTimecode(frame_count-1, fps)  # 视频结尾
                    })
                
                output_video_keys[video_key] = [video_key]

        # 将time_pairs存入sample中返回
        sample['time_pairs'] = time_pairs

        # 更新文本标记（保持原逻辑）
        if self.text_key in sample:
            scene_counts_iter = iter([scene_counts[key] for key in loaded_video_keys])
            updated_text = re.sub(
                re.escape(SpecialTokens.video),
                lambda match: replace_func(match, scene_counts_iter),
                sample[self.text_key],
            )
            sample[self.text_key] = updated_text

        # 更新source_file信息（模拟）
        sample[Fields.source_file] = []
        for value in loaded_video_keys:
            sample[Fields.source_file].extend([value] * len(output_video_keys[value]))

        # 更新video_key信息（模拟）
        sample[self.video_key] = loaded_video_keys  # 保持原始视频路径
        print("Detecting result:", sample)
        return sample

    def cut(self, sample, context=False):
        # 如果没有 time_pairs，返回原样
        if 'time_pairs' not in sample or not sample['time_pairs']:
            return sample

        # 获取 time_pairs
        time_pairs = sample['time_pairs']
        cut_video_dict = {}  # 改为字典存储列表，格式: {原始路径: [切割后路径1, 切割后路径2, ...]}
        source_files = []     # 存储所有原始文件路径（保持与切割后路径的顺序一致）

        for pair in time_pairs:
            video_key = pair['video_path']
            start_time = pair['start_time']
            end_time = pair['end_time']

            # 构建 scene_list：直接使用 FrameTime 对象
            scene_list = [(start_time, end_time)]

            # 构造输出文件路径
            output_path = add_suffix_to_filename(
                video_key, 
                f"_start_{start_time.get_seconds()}_end_{end_time.get_seconds()}"
            )

            # 使用 scenedetect 进行视频切割
            scenedetect.split_video_ffmpeg(
                input_video_path=video_key,
                scene_list=scene_list,
                output_file_template=output_path,
                show_progress=self.show_progress,
            )

            # 更新字典：如果键不存在则初始化列表，然后追加新路径
            if video_key not in cut_video_dict:
                cut_video_dict[video_key] = []
            cut_video_dict[video_key].append(output_path)
            source_files.append(video_key)

        # 展平所有切割后的路径（保持与time_pairs相同的顺序）
        all_cut_paths = [
            path for sublist in cut_video_dict.values() 
            for path in sublist
        ]

        # 更新 sample 中的 video_key 和 source_file
        sample[self.video_key] = all_cut_paths
        sample[Fields.source_file] = source_files

        print("Cutting result:", sample)
        return sample
