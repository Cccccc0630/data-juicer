from PIL import Image
import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    close_video,
    extract_key_frames,
    extract_video_frames_uniformly,
    load_data_with_context,
    load_video,
)

from ...utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS
import cv2
from datetime import datetime
import re
torch = LazyLoader("torch")

OP_NAME = "video_aesthetics_filter"

@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoAestheticsFilter(Filter):
    """Filter to keep data samples with aesthetics scores for specified frames
    in the videos within a specific range.
    """

    _accelerator = "cuda"

    def __init__(
        self,
        hf_scorer_model: str = "",
        trust_remote_code: bool = False,
        min_score: float = 0.4,
        max_score: float = 1.0,
        frame_sampling_method: str = "uniform",
        frame_num: PositiveInt = 3,
        any_or_all: str = "any",
        reduce_mode: str = "avg",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_scorer_model: Huggingface model name for the aesthetics
            predictor. By default, we will use
            'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE',
            refer to pypi.org/project/simple-aesthetics-predictor
        :param min_score: Min score for the predicted aesthetics in a video.
        :param max_score: Max score for the predicted aesthetics in a video.
        :param frame_sampling_method: sampling method of extracting frame
            images from the videos.
            Should be one of ["all_keyframes", "uniform"].
            The former one extracts all key frames and the latter one extract
            specified number of frames uniformly from the video.
            Default: "uniform" with frame_num=3, considering that the number of
            keyframes can be large while their difference is usually small
            in terms of their aesthetics.
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param any_or_all: Keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param reduce_mode: reduce mode when one sample corresponds to
            multiple frames, must be one of ['avg','max', 'min'].
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """
        kwargs.setdefault("mem_required", "1500MB")
        super().__init__(*args, **kwargs)
        if hf_scorer_model == "":
            hf_scorer_model = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
        self.min_score = min_score
        self.max_score = max_score

        if frame_sampling_method not in ["all_keyframes", "uniform"]:
            raise ValueError(
                f"Frame sampling method "
                f"[{frame_sampling_method}] is not supported. "
                f'Can only be one of ["all_keyframes", "uniform"].'
            )

        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        if reduce_mode not in ["avg", "max", "min"]:
            raise ValueError(
                f"Reduce mode [{reduce_mode}] is not supported. " f'Can only be one of ["avg", "max", "min"].'
            )
        self.any = any_or_all == "any"
        self.reduce_mode = reduce_mode

        self.model_key = prepare_model(
            model_type="simple_aesthetics",
            pretrained_model_name_or_path=hf_scorer_model,
            trust_remote_code=trust_remote_code,
        )
        # the original score predicted by laion-ai's scorer is within [0, 10]
        self.need_normalized_by_ten = "shunk031/aesthetics-predictor" in hf_scorer_model
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num

        self.sampled_frames_key_suffix = f"-{frame_sampling_method}" + (
            "" if frame_sampling_method == "all_keyframes" else f"-{frame_num}"
        )

    def compute_stats_single_actor1(self, sample, model, processor, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_frames_aesthetics_score in sample[Fields.stats]:
            return sample
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = np.array([], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        aesthetics_scores = []
        torch.cuda.nvtx.range_push("Aesthetics")
        for key, video in videos.items():
            sampled_frames_key = key + self.sampled_frames_key_suffix
            if video is None:
                continue
            elif context and sampled_frames_key in sample[Fields.context]:
                # sampled frames can be found in the context
                frames = sample[Fields.context][sampled_frames_key]
            else:
                # extract frame images
                if self.frame_sampling_method == "all_keyframes":
                    frames = extract_key_frames(video)
                elif self.frame_sampling_method == "uniform":
                    frames = extract_video_frames_uniformly(video, self.frame_num)
                else:
                    frames = []

                # store the sampled frames in the context
                if context:
                    sample[Fields.context][sampled_frames_key] = frames
            frame_images = [frame.to_image() for frame in frames]
            
            if len(frame_images) > 0:
                # compute aesthetics_scores
                # model, processor = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
                inputs = processor(images=frame_images, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                if self.need_normalized_by_ten:
                    aesthetics_score = outputs.logits / 10.0
                else:
                    aesthetics_score = outputs.logits

                if self.reduce_mode == "avg":
                    aesthetics_score = float(aesthetics_score.mean())
                elif self.reduce_mode == "max":
                    aesthetics_score = float(aesthetics_score.max())
                else:
                    aesthetics_score = float(aesthetics_score.min())
            else:
                aesthetics_score = 0.0

            aesthetics_scores.append(aesthetics_score)
        torch.cuda.nvtx.range_pop()

        logger.debug(f"aesthetics_score: {aesthetics_scores}")

        sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = aesthetics_scores

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        return sample
    def parse_time_string(self, time_str):
        """
        支持两种输入格式:
        1. "00:00:06.000 [frame=144, fps=24.000]"
        2. "00:00:06.000"
        返回 (seconds, frame, fps)
        """
        if not isinstance(time_str, str):
            time_str = str(time_str)

        # 1. 解析主时间戳
        time_part = time_str.split()[0]  # e.g. "00:00:06.000"
        dt = datetime.strptime(time_part, "%H:%M:%S.%f")
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

        # 2. 可选解析帧和fps
        frame, fps = None, None
        match = re.search(r"\[frame=(\d+), fps=([\d\.]+)\]", time_str)
        if match:
            frame = int(match.group(1))
            fps = float(match.group(2))
        
        return seconds, frame, fps
    def compute_stats_single_actor(self, sample, model, processor, rank=None, context=False):
        # 如果已计算过，直接返回
        if StatsKeys.video_frames_aesthetics_score in sample[Fields.stats]:
            return sample

        # 如果没有视频片段，写空结果
        if 'time_pairs' not in sample or not sample['time_pairs']:
            sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = np.array([], dtype=np.float64)
            return sample

        aesthetics_scores = []
        torch.cuda.nvtx.range_push("Aesthetics")

        try:
            for segment in sample['time_pairs']:
                video_path = segment['video_path']
                start_str = segment['start_time']
                end_str = segment['end_time']

                # 解析时间戳
                start_seconds, start_frame, fps1 = self.parse_time_string(start_str)
                end_seconds, end_frame, fps2 = self.parse_time_string(end_str)

                # 打开视频
                video = cv2.VideoCapture(video_path)
                if not video.isOpened():
                    logger.warning(f"Failed to open video: {video_path}")
                    aesthetics_scores.append(0.0)
                    continue

                try:
                    fps = video.get(cv2.CAP_PROP_FPS) or fps1 or fps2

                    # 优先用帧数，其次用秒数*fps
                    if start_frame is None:
                        start_frame = int(start_seconds * fps)
                    if end_frame is None:
                        end_frame = int(end_seconds * fps)

                    frames = []
                    if self.frame_sampling_method == "all_keyframes":
                        step = max(1, (end_frame - start_frame) // self.frame_num)
                        for frame_idx in range(start_frame, end_frame, step):
                            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = video.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(Image.fromarray(frame))
                    else:  # uniform sampling
                        for i in range(self.frame_num):
                            frame_idx = start_frame + int(i * (end_frame - start_frame) / (self.frame_num - 1)) if self.frame_num > 1 else start_frame
                            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = video.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(Image.fromarray(frame))

                    # 保存 context
                    if context:
                        segment_key = f"{video_path}_{start_seconds}_{end_seconds}"
                        sample[Fields.context][segment_key] = frames

                    # 模型推理
                    if len(frames) > 0:
                        inputs = processor(images=frames, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            outputs = model(**inputs)

                        if self.need_normalized_by_ten:
                            aesthetics_score = outputs.logits / 10.0
                        else:
                            aesthetics_score = outputs.logits

                        if self.reduce_mode == "avg":
                            aesthetics_score = float(aesthetics_score.mean())
                        elif self.reduce_mode == "max":
                            aesthetics_score = float(aesthetics_score.max())
                        else:
                            aesthetics_score = float(aesthetics_score.min())
                    else:
                        aesthetics_score = 0.0

                    aesthetics_scores.append(aesthetics_score)
                finally:
                    video.release()

        finally:
            torch.cuda.nvtx.range_pop()

        logger.debug(f"aesthetics_score: {aesthetics_scores}")
        sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = aesthetics_scores
        return sample
    def compute_stats_single(self, sample, rank=None, context=False):
        print(f"sample in compute_stats_single: {sample}")
        # check if it's computed already
        if StatsKeys.video_frames_aesthetics_score in sample[Fields.stats]:
            return sample
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = np.array([], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        aesthetics_scores = []
        for key, video in videos.items():
            sampled_frames_key = key + self.sampled_frames_key_suffix
            if video is None:
                continue
            elif context and sampled_frames_key in sample[Fields.context]:
                # sampled frames can be found in the context
                frames = sample[Fields.context][sampled_frames_key]
            else:
                # extract frame images
                if self.frame_sampling_method == "all_keyframes":
                    frames = extract_key_frames(video)
                elif self.frame_sampling_method == "uniform":
                    frames = extract_video_frames_uniformly(video, self.frame_num)
                else:
                    frames = []

                # store the sampled frames in the context
                if context:
                    sample[Fields.context][sampled_frames_key] = frames
            frame_images = [frame.to_image() for frame in frames]

            if len(frame_images) > 0:
                # compute aesthetics_scores
                model, processor = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
                inputs = processor(images=frame_images, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                if self.need_normalized_by_ten:
                    aesthetics_score = outputs.logits / 10.0
                else:
                    aesthetics_score = outputs.logits

                if self.reduce_mode == "avg":
                    aesthetics_score = float(aesthetics_score.mean())
                elif self.reduce_mode == "max":
                    aesthetics_score = float(aesthetics_score.max())
                else:
                    aesthetics_score = float(aesthetics_score.min())
            else:
                aesthetics_score = 0.0

            aesthetics_scores.append(aesthetics_score)

        logger.debug(f"aesthetics_score: {aesthetics_scores}")

        sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = aesthetics_scores

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        return sample

    def process_single(self, sample):
        aesthetics_scores = (sample)[Fields.stats][StatsKeys.video_frames_aesthetics_score]
        if len(aesthetics_scores) <= 0:
            return True

        keep_bools = np.array(
            [self.min_score <= aesthetics_score <= self.max_score for aesthetics_score in aesthetics_scores]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
