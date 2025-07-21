from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from typing import Dict, Tuple
from xml.parsers.expat import model
import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (close_video, extract_key_frames,
                                        extract_video_frames_uniformly,
                                        load_data_with_context, load_video)

from ...utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS

torch = LazyLoader('torch', 'torch')

OP_NAME = 'video_aesthetics_filter'


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoAestheticsFilter(Filter):
    """Filter to keep data samples with aesthetics scores for specified frames
    in the videos within a specific range.
    """

    _accelerator = 'cuda'
    _batched_op = True
    def __init__(self,
                 hf_scorer_model: str = '',
                 trust_remote_code: bool = False,
                 min_score: float = 0.4,
                 max_score: float = 1.0,
                 frame_sampling_method: str = 'uniform',
                 frame_num: PositiveInt = 3,
                 any_or_all: str = 'any',
                 reduce_mode: str = 'avg',
                 *args,
                 **kwargs):
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
        kwargs.setdefault('mem_required', '1500MB')
        super().__init__(*args, **kwargs)
        if hf_scorer_model == '':
            hf_scorer_model = \
                'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'
        self.min_score = min_score
        self.max_score = max_score

        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method '
                f'[{frame_sampling_method}] is not supported. '
                f'Can only be one of ["all_keyframes", "uniform"].')

        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        if reduce_mode not in ['avg', 'max', 'min']:
            raise ValueError(f'Reduce mode [{reduce_mode}] is not supported. '
                             f'Can only be one of ["avg", "max", "min"].')
        self.any = (any_or_all == 'any')
        self.reduce_mode = reduce_mode

        self.model_key = prepare_model(
            model_type='simple_aesthetics',
            pretrained_model_name_or_path=hf_scorer_model,
            trust_remote_code=trust_remote_code)
        # the original score predicted by laion-ai's scorer is within [0, 10]
        self.need_normalized_by_ten = ('shunk031/aesthetics-predictor'
                                       in hf_scorer_model)
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num

        self.sampled_frames_key_suffix = f'-{frame_sampling_method}' + \
            ('' if frame_sampling_method == 'all_keyframes'
             else f'-{frame_num}')



    def compute_stats_single(self, sample, rank=None, context=False):
        print('Computing aesthetics scores for single sample')
        # check if it's computed already
        if StatsKeys.video_frames_aesthetics_score in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = (
                np.array([], dtype=np.float64))
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

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
                if self.frame_sampling_method == 'all_keyframes':
                    frames, _ = extract_key_frames(video)
                elif self.frame_sampling_method == 'uniform':
                    frames, _ = extract_video_frames_uniformly(
                        video, self.frame_num)
                else:
                    frames = []

                # store the sampled frames in the context
                if context:
                    sample[Fields.context][sampled_frames_key] = frames

            frame_images = [frame.to_image() for frame in frames]

            if len(frame_images) > 0:
                # compute aesthetics_scores
                model, processor = get_model(self.model_key,
                                             rank=rank,
                                             use_cuda=self.use_cuda())
                inputs = processor(images=frame_images,
                                   return_tensors='pt').to(model.device)
                device = model.device
                logger.info(f"device: {device}")
                with torch.no_grad():
                    outputs = model(**inputs)
                if self.need_normalized_by_ten:
                    aesthetics_score = outputs.logits / 10.0
                else:
                    aesthetics_score = outputs.logits

                if self.reduce_mode == 'avg':
                    aesthetics_score = float(aesthetics_score.mean())
                elif self.reduce_mode == 'max':
                    aesthetics_score = float(aesthetics_score.max())
                else:
                    aesthetics_score = float(aesthetics_score.min())
            else:
                aesthetics_score = 0.0

            aesthetics_scores.append(aesthetics_score)

        logger.debug(f'aesthetics_score: {aesthetics_scores}')

        sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = (
            aesthetics_scores)

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        logger.info(f"[Rank {rank}] Peak Memory: Allocated={peak_alloc:.2f} GB, Reserved={peak_reserved:.2f} GB")
        return sample
    

    
    def load_model(self, rank=None):
        model, processor = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
        return model, processor
    
    def compute_stats_batched(self, samples, rank=None, context=False):
        # print('Computing aesthetics scores for batch of samples')
        print(f"[Debug] samples : {samples}")
        # 检查是否已经计算过
        if StatsKeys.video_frames_aesthetics_score in samples[Fields.stats]:
            print('Aesthetics scores already computed, returning samples')
            return samples
        model , processor = self.load_model(rank=rank)
        device = model.device

        keys = samples.keys()
        num_samples = len(samples[Fields.stats])
        # print(samples)
        # print(f'[DEBUG] Total samples to process: {num_samples}')
        # print(f"[Debug] samples keys: {samples.keys()}")
        # print(f"[Debug] samples[{self.video_key}]: {samples.get(self.video_key)}")
        # print(f"[Debug] samples[{self.text_key}]: {samples.get(self.text_key)}")
        reconstructed_samples = [{key: samples[key][i] for key in keys} for i in range(num_samples)]
        # print(f'[DEBUG] Reconstructed samples: {reconstructed_samples}')
        video_tasks = []
        for sample_idx, sample in enumerate(reconstructed_samples):
            if self.video_key not in sample or not sample[self.video_key]:
                continue
            for vid_key in sample[self.video_key]:
                print(f"[DEBUG] sample_idx={sample_idx}, vid_key={vid_key}")
                video_tasks.append({
                    'sample_index': sample_idx,
                    'vid_key': vid_key,
                    'sample': sample,
                })
        # print(f'[DEBUG] Total video tasks: {len(video_tasks)}')
        if not video_tasks:
            return samples
        # logger.info("Start frame extraction ")
        start = time.time()
        def extract_frames(task: Dict) -> Tuple[str, Dict]:
            vid_key = task.get('vid_key', '')
            # print(f"Video path exists: {os.path.exists(vid_key)}")
            if not vid_key:
                return vid_key, {'frames': [], 'size': (0, 0)}
            
            for attempt in range(3):
                try:
                    dummy_sample = {self.video_key: [vid_key]}
                    _, videos = load_data_with_context(dummy_sample, False, [vid_key], load_video)
                    # print(f"videos type: {type(videos)}")  # Debug videos type
                    # print(f"videos keys: {videos.keys()}")  # Debug videos keys
                    # print(f"videos content: {videos}")  # Debug videos content
                    video = videos.get(vid_key) if videos else None
                    # print(f"video type: {type(video)}")  # Debug video type
                    if video is None:
                        return vid_key, {'frames': [], 'size': (0, 0)}
                    
                    if self.frame_sampling_method == 'all_keyframes':
                        frames, size = extract_key_frames(video)
                    elif self.frame_sampling_method == 'uniform':
                        frames, size = extract_video_frames_uniformly(video, self.frame_num)
                        # print(f"extract_video_frames_uniformly returned - frames type: {type(frames)}, size: {size}")
                    else:
                        frames = []
                        size = (0, 0)
                    close_video(video)

                    return vid_key,{
                        'raw_frames':[(f, size) for f in frames],
                        'size': size
                    }
                except Exception as e:
                    logger.error(f"Error extracting frames for {vid_key}: {e}")
                    time.sleep(1)
            return vid_key, {'frames': [], 'size': (0, 0)}
        
        cpu_count = os.cpu_count() or 4
        max_workers = max(2, min(int(cpu_count * 0.75), 64))
        max_workers = 64
        video_frame_cache = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_frames, task): task['vid_key'] for task in video_tasks}
            for future in as_completed(futures):
                vid_key, result = future.result()
                if vid_key:
                    video_frame_cache[vid_key] = result
        torch.cuda.synchronize()
        logger.info(f"Frame extraction completed in {time.time() - start:.2f} seconds")

        size_to_frames = defaultdict(list)
        for task in video_tasks:
            if task['vid_key'] in video_frame_cache:
                frame_data = video_frame_cache[task['vid_key']]
                for frame, size in frame_data['raw_frames']:
                    size_to_frames[size].append(
                        {
                            'frame_obj': frame,
                            'sample_index': task['sample_index'],
                            'vid_key': task['vid_key']
                        }
                    )

        # logger.info("Start batching inference")
        start = time.time()
        for size, frame_items in size_to_frames.items():
            # Process each batch of frames
            # logger.info(f"Processing batch for size {size}: {frame_items}")
            if not frame_items or size == (0, 0):
                continue

            batch_size = 512
            i = 0
            
            while i < len(frame_items):
                try:
                    batch_result = self._safe_batch_inference(
                        model=model,
                        processor=processor,
                        batch_items=frame_items[i:i + batch_size],
                        device=device
                    )

                    for j, score in enumerate(batch_result['scores']):
                        item = frame_items[i + j]
                        sample_idx = item['sample_index']
                        stats = reconstructed_samples[sample_idx][Fields.stats]
                        stats.setdefault(StatsKeys.video_frames_aesthetics_score, []).append(score)
                    i += batch_size
                except Exception as e:
                    if 'CUDA out of memory' in str(e):
                        logger.error(f"CUDA OOM error: {e}. Reducing batch size.")
                        batch_size = max(1, batch_size // 2)
                        torch.cuda.empty_cache()
                        continue
                    raise
        torch.cuda.synchronize()
        # logger.info(f"Batch inference completed in {time.time() - start:.2f} seconds")

        for sample in reconstructed_samples:
            stats = sample.get(Fields.stats, {})
            scores = stats.get(StatsKeys.video_frames_aesthetics_score, [])

            if not scores:
                stats[StatsKeys.video_frames_aesthetics_score] = np.array([], dtype=np.float64)
                continue
            if self.need_normalized_by_ten:
                scores = [score / 10.0 for score in scores]
            if self.reduce_mode == 'avg':
                result = np.mean(scores)
            elif self.reduce_mode == 'max':
                result = np.max(scores)
            elif self.reduce_mode == 'min': 
                result = np.min(scores)
            
            stats[StatsKeys.video_frames_aesthetics_score] = np.array(
                [result], dtype=np.float64)


        torch.cuda.empty_cache()
        logger.info(f"[Rank {device}] Peak Memory: Allocated={torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GB, Reserved={torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GB")
        
        long_dict = {
            'videos': [],
            'text': [],
            'video_frames_aesthetics_score': []
        }

        for sample in reconstructed_samples:

            long_dict['videos'].append(sample['videos'][0])
            long_dict['text'].append(sample['text'])
            long_dict['video_frames_aesthetics_score'].append(sample['__dj__stats__']['video_frames_aesthetics_score'])

        return long_dict
    
    def _safe_batch_inference(self, model, processor, batch_items, device):
        batch_frames = []
        for item in batch_items:
            try:
                frame = item['frame_obj'].to_image()
                if max(frame.size) > 1920:
                    frame = frame.resize((frame.width // 2, frame.height // 2))
                batch_frames.append(frame)
            finally:
                if hasattr(item['frame_obj'], 'close'):
                    item['frame_obj'].close()

        with torch.no_grad(), torch.autocast(device_type='cuda'):
            inputs = processor(images=batch_frames, return_tensors='pt').to(device)
            outputs = model(**inputs)
            scores = outputs.logits.cpu().numpy()
        
        return {
            'scores':[float(score) for score in scores],
            'batch_size': len(batch_frames) 
        }
    
    def process_batched(self, batch):
        
        stats_list = batch[Fields.stats]  # List[dict]
        aesthetics_scores_batch = [
            stats[StatsKeys.video_frames_aesthetics_score]
            for stats in stats_list
        ]
        keep_bools_batch = []

        for idx, aesthetics_scores in enumerate(aesthetics_scores_batch):
            print(f"[Debug] Sample {idx} - aesthetics_scores: {aesthetics_scores}")
            if len(aesthetics_scores) <= 0:
                keep_bools_batch.append(True)
                continue

            keep_bools = np.array([
                self.min_score <= aesthetics_score <= self.max_score
                for aesthetics_score in aesthetics_scores
            ])

            # different strategies
            if self.any:
                keep_bools_batch.append(keep_bools.any())
            else:
                keep_bools_batch.append(keep_bools.all())

        return keep_bools_batch

    def process_single(self, sample):

        aesthetics_scores = (
            sample)[Fields.stats][StatsKeys.video_frames_aesthetics_score]
        if len(aesthetics_scores) <= 0:
            return True

        keep_bools = np.array([
            self.min_score <= aesthetics_score <= self.max_score
            for aesthetics_score in aesthetics_scores
        ])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
    # def compute_stats_batched(self, samples, rank=None, context=False):
    #     print('Computing aesthetics scores for batch of samples')
    #     # print(f'samples: {samples}')
    #     # 检查是否已经计算过
    #     if StatsKeys.video_frames_aesthetics_score in samples[Fields.stats]:
    #         # 如果已经计算过，直接返回
    #         print('Aesthetics scores already computed, returning samples')
    #         return samples
    #     keys = samples.keys()
    #     num_samples = len(samples[Fields.stats])
    #     print(f'num_samples: {num_samples}')

    #     reconstructed_samples = [{key: samples[key][i] for key in keys} for i in range(num_samples)]

    #     # 收集所有视频，生成(video_index, video_key, video_object)列表
    #     video_list = []
    #     for sample_idx, sample in enumerate(reconstructed_samples):
    #         if self.video_key not in sample or not sample[self.video_key]:
    #             continue
    #         loaded_video_keys = sample[self.video_key]
    #         for key in loaded_video_keys:
    #             video_list.append((sample_idx, key))
    #     print(f'video_list: {video_list}')
    #     if len(video_list) == 0:
    #         return samples

    #     # 统一加载所有视频
    #     video_cache = {}
    #     for _, key in video_list:
    #         if key not in video_cache:
    #             dummy_sample = {self.video_key: [key]}
    #             _, videos = load_data_with_context(dummy_sample, context, [key], load_video)
    #             video_cache[key] = videos.get(key, None)
                

    #     # 记录每个样本每个视频帧数，供后续拆分
    #     per_sample_video_frame_counts = [[] for _ in range(num_samples)]

    #     batch = video_list  # 全部视频一次性处理

    #     batch_frame_images = []
    #     batch_info = []  # (sample_idx, video_key)
        

    #     for sample_idx, vid_key in batch:
    #         video = video_cache.get(vid_key, None)
    #         if video is None:
    #             per_sample_video_frame_counts[sample_idx].append(0)
    #             continue

    #         # 拆帧
    #         if self.frame_sampling_method == 'all_keyframes':
    #             frames, (width, height) = extract_key_frames(video)
    #         elif self.frame_sampling_method == 'uniform':
    #             frames, (width, height) = extract_video_frames_uniformly(video, self.frame_num)
    #         else:
    #             frames = []
    #         # print(f'size of {vid_key}: {width, height}')
    #         frame_images = [frame.to_image() for frame in frames]
    #         batch_frame_images.extend(frame_images)
    #         per_sample_video_frame_counts[sample_idx].append(len(frame_images))
    #         batch_info.extend([(sample_idx, vid_key)] * len(frame_images))

    #     if len(batch_frame_images) == 0:
    #         return samples

    #     print(f'Processing {len(batch_frame_images)} images from {len(batch)} videos')

    #     # 批量模型推理
    #     model, processor = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
    #     inputs = processor(images=batch_frame_images, return_tensors='pt').to(model.device)

    #     with torch.no_grad():
    #         outputs = model(**inputs)

    #     logits = outputs.logits
    #     if self.need_normalized_by_ten:
    #         logits = logits / 10.0
    #     logits = logits.cpu().numpy()

    #     # 按sample拆分结果
    #     cursor = 0
    #     for sample_idx, vid_key in batch:
    #         frame_count = per_sample_video_frame_counts[sample_idx].pop(0)
    #         if frame_count == 0:
    #             continue
    #         frame_scores = logits[cursor:cursor + frame_count]
    #         cursor += frame_count

    #         # reduce模式
    #         if self.reduce_mode == 'avg':
    #             score = float(frame_scores.mean())
    #         elif self.reduce_mode == 'max':
    #             score = float(frame_scores.max())
    #         else:
    #             score = float(frame_scores.min())

    #         if StatsKeys.video_frames_aesthetics_score not in reconstructed_samples[sample_idx][Fields.stats]:
    #             reconstructed_samples[sample_idx][Fields.stats][StatsKeys.video_frames_aesthetics_score] = []
    #         reconstructed_samples[sample_idx][Fields.stats][StatsKeys.video_frames_aesthetics_score].append(score)

    #     # 确保所有样本都有对应字段，且转成 np.array
    #     for sample in reconstructed_samples:
    #         if StatsKeys.video_frames_aesthetics_score not in sample[Fields.stats]:
    #             sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = np.array([], dtype=np.float64)
    #         else:
    #             sample[Fields.stats][StatsKeys.video_frames_aesthetics_score] = np.array(
    #                 sample[Fields.stats][StatsKeys.video_frames_aesthetics_score], dtype=np.float64)

    #     result = {key: [sample[key] for sample in reconstructed_samples] for key in keys}
    #     return result
    # def process_batched(self, samples):
    #     """
    #     直接调用父类的process_single对批量中每个样本的stats做处理，返回bool结果生成器。
    #     """
    #     return map(lambda stat: self.process_single({Fields.stats: stat}), samples[Fields.stats])
    