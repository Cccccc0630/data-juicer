# yapf: disable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import random
from time import time
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import ImageOps
from pydantic import PositiveInt

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    close_video,
    extract_key_frames,
    extract_video_frames_uniformly,
    insert_texts_after_placeholders,
    load_data_with_context,
    load_video,
    remove_non_special_tokens,
    remove_special_tokens,
)
from data_juicer.utils.model_utils import get_model, prepare_model, torch

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

simhash = LazyLoader('simhash', 'simhash-pybind')

OP_NAME = 'video_captioning_from_frames_mapper'


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFromFramesMapper(Mapper):
    """Mapper to generate samples whose captions are generated based on
    an image-to-text model and sampled video frames. Captions from different
    frames will be concatenated to a single string."""

    _accelerator = 'cuda'
    _batched_op = True

    def __init__(
        self,
        hf_img2seq: str = 'Salesforce/blip2-opt-2.7b',
        trust_remote_code: bool = False,
        caption_num: PositiveInt = 1,
        keep_candidate_mode: str = 'random_any',
        keep_original_sample: bool = True,
        prompt: Optional[str] = None,
        prompt_key: Optional[str] = None,
        frame_sampling_method: str = 'all_keyframes',
        frame_num: PositiveInt = 3,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_img2seq: model name on huggingface to generate caption
        :param caption_num: how many candidate captions to generate
            for each video
        :param keep_candidate_mode: retain strategy for the generated
            $caption_num$ candidates.

            'random_any': Retain the random one from generated captions

            'similar_one_simhash': Retain the generated one that is most
                similar to the original caption

            'all': Retain all generated captions by concatenation

        Note:
            This is a batched_OP, whose input and output type are
            both list. Suppose there are $N$ list of input samples, whose batch
            size is $b$, and denote caption_num as $M$.
            The number of total samples after generation is $2Nb$ when
            keep_original_sample is True and $Nb$ when keep_original_sample is
            False. For 'random_any' and 'similar_one_simhash' mode,
            it's $(1+M)Nb$ for 'all' mode when keep_original_sample is True
            and $MNb$ when keep_original_sample is False.

        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only generated captions in the
            final datasets and the original captions will be removed. It's True
            in default.
        :param prompt: a string prompt to guide the generation of image-to-text
            model for all samples globally. It's None in default, which means
            no prompt provided.
        :param prompt_key: the key name of fields in samples to store prompts
            for each sample. It's used for set different prompts for different
            samples. If it's none, use prompt in parameter "prompt". It's None
            in default.
        :param frame_sampling_method: sampling method of extracting frame
            videos from the videos. Should be one of
            ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number
            of which depends on the duration of the video) and the latter
            one extract specified number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param horizontal_flip: flip frame video horizontally (left to right).
        :param vertical_flip: flip frame video vertically (top to bottom).
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs.setdefault('mem_required', '20GB')
        super().__init__(*args, **kwargs)

        if keep_candidate_mode not in [
                'random_any', 'similar_one_simhash', 'all'
        ]:
            raise ValueError(
                f'Keep strategy [{keep_candidate_mode}] is not supported. '
                f'Can only be one of '
                f'["random_any", "similar_one_simhash", "all"].')

        if keep_candidate_mode in ['random_any', 'similar_one_simhash']:
            self.num_newly_generated_samples = 1
        elif keep_candidate_mode in ['all']:
            self.num_newly_generated_samples = caption_num
        else:
            self.num_newly_generated_samples = 0

        # report a warning when both prompt and prompt_key are set
        if prompt and prompt_key:
            logger.warning(
                'Both the parameter `prompt` and `prompt_key` are '
                'set. Data-Juicer will consider `prompt_key` first.')

        self.caption_num = caption_num
        self.keep_candidate_mode = keep_candidate_mode
        self.keep_original_sample = keep_original_sample
        self.prompt = prompt
        self.prompt_key = prompt_key
        self.extra_args = kwargs

        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method '
                f'[{frame_sampling_method}] is not supported. '
                f'Can only be one of ["all_keyframes", "uniform"].')

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num

        self.model_key = prepare_model(
            model_type='huggingface',
            pretrained_model_name_or_path=hf_img2seq,
            trust_remote_code=trust_remote_code
        )

    def _process_single_sample(self, ori_sample, rank=None, context=False):

        # there is no videos in this sample
        if self.video_key not in ori_sample or not ori_sample[self.video_key]:
            return []

        # the generated results
        generated_samples = [
            copy.deepcopy(ori_sample)
            for _ in range(self.num_newly_generated_samples)
        ]
        for generated_sample in generated_samples:
            generated_sample[self.text_key] = ''

        # load videos
        loaded_video_keys = ori_sample[self.video_key]
        sample, videos = load_data_with_context(ori_sample, context,
                                                loaded_video_keys, load_video)

        text = sample[self.text_key]
        offset = 0
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        for chunk in text.split(SpecialTokens.eoc):

            video_count = chunk.count(SpecialTokens.video)

            # no video or no text
            if video_count == 0 or len(chunk.strip()) == 0:
                continue
            else:
                text_with_only_special_tokens = remove_non_special_tokens(
                    chunk)
                # generate candidate caption(s) in batch manner
                generated_text_candidates_single_chunk = [
                    [] for _ in range(self.caption_num)
                ]
                for video_key in loaded_video_keys[offset:offset +
                                                   video_count]:
                    video = videos[video_key]
                    video_frame_videos_chunk = []
                    # extract frame videos
                    if self.frame_sampling_method == 'all_keyframes':
                        frames = extract_key_frames(video)
                    elif self.frame_sampling_method == 'uniform':
                        frames = extract_video_frames_uniformly(
                            video, self.frame_num)
                    else:
                        frames = []
                    frame_videos = [frame.to_image() for frame in frames]
                    for frame in frame_videos:
                        if self.horizontal_flip:
                            frame = ImageOps.mirror(frame)
                        if self.vertical_flip:
                            frame = ImageOps.flip(frame)
                        video_frame_videos_chunk.append(frame)

                    # construct prompts
                    if self.prompt_key and isinstance(
                            ori_sample[self.prompt_key], str):
                        # check prompt_key is not None, and it's a str
                        # in the sample
                        prompt_texts = [ori_sample[self.prompt_key]
                                        ] * len(video_frame_videos_chunk)
                    elif self.prompt and isinstance(self.prompt, str):
                        # check prompt is not None, and it's a str
                        prompt_texts = [self.prompt
                                        ] * len(video_frame_videos_chunk)
                    else:
                        prompt_texts = None

                    inputs = processor(
                        text=prompt_texts,
                        images=video_frame_videos_chunk,
                        return_tensors='pt',
                    ).to(model.device)
                    with torch.no_grad():
                        for i in range(self.caption_num):
                            generated_ids = model.generate(**inputs,
                                                           max_new_tokens=128,
                                                           do_sample=True)
                            generated_text = processor.batch_decode(
                                generated_ids, skip_special_tokens=True)
                            generated_text_candidates_single_chunk[i] += [
                                '. '.join([txt.strip() for txt in generated_text])
                            ]

                # 3. insert a list of generated captions into the positions of
                # subsequent placeholders in the original string
                new_generated_text_all_videos = [
                    [] for _ in range(self.num_newly_generated_samples)
                ]
                # new_generated_text_all_videos is a helper array,
                # element [i][j]
                # denotes the reduced $i$-th result for the $j$-th video

                # reduce the captions according to given mode video by video
                for j in range(video_count):
                    new_generated_text_per_video = self._reduce_captions(
                        chunk,
                        [
                            captions[j] for captions in
                            generated_text_candidates_single_chunk
                        ],
                    )
                    assert self.num_newly_generated_samples == len(
                        new_generated_text_per_video)
                    for i in range(len(new_generated_text_per_video)):
                        new_generated_text_all_videos[i].append(
                            new_generated_text_per_video[i])

                # insert the captions according to given mode
                place_holders = [SpecialTokens.video] * video_count
                for i in range(self.num_newly_generated_samples):
                    generated_text_per_chunk = insert_texts_after_placeholders(
                        original_string=text_with_only_special_tokens,
                        placeholders=place_holders,
                        new_texts=new_generated_text_all_videos[i],
                    )
                    generated_samples[i][
                        self.
                        text_key] += f'{generated_text_per_chunk}' \
                                     f'{SpecialTokens.eoc}'

                offset += video_count

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])
        return generated_samples

    def _reduce_captions(self, chunk, generated_text_candidates_single_chunk):
        generated_text_per_chunk = []
        if self.keep_candidate_mode == 'random_any':
            generated_text_per_chunk.append(
                random.choice(generated_text_candidates_single_chunk))
        elif self.keep_candidate_mode == 'all':
            generated_text_per_chunk.extend(
                generated_text_candidates_single_chunk)
        elif self.keep_candidate_mode == 'similar_one_simhash':
            from ..deduplicator.document_simhash_deduplicator import (
                DocumentSimhashDeduplicator,
            )

            ori_normal_text = remove_special_tokens(chunk)
            # using a simhash OP to calculate their similarity
            # NOTE: simhash is just one method to calculate the similarities
            # between texts, but not the most accurate one. More methods (e.g.
            # embedding-based, ...) will be added.
            op_simhash = DocumentSimhashDeduplicator(window_size=2,
                                                     **self.extra_args)
            ori_text_hash = np.uint64(
                op_simhash.compute_hash({op_simhash.text_key:
                                         ori_normal_text})[HashKeys.simhash])
            generated_text_hashes = [
                np.uint64(
                    op_simhash.compute_hash(
                        {op_simhash.text_key:
                         candidate_text})[HashKeys.simhash])
                for candidate_text in generated_text_candidates_single_chunk
            ]
            hamming_distances = [
                simhash.num_differing_bits(ori_text_hash, generated_text_hash)
                for generated_text_hash in generated_text_hashes
            ]
            max_index = min(range(len(hamming_distances)),
                            key=hamming_distances.__getitem__)
            generated_text_per_chunk.append(
                generated_text_candidates_single_chunk[max_index])
        return generated_text_per_chunk
    
    def load_model(self, rank=None):
        model, processor = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
        return model, processor
    
    def process_batched(self, samples, rank=None, context=False):
        # Step 1: 重构样本列表
        reconstructed_samples = [
            {key: samples[key][i] for key in samples}
            for i in range(len(samples[self.text_key]))
        ]
        print(f"[Debug] Reconstructed samples counts: {len(reconstructed_samples)}")

        # Step 2: 收集所有视频任务
        video_tasks = []
        for sample_idx, sample in enumerate(reconstructed_samples):
            if self.video_key not in sample or not sample[self.video_key]:
                continue
            for vid_key in sample[self.video_key]:
                video_tasks.append({
                    'sample_index': sample_idx,
                    'vid_key': vid_key,
                    'sample': sample
                })

        if not video_tasks:
            return samples

        # Step 3: 使用线程池并发抽帧，生成缓存
        def extract_frames(task: Dict) -> Tuple[str, Dict]:
            vid_key = task.get('vid_key', '')
            if not vid_key:
                return vid_key, {'frames': [], 'size': (0, 0)}

            for attempt in range(3):
                try:
                    dummy_sample = {self.video_key: [vid_key]}
                    _, videos = load_data_with_context(dummy_sample, False, [vid_key], load_video)
                    video = videos.get(vid_key) if videos else None
                    if video is None:
                        return vid_key, {'frames': [], 'size': (0, 0)}

                    if self.frame_sampling_method == 'all_keyframes':
                        frames, size = extract_key_frames(video)
                    elif self.frame_sampling_method == 'uniform':
                        frames, size = extract_video_frames_uniformly(video, self.frame_num)
                    else:
                        frames = []
                        size = (0, 0)
                    close_video(video)

                    return vid_key, {
                        'raw_frames': [(f, size) for f in frames],
                        'size': size
                    }
                except Exception as e:
                    logger.error(f"Error extracting frames for {vid_key}: {e}")
                    time.sleep(1)
            return vid_key, {'frames': [], 'size': (0, 0)}

        max_workers = 64
        video_frame_cache = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_frames, task): task['vid_key'] for task in video_tasks}
            for future in as_completed(futures):
                vid_key, result = future.result()
                if vid_key:
                    video_frame_cache[vid_key] = result

        torch.cuda.synchronize()
        logger.info(f"Frame extraction completed, total videos cached: {len(video_frame_cache)}")

        # Step 4: 准备模型与处理器
        model, processor = self.load_model(rank=rank)
        device = model.device

        batch_inputs = []
        generated_result_map = defaultdict(lambda: [[] for _ in range(self.caption_num)])

        # Step 5: 构建所有帧输入列表，基于缓存避免重复抽帧
        for (sample_idx, vid_key) in [(t['sample_index'], t['vid_key']) for t in video_tasks]:
            sample = reconstructed_samples[sample_idx]
            text = sample[self.text_key]

            for chunk in text.split(SpecialTokens.eoc):
                video_count = chunk.count(SpecialTokens.video)
                if video_count == 0 or len(chunk.strip()) == 0:
                    continue

                text_with_only_special_tokens = remove_non_special_tokens(chunk)
                prompt = self._get_prompt(sample)

                frames_info = video_frame_cache.get(vid_key)
                if not frames_info:
                    logger.warning(f"No cached frames found for video {vid_key}")
                    continue

                frames_and_size = frames_info.get('raw_frames', [])
                if not frames_and_size:
                    continue

                frames, size = zip(*frames_and_size)

                for frame in frames:
                    img = frame.to_image()
                    if self.horizontal_flip:
                        img = ImageOps.mirror(img)
                    if self.vertical_flip:
                        img = ImageOps.flip(img)

                    batch_inputs.append({
                        'image': img,
                        'prompt': prompt,
                        'chunk': chunk,
                        'sample_idx': sample_idx,
                        'vid_key': vid_key,
                        'text_with_tokens': text_with_only_special_tokens
                    })

        logger.info(f"[Batch] Total frames for inference: {len(batch_inputs)}")

        if not batch_inputs:
            return samples

        # Step 6: 批量推理
        batch_size = 32
        i = 0
        while i < len(batch_inputs):
            sub_batch = batch_inputs[i:i + batch_size]
            images = [bi['image'] for bi in sub_batch]
            prompts = [bi['prompt'] for bi in sub_batch] if sub_batch[0]['prompt'] else None

            inputs = processor(text=prompts, images=images, return_tensors='pt').to(device)

            with torch.no_grad():
                for k in range(self.caption_num):
                    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True)
                    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    for j, gen_text in enumerate(generated_texts):
                        bi = sub_batch[j]
                        generated_result_map[(bi['sample_idx'], bi['chunk'])][k].append(gen_text.strip())

            i += batch_size
        logger.info(f"[Rank {device}] Peak Memory: Allocated={torch.cuda.max_memory_allocated(device) / (1024 ** 3):.2f} GB, Reserved={torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GB")
    
        # Step 6: 组装输出
        samples_after_generation = []
        for sample_idx, sample in enumerate(reconstructed_samples):
            if self.keep_original_sample:
                samples_after_generation.append(sample)

            text = sample[self.text_key]
            offset = 0
            generated_sample_group = [
                copy.deepcopy(sample) for _ in range(self.num_newly_generated_samples)
            ]
            for s in generated_sample_group:
                s[self.text_key] = ''

            for chunk in text.split(SpecialTokens.eoc):
                video_count = chunk.count(SpecialTokens.video)
                if video_count == 0 or len(chunk.strip()) == 0:
                    continue

                text_with_only_special_tokens = remove_non_special_tokens(chunk)
                place_holders = [SpecialTokens.video] * video_count

                # 取该 sample_idx 和 chunk 的所有生成文本，结构是 caption_num 个列表，每个列表包含所有生成的文本
                gen_candidates = generated_result_map.get((sample_idx, chunk), [[] for _ in range(self.caption_num)])

                # 调用 reduce_captions，得到 num_newly_generated_samples 个列表，每个列表长度是 video_count（对应占位符数），每个元素是 str 或 list[str]

                gen_text_all = self._reduce_captions(chunk, gen_candidates)
                gen_text_all = self.normalize_reduce_output(gen_text_all, video_count, self.num_newly_generated_samples)
                # print(f"[Debug] video_count={video_count}, placeholders={place_holders}")
                # print(f"[Debug] gen_text_all={gen_text_all}, len={len(gen_text_all)}")
                for i in range(len(gen_text_all)):
                    new_text = insert_texts_after_placeholders(
                        original_string=text_with_only_special_tokens,
                        placeholders=place_holders,
                        new_texts=gen_text_all[i],  # 一定是 List[str]，长度 == video_count
                    )
                    generated_sample_group[i][self.text_key] += new_text + SpecialTokens.eoc

                offset += video_count

            samples_after_generation.extend(generated_sample_group)

        # Step 7: 转回 dict of lists
        keys = samples_after_generation[0].keys()
        res_samples = {key: [s[key] for s in samples_after_generation] for key in keys}
        # print(res_samples)
        return res_samples

    
    def _get_prompt(self, sample):
        if self.prompt_key and isinstance(sample.get(self.prompt_key), str):
            return sample[self.prompt_key]
        elif self.prompt and isinstance(self.prompt, str):
            return self.prompt
        return None
    
    def normalize_reduce_output(self, gen_texts, video_count, num_samples):
        """
        gen_texts: List[str] or List[List[str]] from _reduce_captions
        video_count: int, 占位符个数
        num_samples: int, num_newly_generated_samples

        返回 List[List[str]]，符合格式
        """
        # 如果已经是List[List[str]]，检查长度匹配，否则转成需要的格式
        if len(gen_texts) == 0:
            # 空则返回空嵌套列表
            return [[] for _ in range(num_samples)]

        if isinstance(gen_texts[0], list):
            # 是二维结构，补齐长度
            ret = []
            for item in gen_texts:
                if len(item) < video_count:
                    ret.append(item + [''] * (video_count - len(item)))
                else:
                    ret.append(item[:video_count])
            # 补齐外层长度
            while len(ret) < num_samples:
                ret.append([''] * video_count)
            return ret

        else:
            # 是一维字符串列表，包装成二维，内层长度video_count
            ret = []
            for i in range(num_samples):
                # 按顺序或者全部一样的，都可以
                # 这里假设gen_texts长度>=num_samples，或者循环取
                text = gen_texts[i] if i < len(gen_texts) else ''
                ret.append([text] * video_count)
            return ret
    # def process_batched(self, samples, rank=None, context=False):
    #     """
    #     :param samples:
    #     :return:

    #     Note:
    #         This is a batched_OP, whose the input and output type are
    #         both list. Suppose there are $N$ input sample list with batch
    #         size as $b$, and denote caption_num as $M$.
    #         the number of total samples after generation is $2Nb$
    #         for 'random_any' and 'similar_one' mode,
    #         and $(1+M)Nb$ for 'all' mode.
    #     """
    #     # reconstruct samples from "dict of lists" to "list of dicts"
    #     reconstructed_samples = []
    #     for i in range(len(samples[self.text_key])):
    #         reconstructed_samples.append(
    #             {key: samples[key][i]
    #              for key in samples})
    #     samples_after_generation = []
    #     # do generation for each sample within the batch
    #     for ori_sample in reconstructed_samples:
    #         if self.keep_original_sample:
    #             samples_after_generation.append(ori_sample)
    #         generated_samples = self._process_single_sample(ori_sample,
    #                                                         rank=rank,
    #                                                         context=context)
    #         if len(generated_samples) != 0:
    #             samples_after_generation.extend(generated_samples)
    #     # reconstruct samples from "list of dicts" to "dict of lists"
    #     keys = samples_after_generation[0].keys()
    #     res_samples = {}
    #     for key in keys:
    #         res_samples[key] = [s[key] for s in samples_after_generation]

    #     return res_samples