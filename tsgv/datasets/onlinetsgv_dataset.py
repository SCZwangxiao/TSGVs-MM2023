# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
from copy import deepcopy
import warnings
from collections import OrderedDict
from tqdm import tqdm
from math import ceil, log
import multiprocessing as mp

import mmcv
import h5py
import torch
from scipy import signal
import numpy as np
from torch import distributed

from .tsgv_dataset import TSGVDataset
from .builder import DATASETS
from ..core.evaluation import iou, nms
from ..core.evaluation import frame_level_map, recall_at_iou_at


def binSearch(a, val):
    # Assume a is in ascending order
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if val < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo - 1


@DATASETS.register_module()
class OnlineTSGVDataset(TSGVDataset):
    """Dataset for online temporal sentence grounding in videos.

    Based on BaseDataset, the dataset chunking the original video to imitate
    the streaming videos.

    Args:
        ann_file (str ｜ list[str]): Path | list of paths to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        short_memory_sample_length (int): Length of video feature sequence sampled 
            in the short memory.
        data_prefix (str): Path to a directory where features are held.
        video_feat_filename (str): hdf5 filename of video features.
        sentence_feat_filename (str): hdf5 filename of sentence features.
            Default: 'bert_features.hdf5'.
        bert_model (str): BERT model name in huggingface transformers.
            Default: 'bert-base-uncased'.
        max_sentence_length (int): Maximum number of tokens of the sentence.
            Default: 20.
        long_memory_sample_length (int): Length of video feature sequence sampled 
            in the long memory. It is used for models with long term memory, like LSTR.
            Default: 0.
        future_memory_sample_length (int): Length of video feature sequenc sampled 
            in the future. It is used for models with future anticipation, like TRN, OadTR.
            Default: 0.
        short_memory_stride (int): Number of sampling strides for short memory.
            Default: 1.
        long_memory_stride (int): Number of sampling strides for long memory.
            Default: 1.
        future_memory_stride (int): Number of sampling strides for future memory.
            Default: 1.
        load_future_memory (bool): Whether to load future memory feature.
            Default: False.
        gaussian_label (bool): Whether to set the label mask be gaussian distribution.
            Default: True.
        gaussian_label_width (list[int]): The ``alpha_0`` in gaussian distribution.
            Default: [0.25, 0.25, 0.21].
        smooth_filter_win_len (int): Window length of streaming results smoothing filter.
            Default: 3.
        downsample_stride (int): Downsample stride for streaming results.
            Default: 4.
        min_num_segments (int): Minimal number of clips after downsampling.
            Default: 128.
        group_size_2d (int): Group size of 2d temporal proposers.
            Default: 4.
        nms_thresh (float): Threshold for non maximal supression.
            Default: 0.5.
        num_workers (int): Number of the workers for post processing.
            Default: 8.
        in_memory (bool): Store True when loading features into RAM.
            Default: True.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        portion (float | int | list<int rank, int N>): If ``float``, it indicates the raito of the 
            annotations to be used. If ``int``, it indicates the first 
            ``portion`` annotations to be used. If ``list<rank, N>, in indicates
            the rank-th sample of a total of N.
            Defailt: 1.0
    """

    def __init__(self, 
                 ann_file, 
                 pipeline, 
                 short_memory_sample_length, 
                 data_prefix, 
                 video_feat_filename, 
                 split,
                 language_model='bert-base-uncased', 
                 max_sentence_length=20,
                 long_memory_sample_length=0, 
                 future_memory_sample_length=0,
                 short_memory_stride=1, 
                 long_memory_stride=1, 
                 future_memory_stride=1, 
                 chunk_options=dict(chunk_interval=1,
                                    pos_expansion_ratio=1.,
                                    neg_pos_ratio=None,
                                    cache_chunk_info=True),
                 load_future_memory=False,
                 gaussian_label=True,
                 asym_gaussian_label=False,
                 gaussian_label_width=[0.25,0.25,0.21],
                 proposal_options=dict(smooth_filter_win_len=3,
                                       downsample_stride=4,
                                       min_num_segments=128,
                                       max_proposal_len=None,
                                       group_size_2d=4),
                 nms_thresh=0.5,
                 num_workers=8,
                 video_in_memory=True, 
                 sent_in_memory=True, 
                 cache_anno_label=True,
                 test_mode=False,
                 portion=1.0):
        super().__init__(ann_file=ann_file, 
                         pipeline=pipeline, 
                         data_prefix=data_prefix, 
                         video_feat_filename=video_feat_filename, 
                         split=split, 
                         language_model=language_model, 
                         max_sentence_length=max_sentence_length,
                         num_segments=None,
                         proposal_options=dict(type='span_based',
                                               gaussian_label=gaussian_label,
                                               gaussian_label_width=gaussian_label_width,
                                               asym_gaussian_label=asym_gaussian_label,
                                               cache_anno_label=cache_anno_label),
                         nms_thresh=nms_thresh,
                         num_workers=num_workers,
                         video_in_memory=video_in_memory, 
                         sent_in_memory=sent_in_memory,
                         test_mode=test_mode,
                         portion=portion)
        # Chunk memory options
        self.short_memory_sample_length = short_memory_sample_length
        self.long_memory_sample_length = long_memory_sample_length
        self.future_memory_sample_length = future_memory_sample_length
        self.short_memory_stride = short_memory_stride
        self.long_memory_stride = long_memory_stride
        self.future_memory_stride = future_memory_stride
        self.load_future_memory = load_future_memory
        # Chunk options
        self.chunk_interval = int(chunk_options.get('chunk_interval', 1))
        if not self.test_mode and chunk_options.get('neg_pos_ratio', None):
            self.pos_expansion_ratio = chunk_options.get('pos_expansion_ratio', 1.)
            self.neg_pos_ratio = chunk_options['neg_pos_ratio']
            assert self.neg_pos_ratio - (self.pos_expansion_ratio - 1) >= .0
            self.cache_chunk_info = True
        else:
            self.pos_expansion_ratio = None
            self.neg_pos_ratio = None
            self.cache_chunk_info = chunk_options.get('cache_chunk_info', True)
        # Proposal options
        self.smooth_filter_win_len = proposal_options.get('smooth_filter_win_len', 3)
        self.downsample_stride = proposal_options.get('downsample_stride', 4)
        self.min_num_segments = proposal_options.get('min_num_segments', 128)
        self.max_proposal_len = proposal_options.get('max_proposal_len', None)
        self.group_size_2d = proposal_options.get('group_size_2d', 4)
        if self.test_mode:
            self.future_memory_sample_length = 0
            self.load_future_memory = False
        self._chunk_infos, self._len = self.chunk_all_videos()
        if self.test_mode:
            self.mask2d_info = self.get_mask2d(
                self.group_size_2d, self.downsample_stride, self.chunk_interval,
                self.min_num_segments, self.max_proposal_len, self._max_duration_frame)

    def re_generate_dataset(self):
        """Regenerate the datasets for data augmentation.
        
        Not that the pytorch ``Dataloader`` uses a reference to the 
        ``Dataset``, so the dataset regeneration will be reflected
        in all dataset copies in the dataloader. See more details in: 
        https://discuss.pytorch.org/t/solved-will-change-in-dataset-be-reflected-on-dataloader-automatically/10206
        """
        self._chunk_infos = None
        self._chunk_infos, self._len = self.chunk_all_videos()

    def chunk_all_videos(self):
        """Chunk the videos to imitate streaming video.
            Caution for the validty of these three boundaries:
            In [short_memory_start, short_memory_end], ``short_memory_start`` may be invalid.
            In [long_memory_start, long_memory_end], both boundary may be invalid.
            [future_memory_start, future_memory_end] are always valid.
            All the invalid boundary could only be < 0, and wound not be > duration_frame-1.
        """
        if self.cache_chunk_info:
            chunk_infos = [] # store each chunk info
        else:
            chunk_infos = {} # store the boundary info between each anno
            chunk_boundary_indices = []
        total_length = 0
        if distributed.get_rank() == 0:
            pbar = tqdm(total=self.anno_infos.shape[0], desc='Chunking videos...', mininterval=10.)
        for anno_idx, anno_info in enumerate(self.anno_infos):
            duration_frame = anno_info['duration_frame']
            # Get sampling offset
            if not self.test_mode:
                interval = self.chunk_interval - 1 + self.short_memory_sample_length
                offset = np.random.randint(interval)
                # The offset acts as an augmentation. 
            else:
                offset = 0
                interval = self.chunk_interval
            if not self.cache_chunk_info:
                chunk_infos[total_length] = [anno_idx, offset]
                chunk_boundary_indices.append(total_length)
            # Get in-anno, out-anno sampling ratio
            if not self.test_mode and self.neg_pos_ratio:
                framestamp = np.array(anno_info['timestamp']) / anno_info['duration'] * (duration_frame-1)
                s, e = framestamp.tolist()
                pos_chunk_cnt = e - s
                alpha_pos, alpha_neg = 1 + self.pos_expansion_ratio, 1 - self.pos_expansion_ratio
                expanded_framestamp = [
                    max(0, int((alpha_pos*s + alpha_neg*e) / 2)),
                    min(duration_frame, int((alpha_neg*s + alpha_pos*e) / 2))
                ]
                out_neg_chunk_cnt = (self.neg_pos_ratio - self.pos_expansion_ratio + 1) * pos_chunk_cnt
                out_sample_ratio = min(
                    1.0, 
                    out_neg_chunk_cnt / (duration_frame - offset - self.pos_expansion_ratio * pos_chunk_cnt))
                sample_score = np.random.uniform(.0, 1.0, duration_frame-offset)
                sample_flag = (sample_score <= out_sample_ratio)
                sample_flag[expanded_framestamp[0]:expanded_framestamp[1]] = True
                sample_flag = sample_flag[::interval]
            else:
                sample_flag = np.ones(max(1, duration_frame-offset)//interval+1).astype(np.bool)
            # Start sampling
            for idx, short_memory_start in enumerate(
                range(offset+1-self.short_memory_sample_length, 
                      duration_frame+1-self.short_memory_sample_length, 
                      interval)):
                # Caution, short_memory_start and short_memory_end is the index for frame feature sequences,
                # Specifically, we sample features in [short_memory_start, short_memory_end).
                # Therefore, we put a ``1``.
                if sample_flag[idx]:
                    chunk_info = [anno_idx, short_memory_start]
                    # For saving RAM, we only store 'short_memory_start', using list instead of dict.
                    if self.cache_chunk_info:
                        chunk_infos.append(chunk_info)
                    total_length += 1
            if distributed.get_rank() == 0:
                pbar.update(1)
        if distributed.get_rank() == 0:
            pbar.close()
        # Use numpy.array to replace python list to avoid copy-on-write
        # in dataloader workers caused by Python multiprocessing.
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        if self.cache_chunk_info:
            chunk_infos = np.array(chunk_infos)
        else:
            self.chunk_boundary_indices = np.array(chunk_boundary_indices)
        return chunk_infos, total_length

    def get_chunk_info(self, idx):
        if self.cache_chunk_info:
            return self._chunk_infos[idx]
        else:
            i = binSearch(self.chunk_boundary_indices, idx)
            anno_start_idx = self.chunk_boundary_indices[i]
            anno_idx, offset = self._chunk_infos[anno_start_idx]
            idx = idx - anno_start_idx
            short_memory_start = offset + 1 - self.short_memory_sample_length + idx * self.chunk_interval
            return [anno_idx, short_memory_start]

    @staticmethod
    def get_mask2d(group_size, 
                   downsample_stride, 
                   chunk_interval,
                   min_num_segments, 
                   max_proposal_len,
                   max_duration_frame):
        """
        Generate compressed indices for 2D temporal map proposal.

        Return:
            mask2d_info
        """
        def is_in_2d_mask(a, b, g=8):
            k = ceil(log((b-a+1)/g, 2))
            s = pow(2, k-1)
            s_ = 0 if k == 1 else pow(2, k+2) - 1
            return (a % s == 0) &  ((b-s_) % s == 0)

        group_intervals = []
        intercepts = []
        max_len = max(
            min_num_segments, 
            max_duration_frame // chunk_interval // downsample_stride)
        max_proposal_len = max_proposal_len if max_proposal_len else max_len
        prev_intercept = -1
        group_interval = 0
        for intercept in range(max_len):
            if is_in_2d_mask(0, intercept, group_size):
                group_interval = intercept - prev_intercept
                prev_intercept = intercept
                group_intervals.append(group_interval)
                intercepts.append(intercept)
                # Compress the start & end points below:
                """
                Return:
                    mask2d_s (np.array): start index of 2D proposal.
                    mask2d_e (np.array): end index of 2D proposal.
                
                for a in range(0, max_len, group_interval):
                    b = a + intercept
                    if b >= max_len or b - a > max_proposal_len:
                        continue
                    else:
                        mask2d_s.append(a)
                        mask2d_e.append(b)
                """
        group_intervals = np.array(group_intervals)
        intercepts = np.array(intercepts)
        mask2d_info = dict(max_len=max_len,
                           max_proposal_len=max_proposal_len,
                           group_intervals=group_intervals,
                           intercepts=intercepts)
        return mask2d_info

    def get_memory_labels(self, boundary, anno_framestamp, stride):
        """Get the video features, zero padding if needed.
        Args:
            boundary (list[int]): frame boundary of memory features.
            anno_framestamp (list[float]):
            stride (int):
        """
        s, e = boundary
        mem_len = e - s
        s_gt, e_gt = anno_framestamp
        if self.gaussian_label:
            span_length = e_gt - s_gt
            t = np.arange(s, e, stride)
            sigma_s = self.gaussian_label_width[0] * span_length
            sigma_e = self.gaussian_label_width[1] * span_length
            sigma_se = self.gaussian_label_width[2] * span_length
            s_label = self.__gaussian_label__(t, s_gt, sigma_s, 'start')
            e_label = self.__gaussian_label__(t, e_gt, sigma_e, 'end')
            se_label = self.__gaussian_label__(t, (s_gt+e_gt)/2, sigma_se, 'semantic')
        else:
            s_gt, e_gt = round(s_gt), round(e_gt)
            s_label = np.zeros(mem_len)
            e_label = np.zeros(mem_len)
            se_label = np.zeros(mem_len)
            diff_s = s_gt - s
            diff_e = e_gt - s
            if diff_s >= 0 and diff_s < mem_len:
                s_label[s_gt-s] = 1
            if diff_e >= 0 and diff_e < mem_len:
                e_label[e_gt-s] = 1
            diff_s = max(0, diff_s)
            diff_e = min(mem_len, diff_e)
            if diff_s < mem_len:
                if diff_e < mem_len:
                    se_label[diff_s:diff_e+1] = 1
                else:
                    se_label[diff_s:] = 1
            if stride != 1:
                s_label = s_label[::stride]
                e_label = e_label[::stride]
                se_label = se_label[::stride]
        return s_label, e_label, se_label

    def get_chunk_features(self, memory_frame_boundary, duration_frame, video_name):
        """Get features of the whole video according to key ``video_name``."""
        s, e = memory_frame_boundary
        s_valid = max(0, s)
        e_valid = min(duration_frame, e)
        if self.video_in_memory:
            vid_feat_valid = self.video_name2video_feature.loc[video_name][0][s_valid:e_valid]
        else:
            with h5py.File(self.video_feat_file, 'r') as VideoFeatFile:
                vid_feat_valid = VideoFeatFile[video_name][s_valid:e_valid].squeeze().astype(np.float32)
        vid_feat = np.zeros((e-s, vid_feat_valid.shape[-1]), dtype=np.float32)
        feat_mask = np.ones(e-s, dtype=np.bool)
        vid_feat[s_valid-s:e_valid-s,:] = vid_feat_valid
        feat_mask[s_valid-s:e_valid-s] = False
        return vid_feat, feat_mask

    def prepare_frames(self, idx):
        """Prepare the frames for training or testing given the index."""
        chunk_info = self.get_chunk_info(idx)
        
        # Get basic info
        anno_idx, short_memory_start = chunk_info
        anno_id = self.anno_infos[anno_idx]['anno_id']
        video_name = self.anno_infos[anno_idx]['video_name']
        short_memory_end = short_memory_start + self.short_memory_sample_length
        short_memory_frame_boundary = [short_memory_start, short_memory_end]
        whole_memory_frame_boundary = [short_memory_start, short_memory_end]
        if self.long_memory_sample_length > 0:
            long_memory_start = short_memory_start - self.long_memory_sample_length
            long_memory_end = short_memory_start
            long_memory_frame_boundary = [long_memory_start, long_memory_end]
            whole_memory_frame_boundary[0] -= self.long_memory_sample_length
        if self.future_memory_sample_length > 0:
            future_memory_start = short_memory_end
            future_memory_end = short_memory_end + self.future_memory_sample_length
            future_memory_boundary = [future_memory_start, future_memory_end]
            whole_memory_frame_boundary[1] += self.future_memory_sample_length

        # Meta data
        results = dict()
        results['anno_idx'] = anno_idx
        results['short_memory_start'] = short_memory_start
        results['video_name'] = self.video_name2video_idx[video_name]
            # Cannot use hash(), because seed differs between processes
        duration = self.anno_infos[anno_idx]['duration']
        duration_frame = self.anno_infos[anno_idx]['duration_frame']
        timestamp = self.anno_infos[anno_idx]['timestamp']
        results['anno_framestamp'] = (np.array(timestamp) / duration * (duration_frame-1)).tolist()
        results['memory_framestamp'] = list(
            range(short_memory_start, short_memory_end))
        
        # Sentence feature
        sent_feat, sentence_length = self.get_sent_features(anno_id)
        results['sentence_length'] = sentence_length
        results['sentence_features'] = sent_feat

        # Video feature
        chunk_memory, chunk_memory_mask = self.get_chunk_features(
            whole_memory_frame_boundary, duration_frame, video_name)
        ## Long-term memory
        if self.long_memory_sample_length > 0:
            long_memory = chunk_memory[:self.long_memory_sample_length:self.long_memory_stride, :]
            long_memory_mask = chunk_memory_mask[:self.long_memory_sample_length:self.long_memory_stride]
            chunk_memory = chunk_memory[self.long_memory_sample_length:, :]
            chunk_memory_mask = chunk_memory_mask[self.long_memory_sample_length:]
            results['long_memories'] = long_memory
            results['long_memory_masks'] = long_memory_mask
        ## Future memory
        if self.future_memory_sample_length > 0:
            future_memory = chunk_memory[self.short_memory_sample_length::self.future_memory_stride, :]
            future_memory_mask = chunk_memory_mask[self.short_memory_sample_length::self.future_memory_stride]
            chunk_memory = chunk_memory[:self.short_memory_sample_length, :]
            chunk_memory_mask = chunk_memory_mask[:self.short_memory_sample_length]
            if self.load_future_memory:
                results['future_memories'] = future_memory
            results['future_memory_masks'] = future_memory_mask
        ## Short-term memory
        results['short_memories'] = chunk_memory[::self.short_memory_stride,:]
        results['short_memory_masks'] = chunk_memory_mask[::self.short_memory_stride]
        
        # Label
        if not self.test_mode:
            ## Short-term label
            short_start_label, short_end_label, short_semantic_label = \
                self.get_memory_labels(short_memory_frame_boundary, 
                                       results['anno_framestamp'], self.short_memory_stride)
            results['start_label'] = short_start_label
            results['end_label'] = short_end_label
            results['semantic_label'] = short_semantic_label
            ## Future anticapation label
            if self.future_memory_sample_length > 0:
                future_start_label, future_end_label, future_semantic_label = \
                    self.get_memory_labels(future_memory_boundary, 
                                           results['anno_framestamp'], self.future_memory_stride)
                results['future_start_label'] = future_start_label
                results['future_end_label'] = future_end_label
                results['future_semantic_label'] = future_semantic_label

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return self._len

    def read_streaming_results(self, results):
        """Read streaming predictions to get video-annotation level results.
        
        Args:
            results (list[np.array]): The ``np.array`` has three elements: 
            start probability, end probability, and within probability.
        Return:
            results_list (list[dict]): The ``dict`` is the annotation info
            with the prediction probabilities and proposals.
        """
        results_list = []
        prev_anno_idx = None
        anno_score = []
        for idx in range(len(results)):
            score = results[idx]
            info = self.get_chunk_info(idx)
            anno_idx = info[0]
            if prev_anno_idx is None: # Cold start
                prev_anno_idx = anno_idx
            if prev_anno_idx == anno_idx: # same annotation (video-query pair)
                anno_score.append(score)
            else:
                if self.future_memory_sample_length > 0:
                    anno_score.extend(
                        [np.zeros(3) for _ in range(self.future_memory_sample_length)])
                anno_score = np.array(anno_score)
                anno_info = dict(anno_idx=prev_anno_idx,
                                 anno_score=anno_score)
                results_list.append(anno_info)
                prev_anno_idx = anno_idx
                anno_score = [score]
        # Process the last frames of the last annotation
        if self.future_memory_sample_length > 0:
            anno_score.extend(
                [np.zeros(3) for _ in range(self.future_memory_sample_length)])
        anno_score = np.array(anno_score)
        anno_info = dict(anno_idx=prev_anno_idx,
                         anno_score=anno_score)
        results_list.append(anno_info)
        return results_list

    @staticmethod
    def downsample_scores(anno_score, 
                          smooth_window_len=3,
                          downsample_stride=4,
                          min_num_segments=128):
        """Downsample the predicted scores time sequence.
        Args:
            anno_score (np.array):
            min_window_length (int): Window length of Mean filter.
        """
        total_num_clips = anno_score.shape[0]
        if total_num_clips > min_num_segments:
            # Smoothing
            anno_score[:, 0] = signal.medfilt(anno_score[:, 0], 
                                              smooth_window_len)
            anno_score[:, 1] = signal.medfilt(anno_score[:, 1], 
                                              smooth_window_len)
            anno_score[:, 2] = signal.medfilt(anno_score[:, 2], 
                                              smooth_window_len)
            # Downsampling
            num_segments = max(min_num_segments, total_num_clips//downsample_stride)
            if total_num_clips != num_segments:
                indices = np.linspace(0, total_num_clips-1, num_segments).astype(np.int32)
                anno_score_downsampled = anno_score[indices]
            else:
                anno_score_downsampled = anno_score
        else:
            # No downsample for short videos
            anno_score_downsampled = anno_score
            num_segments = total_num_clips
        return anno_score_downsampled, num_segments

    @staticmethod
    def score2proposals(anno_score, duration, mask2d_info):
        """Convert prediction score to moment proposals.

        Args:
            anno_score (np.array): Shape: [T, 3]. Each column corresponds to:
                start probability, end probability, and within probability.
            duration (float): Length of the video, in seconds.
            mask2d_info
        Reture:
            proposals (np.array): Shape: [num_proposals, 3]. Each column corresponds to:
                start, end, confidence scores.
        """
        T = anno_score.shape[0]
        start_score = anno_score[:,0][:,None]
        end_score = anno_score[:,1][:,None]
        # [T, 1]
        # Get proposal start & ends
        max_len = min(mask2d_info['max_len'], T)
        max_proposal_len = mask2d_info['max_proposal_len']
        group_intervals = mask2d_info['group_intervals']
        intercepts = mask2d_info['intercepts']
        starts = []
        ends = []
        for group_interval, k in zip(group_intervals, intercepts):
            for s in range(0, max_len, group_interval):
                e = s + k
                if e >= max_len or e - s > max_proposal_len:
                    continue
                else:
                    starts.append(s)
                    ends.append(e)
        starts = np.array(starts)
        ends = np.array(ends)
        # for_loop_select
        scores = np.zeros_like(starts).astype(np.float32)
        for k, (i, j) in enumerate(zip(starts, ends)):
            scores[k] = start_score[i] * end_score[j]
        # Convert start & and from frame id to seconds
        starts = starts.astype(np.float32) / (T-1) * duration
        ends = ends.astype(np.float32) / (T-1) * duration
        proposals = np.stack([starts, ends, scores], axis=1)
        # [num_proposals, 3]
        return proposals

    def score2proposals_multiprocess(self, rank, task_queue, result_queue):
        while True:
            idx, res, mask2d_info, top_n, smooth_filter_win_len, \
                downsample_stride, nms_thresh = task_queue.get()
            if idx == -1:
                # stop signal
                break
            anno_score = res['anno_score']
            anno_idx = res['anno_idx']
            duration = self.anno_infos[anno_idx]['duration']
            anno_score, num_segments = self.downsample_scores(
                anno_score, smooth_filter_win_len, downsample_stride)
            proposals = self.score2proposals(anno_score, duration, mask2d_info)
            moments, scores = proposals[:,:2], proposals[:,2]
            moments, scores = nms(moments, scores, thresh=nms_thresh, topn=top_n)
            proposals = np.concatenate([moments, scores[:, None]], axis=1)
            result_queue.put((idx, proposals))

    def get_proposals(self, results_list, top_n=None):
        """Get proposals following the procedure:
            smoothing + downsampling + 2d proposal

        The processing is done using multiprocessing
        """
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        # Create workers
        proc_list = []
        for i in range(self.num_workers):
            p = mp.Process(target=self.score2proposals_multiprocess, 
                           args=(i+1, task_queue, result_queue))
            p.start()
            proc_list.append(p)
        # Create tasks
        for idx, res in enumerate(results_list):
            task_queue.put((idx, res, self.mask2d_info, top_n,
                            self.smooth_filter_win_len, self.downsample_stride, self.nms_thresh))
        for i in range(self.num_workers):
            # stop signal
            task_queue.put((-1, None, self.mask2d_info, top_n,
                            self.smooth_filter_win_len, self.downsample_stride, self.nms_thresh))
        # Get results
        for _ in range(len(results_list)):
            idx, proposals = result_queue.get()
            results_list[idx]['proposals'] = proposals
        # Clean
        for p in proc_list:
            p.join()
        task_queue.close()
        result_queue.close()
        task_queue.join_thread()
        result_queue.join_thread()

        return results_list

    def results2json(self, results, top_n=None):
        """Read model results to get the final prediction results in json.
        
        Args:
            results (list[np.array]): The ``np.array`` has three elements: 
            start probability, end probability, and within probability.
        Return:
            results_list (list[dict]): The ``dict`` is the annotation info
            with the prediction probabilities and proposals.
        """
        results_list = self.read_streaming_results(results)
        results_list = self.get_proposals(results_list, top_n)
        return results_list

    def evaluate(
            self,
            results,
            metrics='R@N,IoU=M',
            metric_options={'R@N,IoU=M': 
                dict(recall_at=[1,5],iou_at=[0.3,0.5,0.7])},
            logger=None):
        """Evaluation in feature dataset.

        Args:
            results (list[dict]): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'R@N,IoU=M'.
            metric_options (dict): Dict for metric options. Options are
                ``recall_at``, ``iou_at`` for
                ``R@N,IoU=M``.
                Default: ``{'R@N,IoU=M': dict(recall_at=[1,5], 
                iou_at=[0.3,0.5,0.7])}``.
            logger (logging.Logger | None): Training logger. 
                Defaults: None.

        Returns:
            dict: Evaluation results for evaluation metrics.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['R@N,IoU=M', 'mAP', 'mcAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        top_n = metric_options.get('R@N,IoU=M', {}).get('recall_at', [100])
        top_n = max(top_n)

        eval_results = OrderedDict()
        results_list = self.results2json(results, top_n)

        if 'mAP' in metrics or 'mcAP' in metrics:
            gaussian_label = self.gaussian_label
            self.gaussian_label = False
            semantic_scores = [res['anno_score'][:, 2] for res in results_list]
            semantic_labels = []
            for anno_idx in range(len(results_list)):
                label = self.get_span_based_label(self.anno_infos[anno_idx])
                semantic_label = label[:,2]
                if self.chunk_interval != 1:
                    semantic_label = semantic_label[::self.chunk_interval]
                semantic_labels.append(semantic_label)
            mAP, _, mcAP, _ = frame_level_map(semantic_scores, semantic_labels)
            self.gaussian_label = gaussian_label

        for metric in metrics:
            if metric == 'R@N,IoU=M':
                recall_at = metric_options['R@N,IoU=M']['recall_at']
                iou_at = metric_options['R@N,IoU=M']['iou_at']
                proposals = [res['proposals'] for res in results_list]
                gt_moments = [np.array(info['timestamp']) for info in self.anno_infos]
                recall_x_iou = recall_at_iou_at(proposals,
                                                gt_moments,
                                                recall_at,
                                                iou_at)
                for i, r in enumerate(recall_at):
                    for j, iou in enumerate(iou_at):
                        eval_results['R@%d,IoU=%.1f' % (r, iou)] = recall_x_iou[i, j]
            if metric == 'mAP':
                eval_results['mAP'] = mAP
            if metric == 'mcAP':
                eval_results['mcAP'] = mcAP

        return eval_results


if __name__ == "__main__":
    # Test OnlineTSGVDataset
    ann_file = '/home/wangxiao13/otsgv/ilearnTemporalSentenceGrounding/data/tacos/test.json'
    pipeline = []
    short_memory_sample_length = 8
    long_memory_sample_length = 128
    future_memory_sample_length = 0
    data_prefix = '/home/wangxiao13/otsgv/ilearnTemporalSentenceGrounding/data/tacos/feature'
    video_feat_filename = 'tall_c3d_features.hdf5'
    # Load dataset
    dataset = OnlineTSGVDataset(ann_file, 
                                pipeline, 
                                short_memory_sample_length, 
                                data_prefix, 
                                video_feat_filename, 
                                sentence_feat_filename='bert_features.hdf5', 
                                bert_model='bert-base-uncased', 
                                long_memory_sample_length=long_memory_sample_length, 
                                future_memory_sample_length=future_memory_sample_length, 
                                in_memory=True, 
                                test_mode=True)
    # GT as results
    results = []
    for sample in tqdm(dataset.chunk_infos):
        start_label = sample['start_label']
        end_label = sample['end_label']
        semantic_label = sample['semantic_label']
        start_score = start_label[-1].astype(np.int32)
        end_score = end_label[-1].astype(np.int32)
        semantic_score = semantic_label[-1].astype(np.int32)
        if start_score == 1:
            assert semantic_score == 1
        if end_score == 1:
            assert semantic_score == 1
        score = np.array([start_score, end_score, semantic_score])
        results.append(score)
    results_list = dataset.results2json(results)
    # Evaluation
    eval_results = dataset.evaluate(results)