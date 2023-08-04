# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import random
import warnings
from collections import OrderedDict
from tqdm import tqdm
from math import log
import multiprocessing as mp

import mmcv
import h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import torch
import numpy as np
import pandas as pd
from torch import distributed

from .tsgv_dataset import TSGVDataset
from .builder import DATASETS
from ..localization import get_2d_mask, score2d_to_moments_scores
from ..core.evaluation import iou, nms
from ..core.evaluation import recall_at_iou_at


@DATASETS.register_module()
class MADTSGVDataset(TSGVDataset):
    """dataset in MAD's paper for temporal sentence grounding in videos.

    The dataset loads raw features and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a json file with multiple objects, and each object has a
    key of the name of a video, and value of total seconds of the video, 
    annotated sentence of a video, and annotated timestamps of a video.
    Example of a annotation file:

    .. code-block:: JSON

        {
            "v_QOlSCBRmfWY":  {
                "duration": 82.73,
                "duration_frame": 515,
                "timestamps": [[0.83, 19.86], [17.37, 60.81], [56.26, 79.42]],
                "sentences": ['A young woman is seen standing in room and leads into her dancing',
                    'The girl dances around room while camera captures her movements',
                    'She continues dancing around room and ends by laying on floor']
            },
            ...
        }


    Args:
        ann_file (str ｜ list[str]): Path | list of paths to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where features are held.
        video_feat_filename (str): hdf5 filename of video features.
        language_model (str): BERT model name in huggingface transformers.
            Default: 'bert-base-uncased'.
        max_sentence_length (int): Maximum number of words in a sentence.
            Default: 20.
        gaussian_label (bool): Whether to set the label mask be gaussian distribution.
            Default: Flase.
        gaussian_label_width (list[int]): The ``alpha_0`` in gaussian distribution.
            Default: [0.25, 0.25, 0.21].
        nms_thresh (float): Threshold for non maximal supression.
            Default: 0.5.
        num_workers (int): Number of the workers for post processing.
            Default: 8.
        in_memory (bool): Store True when loading video features into RAM.
            Default: True.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, 
                 ann_file, 
                 pipeline, 
                 data_prefix, 
                 split,
                 video_feat_filename, 
                 language_model='bert-base-uncased', 
                 max_sentence_length=20,
                 num_segments=256,
                 window_sample_options=dict(neg_sample_ratio=0.7,
                                            sample_stride=1,
                                            test_interval=64),
                 proposal_options=dict(type='span_based',
                                       gaussian_label=False,
                                       gaussian_label_width=[0.25,0.25,0.21],
                                       asym_gaussian_label=False,
                                       cache_anno_label=True),
                 nms_thresh=0.5,
                 num_workers=8,
                 video_in_memory=True,
                 sent_in_memory=True, 
                 test_mode=False,
                 portion=1.0):
        super().__init__(ann_file=ann_file, 
                         pipeline=pipeline, 
                         data_prefix=data_prefix, 
                         split=split, 
                         video_feat_filename=video_feat_filename, 
                         language_model=language_model, 
                         max_sentence_length=max_sentence_length,
                         num_segments=None,
                         proposal_options=proposal_options,
                         nms_thresh=nms_thresh,
                         num_workers=num_workers,
                         video_in_memory=video_in_memory, 
                         sent_in_memory=sent_in_memory,
                         test_mode=test_mode,
                         portion=portion)
        self.window_length = num_segments
        self.neg_sample_ratio = window_sample_options.get('neg_sample_ratio', 0.7)
        self.sample_stride = window_sample_options.get('sample_stride', 1)
        self.test_interval = window_sample_options.get('test_interval', 64)
        assert self.neg_sample_ratio > .0 and self.neg_sample_ratio < 1.
        self.cache_anno_label = False

        if test_mode:
            self.window_infos = self.chunk_video_windows(self.anno_infos)

    def chunk_video_windows(self, anno_infos):
        window_infos = []
        if distributed.get_rank() == 0:
            pbar = tqdm(total=self.anno_infos.shape[0], desc='Chunking videos...', mininterval=10.)
        for anno_idx, anno_info in enumerate(self.anno_infos):
            duration_frame = anno_info['duration_frame']
            # Get sampling offset
            window_length = self.window_length * self.sample_stride
            interval = self.test_interval * self.sample_stride
            # Start sampling
            for start_window in range(0, duration_frame - window_length, interval):
                window_info = [anno_idx, start_window]
                # For saving RAM, we only store 'start_window'.
                # Use list instead of dict to avoid python COW.
                window_infos.append(window_info)
            if distributed.get_rank() == 0:
                pbar.update(1)
        if distributed.get_rank() == 0:
            pbar.close()
        return np.array(window_infos)

    def __len__(self):
        if self.test_mode:
            return self.window_infos.shape[0]
        else:
            return self.anno_infos.shape[0]

    def prepare_test_data(self, idx):
        """Prepare the frames for testing given the index."""
        # Get window boundary
        anno_idx, start_window = self.window_infos[idx]
        stop_window  = start_window + self.window_length * self.sample_stride
        # Get window-anno features
        results = copy.deepcopy(self.anno_infos[anno_idx])
        vid_feat, _ = self.get_video_features(results['video_name'])
        vid_feat = vid_feat[start_window:stop_window]
        if self.sample_stride > 1:
            vid_feat = self.avgfeats(vid_feat, self.window_length)
        sent_feat, sentence_length = self.get_sent_features(results['anno_id'])

        results['video_features'] = vid_feat
        results['video_masks'] = np.zeros(self.window_length).astype(bool)
        results['sentence_features'] = sent_feat
        results['sentence_length'] = sentence_length
        results['sentence_masks'] = np.pad(
            np.zeros(sentence_length),
            (0, sent_feat.shape[0] - sentence_length),
            'constant',
            constant_values = 1
        ).astype(bool)
        return self.pipeline(results)

    def prepare_train_data(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.anno_infos[idx])
        duration_frame = results['duration_frame']
        duration = results['duration']
        timestamp = results['timestamp']
        vid_feat, _ = self.get_video_features(results['video_name'])
        sent_feat, sentence_length = self.get_sent_features(results['anno_id'])

        if random.random() > self.neg_sample_ratio:
            # Generate positive targets for training
            start_idx = int(timestamp[0] / duration * duration_frame)
            stop_idx = int(timestamp[1] / duration * duration_frame)
            num_frames = stop_idx - start_idx
            # Sample window
            offset = random.randint(
                0, 
                abs(self.window_length * self.sample_stride - num_frames))
            if self.window_length * self.sample_stride > num_frames:
                start_window = max(start_idx - offset, 0)
            else:
                start_window = start_idx + offset
            stop_window  = start_window + self.window_length * self.sample_stride
            if stop_window > duration_frame:
                stop_window = duration_frame
                start_window = stop_window - self.window_length * self.sample_stride
            # Compute moment position withint the window
            start_moment = max((start_idx - start_window)//self.sample_stride, 0)
            stop_moment = min((stop_idx - start_window)//self.sample_stride,  self.window_length)
            anno_info = dict(duration=self.window_length,
                             duration_frame=self.window_length,
                             timestamp=[start_moment, stop_moment])
            label = self.get_anno_label(anno_info=anno_info) # hack
        else:
            # Generate negative targets for training
            # Sample window
            start_window = random.randint(0, duration_frame-self.window_length * self.sample_stride)
            stop_window = start_window + self.window_length* self.sample_stride
            if stop_window > duration_frame:
                stop_window = duration_frame
                start_window = stop_window - self.window_length * self.sample_stride
            anno_info = dict(duration=self.window_length,
                             duration_frame=self.window_length,
                             timestamp=[0, 1])
            label = self.get_anno_label(anno_info=anno_info) # hack
            label = np.zeros_like(label)
        
        vid_feat = vid_feat[start_window:stop_window]
        if self.sample_stride > 1:
            vid_feat = self.avgfeats(vid_feat, self.window_length)

        results['video_features'] = vid_feat
        results['video_masks'] = np.zeros(self.window_length).astype(bool)
        results['sentence_features'] = sent_feat
        results['sentence_length'] = sentence_length
        results['sentence_masks'] = np.pad(
            np.zeros(sentence_length),
            (0, sent_feat.shape[0] - sentence_length),
            'constant',
            constant_values = 1
        ).astype(bool)
        results['label'] = label
        return self.pipeline(results)

    def read_results(self, results):
        """Read window predictions to get video-annotation level results.
        
        Args:
            results (list[np.array]): The ``np.array`` has three elements: 
            start probability, end probability, and within probability.
        Return:
            results_list (list[dict]): The ``dict`` is the annotation info
            with the prediction probabilities and proposals.
        """
        results_list = []
        prev_anno_idx = None
        start_idxs = []
        anno_scores = []
        for score, info in zip(results, self.window_infos):
            anno_idx, start_idx = info
            if prev_anno_idx is None: # Cold start
                prev_anno_idx = anno_idx
            if prev_anno_idx == anno_idx: # same annotation (video-query pair)
                start_idxs.append(start_idx)
                anno_scores.append(score)
            else:
                anno_info = dict(anno_idx=prev_anno_idx,
                                 start_idxs=start_idxs,
                                 anno_scores=anno_scores)
                results_list.append(anno_info)
                prev_anno_idx = anno_idx
                start_idxs = [start_idx]
                anno_scores = [score]
        # Process the last window of the last annotation
        anno_info = dict(anno_idx=prev_anno_idx,
                         start_idxs=start_idxs,
                         anno_scores=anno_scores)
        results_list.append(anno_info)
        return results_list

    def score2proposals_multiprocess(self, rank, task_queue, result_queue):
        while True:
            idx, res, topn, nms_thresh = task_queue.get()
            if idx == -1:
                # stop signal
                break
            anno_idx = res['anno_idx']
            start_idxs = res['start_idxs']
            anno_scores = res['anno_scores']
            duration = self.anno_infos[anno_idx]['duration']
            duration_frame = self.anno_infos[anno_idx]['duration_frame']
            # Get proposals in all windows
            moments_all = []
            scores_all = []
            for start_idx, anno_score in zip(start_idxs, anno_scores):
                proposals = self.score2proposals(
                    anno_score, self.window_length * self.sample_stride, duration_frame)
                moments, scores = proposals[:,:2], proposals[:,2]
                moments, scores = nms(moments, scores, thresh=nms_thresh, topn=topn)
                moments = (start_idx + moments) / duration_frame * duration
                moments_all.append(moments)
                scores_all.append(scores)
            # Rerank all proposals
            moments_all = np.concatenate(moments_all, axis=0)
            scores_all = np.concatenate(scores_all, axis=0)
            moments, scores = nms(moments_all, scores_all, thresh=nms_thresh, topn=topn)
            proposals = np.concatenate([moments, scores[:, None]], axis=1)
            result_queue.put((idx, proposals))