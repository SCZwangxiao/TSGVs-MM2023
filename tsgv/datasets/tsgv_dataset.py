# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
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

from .base import BaseDataset
from .builder import DATASETS
from ..localization import get_2d_mask, score2d_to_moments_scores
from ..core.evaluation import iou, nms
from ..core.evaluation import recall_at_iou_at


@DATASETS.register_module()
class TSGVDataset(BaseDataset):
    """Base dataset for temporal sentence grounding in videos.

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
        portion (float | int | list<int rank, int N>): If ``float``, it indicates the raito of the 
            annotations to be used. If ``int``, it indicates the first 
            ``portion`` annotations to be used. If ``list<rank, N>, in indicates
            the rank-th sample of a total of N.
            Defailt: 1.0
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
                 proposal_options=dict(type='span_based',
                                       upsample_short_video=True,
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
        super().__init__(ann_file, pipeline, data_prefix, test_mode)
        self.video_feat_file = osp.join(data_prefix, video_feat_filename)
        self.sentence_feat_file = osp.join(
            data_prefix, f'{language_model}_language_tokens_features_{split}.hdf5')
        self.split = split
        self.portion = portion
        self.language_model = language_model
        self.max_sentence_length = max_sentence_length
        self.num_segments = num_segments
        self.proposal_method, cache_anno_label = self.merge_proposal_options(proposal_options)
        self.nms_thresh = nms_thresh
        self.num_workers = num_workers
        self.video_in_memory = video_in_memory
        self.sent_in_memory = sent_in_memory
        self.cache_anno_label = cache_anno_label if not test_mode else False

        self.anno_infos, self.video_name2video_idx = self.parse_video_annotations(self.video_infos)
        self.check_sentence_features()
        self.anno_infos, self.video_names = self.sample_dataset(self.anno_infos, self.portion)
        self.calculate_dataset_statistics()
        self.video_name2video_feature, self.anno_id2sent_feature = self.prepare_features()

    def load_annotations(self):
        """Load the video-level annotation according to ann_file into video_infos."""
        video_infos = []
        if type(self.ann_file) == str:
            self.ann_file = [self.ann_file]
        for ann_file in self.ann_file:
            anno_database = mmcv.load(ann_file)
            for video_name, video_info in anno_database.items():
                video_info['video_name'] = video_name
                video_infos.append(video_info)
        return video_infos

    def merge_span_based_options(self, proposal_options):
        self.upsample_short_video = proposal_options.pop('upsample_short_video', True)
        self.gaussian_label = proposal_options.pop('gaussian_label', False)
        self.gaussian_label_width = proposal_options.pop('gaussian_label_width', [0.25,0.25,0.21])
        self.asym_gaussian_label = proposal_options.pop('asym_gaussian_label', False)
        self.extension_ratio = proposal_options.pop('extension_ratio', 1.0)

    def merge_2dtan_options(self, proposal_options):
        self.upsample_short_video = False
        self.group_size = proposal_options.pop('group_size', 8)
        self.num_clips = proposal_options.pop('num_clips')
        # Check whether group_size, num_clips is a power of 2
        assert (self.group_size & (self.group_size-1) == 0) and self.group_size != 0
        assert (self.num_clips & (self.num_clips-1) == 0) and self.num_clips != 0
        # self.num_clips = 2 * self.group_size * 2^(num_group-1)
        assert self.num_clips % self.group_size == 0
        self.num_group = int(log(self.num_clips / (2 * self.group_size), 2)) + 1
        self.mask2d, _ = get_2d_mask(self.num_clips, self.group_size, self.num_group)

    def merge_proposal_options(self, proposal_options):
        proposal_method = proposal_options.pop('type')
        # Some dataset, like MAD, has extreme video length
        # cache anno_label is untolarable
        cache_anno_label = proposal_options.pop('cache_anno_label', True)
        if proposal_method == 'span_based':
            self.merge_span_based_options(proposal_options)
        elif proposal_method == '2dtan':
            self.merge_2dtan_options(proposal_options)
        elif proposal_method == 'smin':
            self.merge_span_based_options(proposal_options)
            self.merge_2dtan_options(proposal_options)
        else:
            raise NotImplementedError
        return proposal_method, cache_anno_label

    @staticmethod
    def parse_video_annotations(video_infos):
        """Seperate the video-level annotation into the moment-level, getting anno_infos."""
        anno_infos = []
        video_name2video_idx = {}
        for video_info in video_infos:
            video_name = video_info['video_name']
            duration = video_info['duration']
            duration_frame = video_info['duration_frame']
            if duration_frame <= 70:
                # Filter too short annotations
                continue
            assert video_name not in video_name2video_idx
            video_name2video_idx[video_name] = len(video_name2video_idx)
            for timestamp, sentence, anno_id in zip(video_info['timestamps'], video_info['sentences'], video_info['anno_ids']):
                timestamp = [max(0,timestamp[0]), min(duration,timestamp[1])] # some anno may be inaccurate.
                if timestamp[0] >= timestamp[1]:
                    # Filter invalid annotations
                    continue
                anno_info = {
                    'anno_id': anno_id,
                    'video_name': video_name,
                    'duration': duration,
                    'duration_frame': duration_frame,
                    'timestamp': timestamp,
                    'sentence': sentence}
                anno_infos.append(anno_info)
        return anno_infos, video_name2video_idx

    def calculate_dataset_statistics(self):
        max_duration_frame = [video_info['duration_frame'] for video_info in self.video_infos]
        max_duration_frame = max(max_duration_frame)
        self._max_duration_frame = max_duration_frame

    def prepare_features(self):
        """Prepare video and sentence features in two steps:
        1. Check sentence features.
        2. Padding the sentence feature, and calculating its length.
        3. If in_memory=True, load only video features into memory. 
        4. Video features are loaded into video_name2video_feature, 
            while sentence features are loaded into anno_id2sent_feature.
        """
        # Use pandas.Dataframe to replace python dict to avoid copy-on-write
        # in dataloader workers caused by Python multiprocessing.
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        rank = distributed.get_rank()
        video_name2video_feature = {}
        anno_id2sent_feature = {}
        # Load video features
        def load_video_features(video_name, video_name2video_feature, VideoFeatFile):
            if video_name in self.video_names:
                vid_feat = VideoFeatFile[video_name][:].squeeze().astype(np.float32)
                if self.num_segments is not None:
                    vid_feat = self.avgfeats(vid_feat, self.num_segments)
                video_name2video_feature[video_name] = vid_feat
        if self.video_in_memory:
            with h5py.File(self.video_feat_file, 'r') as VideoFeatFile:
                if rank == 0:
                    for video_name in tqdm(VideoFeatFile, desc='Loading video features...', mininterval=10.):
                        load_video_features(video_name, video_name2video_feature, VideoFeatFile)
                else:
                    for video_name in VideoFeatFile:
                        load_video_features(video_name, video_name2video_feature, VideoFeatFile)
                video_name2video_feature['hack_key'] = np.array([6,6])
                # Hack to enable pd to store np.arrays
                # pd.from_dict will raise an error when all np.arrays are of the same length
                # See https://github.com/pandas-dev/pandas/issues/26858
                video_name2video_feature = pd.DataFrame.from_dict(video_name2video_feature, orient='index', dtype=object)
        # Load sentence features
        def load_sentence_features(anno_idx, anno_id2sent_feature, SentenceFeatFile):
            anno_id = self.anno_infos[anno_idx]['anno_id']
            feature = SentenceFeatFile[anno_id][:].astype(np.float32)
            feature, sentence_length = self.__padding_sentence_feature__(feature)
            anno_id2sent_feature[anno_id] = (feature, sentence_length)
        if self.sent_in_memory:
            with h5py.File(self.sentence_feat_file, 'r') as SentenceFeatFile:
                if rank == 0:
                    for anno_idx in tqdm(range(len(self.anno_infos)), desc='Loading sentence features...', mininterval=10.):
                        load_sentence_features(anno_idx, anno_id2sent_feature, SentenceFeatFile)
                else:
                    for anno_idx in range(len(self.anno_infos)):
                        load_sentence_features(anno_idx, anno_id2sent_feature, SentenceFeatFile)
                anno_id2sent_feature = pd.DataFrame.from_dict(anno_id2sent_feature, orient='index', dtype=object)
        return video_name2video_feature, anno_id2sent_feature

    @staticmethod
    def sample_dataset(anno_infos, portion):
        import random
        random.seed(1949)
        if type(portion) != list:
            if type(portion) == int:
                assert portion > 1
                num_samples = portion
            else:
                assert portion > 0.0
                assert portion <= 1.0
                num_samples = int(portion*len(anno_infos))
            indices = list(range(0, len(anno_infos)))
            if num_samples != len(anno_infos):
                indices = random.sample(indices, num_samples)
        else:
            assert len(portion) == 2
            rank, N = portion
            assert type(rank) == int and type(N) == int
            assert rank >= 0 and rank < N
            indices = []
            for idx in range(0, len(anno_infos)):
                if idx % N == rank:
                    indices.append(idx)
        anno_infos = np.array([anno_infos[idx] for idx in indices])
        video_names = set([anno_info['video_name'] for anno_info in anno_infos])
        return anno_infos, video_names

    def __padding_sentence_feature__(self, feature):
        # [L, h]
        sentence_length = feature.shape[0]
        if sentence_length > self.max_sentence_length:
            feature = feature[:self.max_sentence_length,:]
            sentence_length = self.max_sentence_length
        elif sentence_length < self.max_sentence_length:
            padding = self.max_sentence_length - sentence_length
            feature = np.pad(feature, ((0,padding), (0,0)), 'constant', constant_values=0)
        return feature, sentence_length

    def check_sentence_features(self):
        """Check whether the sentence features are already generated."""
        if distributed.get_rank() == 0:
            features_ok = True
            if osp.exists(self.sentence_feat_file):
                # Check whether all sentences' features in the file.
                with h5py.File(self.sentence_feat_file, 'r+') as File:
                    if 'Integrity' not in File:
                        for anno_info in self.anno_infos:
                            anno_id = anno_info['anno_id']
                            if anno_id not in File:
                                print('Warning! Missing sentence feature.')
                                print(f'Anno id {anno_id} is missing in file {self.sentence_feat_file}.')
                                File.close()
                                features_ok = False
                                break
                        if features_ok:
                            File.create_dataset('Integrity', data=(0,)) # Create integrity FLAG 
            else:
                print('No sentence feature detected, generating sentence feature:')
                features_ok = False
            features_ok = [features_ok]
        else:
            features_ok = [None]
        distributed.broadcast_object_list(features_ok, src=0)
        features_ok = features_ok[0]
        if not features_ok:
            self.__generate_sentence_features__()

    def __generate_sentence_features__(self):
        """Generate sentence features.
        Here we use second-to-last hidden layer of BERT model
        See 3.5 "Pooling Strategy & Layer Choice" in article 
        https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings       
        """
        if distributed.get_rank() == 0: # h5py dose not support multiple I/O
            from transformers import BertTokenizer, BertForPreTraining
            tokenizer = BertTokenizer.from_pretrained(self.language_model)
            bert = BertForPreTraining.from_pretrained(self.language_model, return_dict=True)
            bert.to('cuda')
            with h5py.File(self.sentence_feat_file, 'a') as File:
                for anno_idx in tqdm(range(len(self.anno_infos)), desc='Generating sentence features...', mininterval=10.):
                    anno_info = self.anno_infos[anno_idx]
                    anno_id = self.anno_infos[anno_idx]['anno_id']
                    sentence = anno_info['sentence']
                    if anno_id not in File:
                        sentence_tokenized = tokenizer(sentence, return_tensors="pt")
                        # token_num = sentence_num + 2
                        # [CLS] sentence [SEP]
                        with torch.no_grad():
                            for key in sentence_tokenized:
                                sentence_tokenized[key] = sentence_tokenized[key].to('cuda')
                            sentence_emb = bert(**sentence_tokenized, output_hidden_states=True)['hidden_states'][-2]
                            sentence_emb = sentence_emb.squeeze_().to('cpu').numpy()
                            # [token_num, 768]
                        File.create_dataset(anno_id, data=sentence_emb)
        torch.distributed.barrier()

    def __len__(self):
        """Get the size of the dataset."""
        return self.anno_infos.shape[0]

    def __gaussian_label__(self, t, mu, sigma, label_type):
        if not self.asym_gaussian_label or label_type == 'semantic':
            t = - np.power( (t - mu) / sigma, 2) / 2
        else:
            # ratio_ori = 0.2
            # ratio_new = 0.6
            ratio_ori = 0.1
            ratio_new = 0.3
            if label_type == 'start':
                mask = (t < mu)
                ratio_ori *= -1
                ratio_new *= -1
            else:
                mask = (t > mu)
            s_ori = ratio_ori * (mu - t) + sigma
            s_new = ratio_new * (t - mu) + sigma
            s = np.ma.array(s_ori, mask=mask)
            s = s.filled(s_new)
            t = - np.power((t - mu) / s, 2) / 2
        return np.exp(t)

    def get_span_based_label(self, anno_info, fixed_num_frames=None):
        """
        Args:
            anno_info (dict): information of annotation (element of self.anno_infos)
            fixed_num_frames (int|None): 
        Return:
            label (np.array): [fixed_num_frames, 3] if fixed_num_frames is not ``None``
                else [duration_frame of annotation, 3]
        """
        if not self.cache_anno_label or 'label' not in anno_info:
            # Calculate and add it to cache if anno_label is not available
            duration = anno_info['duration']
            duration_frame = anno_info['duration_frame']
            # infer the length of labels
            if fixed_num_frames is None:
                label_length = duration_frame
                self.upsample_short_video = False
            else:
                if duration_frame >= fixed_num_frames:
                    # downsample long videos
                    label_length = fixed_num_frames
                else:
                    # For short videos:
                    if self.upsample_short_video:
                        label_length = fixed_num_frames
                    else:
                        label_length = duration_frame
            # Calculate start and end point
            anno_timestamp = anno_info['timestamp']
            anno_framestamp = np.array(anno_timestamp) / duration * (label_length - 1)
            s, e = anno_framestamp[0], anno_framestamp[1]
            mid = anno_framestamp.mean()
            span_length = e - s
            # We denote anno_framestamp as the frame sequence index, which
            # can only be in [0, label_length-1]
            if self.gaussian_label:
                t = np.arange(0, label_length)
                sigma_s = self.gaussian_label_width[0] * span_length
                sigma_e = self.gaussian_label_width[1] * span_length
                sigma_se = self.gaussian_label_width[2] * span_length * self.extension_ratio
                start_label = self.__gaussian_label__(t, s, sigma_s, 'start')
                end_label = self.__gaussian_label__(t, e, sigma_e, 'end')
                semantic_label = self.__gaussian_label__(t, (s+e)/2, sigma_se, 'semantic')
                # To avoid span_length==0. anno_framestamp integrization should be put hehind.
                # anno_framestamp = np.round(anno_framestamp).astype(np.int32)
            else:
                s_ex = max(int(s - self.extension_ratio*span_length), 0)
                e_ex = min(int(e + self.extension_ratio*span_length), label_length-1)
                start_label = np.zeros(label_length)
                end_label = np.zeros(label_length)
                semantic_label = np.zeros(label_length)
                start_label[int(s)] = 1
                end_label[int(e)] = 1
                semantic_label[s_ex:e_ex+1] = 1
            label = np.stack([start_label, end_label, semantic_label], 1)
            if fixed_num_frames and label_length < fixed_num_frames:
                # Padding the short videos
                padding = fixed_num_frames - label_length
                label = np.pad(label, ((0,padding), (0,0)), 'constant')
            if self.cache_anno_label:
                anno_info['label'] = label
        elif self.cache_anno_label:
            label = anno_info['label']
        return label

    def get_2dtan_label(self, anno_info):
        """
        Args:
            anno_info (dict): information of annotation (element of self.anno_infos)
        return:
            label (np.array): [num_clips, num_clips]
        """
        if not self.cache_anno_label or 'label' not in anno_info:
            duration = anno_info['duration']
            duration_frame = anno_info['duration_frame']
            anno_timestamp = anno_info['timestamp']
            anno_framestamp = np.array(anno_timestamp) / duration * self.num_clips
            s = np.repeat(np.arange(0, self.num_clips)[:,None], self.num_clips, 1)
            e = np.repeat(np.arange(0, self.num_clips)[None,:], self.num_clips, 0)
            moment_2d = np.stack([s, e], 2)
            label = iou(moment_2d, anno_framestamp) * self.mask2d
            if self.cache_anno_label:
                anno_info['label'] = label
        elif self.cache_anno_label:
            label = anno_info['label']
        return label

    def get_anno_label(self, anno_idx=None, anno_info=None):
        """Get the video-sentence annotation labels."""
        assert anno_idx is None or anno_info is None # polymorphism
        if anno_info is None:
            anno_info = self.anno_infos[anno_idx]
        if self.proposal_method == 'span_based':
            return self.get_span_based_label(anno_info, self.num_segments)
        elif self.proposal_method == '2dtan':
            return self.get_2dtan_label(anno_info)
        elif self.proposal_method == 'smin':
            tan_label = self.get_2dtan_label(anno_info)
            # [num_clips, num_clips]
            span_label = self.get_span_based_label(anno_info, self.num_clips)
            # [num_clips, 3]
            smin_label = np.concatenate([tan_label, span_label], axis=1)
            # [num_clips, num_clips + 3]
            return smin_label

    @staticmethod
    def avgfeats(feats, num_segments):
        # Produce the feature of per video into fixed shape (e.g. 256*4096)
        # Input Example: feats (torch.tensor, ?x4096); num_segments (256)
        num_src_clips = feats.shape[0]
        idxs = torch.arange(0, num_segments+1, 1.0) / num_segments * num_src_clips
        idxs = idxs.round().long().clamp(max=num_src_clips-1)
        # To prevent a empty selection, check the idxs
        meanfeats = []
        for i in range(num_segments):
            s, e = idxs[i], idxs[i+1]
            if s < e:
                meanfeats.append(feats[s:e].mean(axis=0))
            else:
                meanfeats.append(feats[s])
        return np.stack(meanfeats)

    def get_video_features(self, video_name):
        """Get features of the whole video according to key ``video_name``."""
        # Load features
        if self.video_in_memory:
            vid_feat = self.video_name2video_feature.loc[video_name][0]
        else:
            with h5py.File(self.video_feat_file, 'r') as VideoFeatFile:
                vid_feat = VideoFeatFile[video_name][:].squeeze().astype(np.float32)
        # Resample
        if self.num_segments is not None:
            num_src_clips = vid_feat.shape[0]
            if self.upsample_short_video or num_src_clips >= self.num_segments:
                vid_feat = self.avgfeats(vid_feat, self.num_segments)
                vid_length = self.num_segments
            else:
                padding = self.num_segments - num_src_clips
                vid_feat = np.pad(vid_feat, ((0,padding),(0,0)), 'constant')
                vid_length = num_src_clips
        else:
            vid_length = vid_feat.shape[0]
        return vid_feat, vid_length

    def get_sent_features(self, anno_id):
        """Get features of the whole video according to key ``anno_id``."""
        if self.sent_in_memory:
            sent_feat, sentence_length = self.anno_id2sent_feature.loc[anno_id]
            sentence_length = int(sentence_length)
        else:
            with h5py.File(self.sentence_feat_file, 'r') as SentFeatFile:
                sent_feat = SentFeatFile[anno_id][:].astype(np.float32)
            sent_feat, sentence_length = self.__padding_sentence_feature__(sent_feat)
        return sent_feat, sentence_length

    @staticmethod
    def length2mask(length, max_length):
        padding = max_length - length
        mask = np.zeros(length)
        mask = np.pad(mask, (0, padding), 'constant', constant_values=1)
        return mask.astype(bool)

    def prepare_test_data(self, idx):
        """Prepare the frames for testing given the index."""
        return self.prepare_frames(idx)

    def prepare_train_data(self, idx):
        """Prepare the frames for training given the index."""
        return self.prepare_frames(idx)

    def prepare_frames(self, idx):
        """Prepare the frames for training or testing given the index."""
        results = copy.deepcopy(self.anno_infos[idx])
        vid_feat, video_length = self.get_video_features(results['video_name'])
        sent_feat, sentence_length = self.get_sent_features(results['anno_id'])
        label = self.get_anno_label(idx)
        results['video_features'] = vid_feat
        results['sentence_features'] = sent_feat
        results['video_length'] = video_length
        results['video_masks'] = self.length2mask(video_length, 
                                                  self.num_segments)
        results['sentence_length'] = sentence_length
        results['sentence_masks'] = self.length2mask(sentence_length, 
                                                     self.max_sentence_length)
        results['label'] = label
        return self.pipeline(results)

    def read_results(self, results):
        results_list = []
        for idx, res in enumerate(results):
            result=dict(anno_idx=idx,
                        anno_score=res)
            results_list.append(result)
        return results_list

    def score2proposals(self, anno_score, duration, duration_frame):
        """Convert prediction score to moment proposals.

        Args:
            anno_score (np.array): Shape: [T, 3]. Each column corresponds to:
                start probability, end probability, and within probability.
            duration (float|int): Scaling factor of ``T``.
        Reture:
            proposals (np.array): Shape: [num_proposals, 3]. Each column corresponds to:
                start, end, confidence scores.
        """
        if self.proposal_method == 'span_based':
            T = anno_score.shape[0]
            if not self.upsample_short_video and duration_frame < T:
                label_length = duration_frame
            else:
                label_length = T
            start_score = anno_score[:,0]
            end_score = anno_score[:,1]
            s = np.argmax(start_score)
            e = np.argmax(end_score)
            score = start_score[s] * end_score[e]
            s = s / (label_length - 1) * duration
            e = e / (label_length - 1) * duration
            proposals = np.array([[s, e, score]])
        elif self.proposal_method in ['2dtan', 'smin']:
            moments, scores = score2d_to_moments_scores(
                anno_score, self.num_clips, duration)
            proposals = np.concatenate([moments, scores[:,None]], 1)
        # [num_proposals, 3]
        return proposals

    def score2proposals_multiprocess(self, rank, task_queue, result_queue):
        while True:
            idx, res, topn, nms_thresh = task_queue.get()
            if idx == -1:
                # stop signal
                break
            anno_score = res['anno_score']
            anno_idx = res['anno_idx']
            duration = self.anno_infos[anno_idx]['duration']
            duration_frame = self.anno_infos[anno_idx]['duration_frame']
            proposals = self.score2proposals(anno_score, duration, duration_frame)
            moments, scores = proposals[:,:2], proposals[:,2]
            moments, scores = nms(moments, scores, thresh=nms_thresh, topn=topn)
            proposals = np.concatenate([moments, scores[:, None]], axis=1)
            result_queue.put((idx, proposals))

    def get_proposals(self, results_list, topn=None):
        """Get proposals following the procedure:

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
            task_queue.put((idx, res, topn, self.nms_thresh))
        for i in range(self.num_workers):
            # stop signal
            task_queue.put((-1, None, topn, self.nms_thresh))
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

    def results2json(self, results, topn=None):
        """Read model results to get the final prediction results in json.
        
        Args:
            results (list[np.array]): The ``np.array`` has three elements: 
            start probability, end probability, and within probability.
        Return:
            results_list (list[dict]): The ``dict`` is the annotation info
            with the prediction probabilities and proposals.
        """
        results_list = self.read_results(results)
        results_list = self.get_proposals(results_list, topn)
        return results_list

    def dump_results(self, 
                     results, 
                     out, 
                     output_format='json', 
                     version='VERSION 1.3',
                     topk=100):
        """Dump data to json/csv files."""
        result_dict = self.results2json(results, topk)
        external_data = dict(anno_infos=self.anno_infos.tolist())
        if output_format == 'json':
            output_dict = {
                'version': version,
                'results': result_dict,
                'external_data': external_data
            }
            mmcv.dump(output_dict, out)
        else:
            raise ValueError(
                f'The output format {output_format} is not supported.')

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
        allowed_metrics = ['R@N,IoU=M']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        top_n = metric_options.get('R@N,IoU=M', {}).get('recall_at', [100])
        top_n = max(top_n)

        eval_results = OrderedDict()
        results_list = self.results2json(results, top_n)

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

        return eval_results