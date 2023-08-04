import os
import os.path as osp
import json
from tqdm import tqdm

import h5py

#### Set the directory of your downloaded MAD dataset ####
dataset_dir = '/home/wangxiao13/otsgv/ilearnTemporalSentenceGrounding/data'

# check
assert osp.exists(dataset_dir)
dataset_path = osp.join(dataset_dir, 'mad')
feat_dir = osp.join(dataset_path, 'feature')
if not osp.exists(dataset_path):
    os.mkdir(dataset_path)
if not osp.exists(feat_dir):
    os.mkdir(feat_dir)

# Load annotations
train = json.load(open('./annotations/MAD_train.json', 'r'))
val   = json.load(open('./annotations/MAD_val.json', 'r'))
test  = json.load(open('./annotations/MAD_test.json', 'r'))

# Features
LANG_FEAT   = h5py.File('./features/CLIP_language_tokens_features.h5', 'r') 
VISUAL_FEAT = h5py.File('./features/CLIP_frames_features_5fps.h5', 'r') 

for split in ['train', 'val', 'test']:
    raw_annos = json.load(open(f'./annotations/MAD_{split}.json','r'))
    anno_file = osp.join(dataset_path, f'{split}.json')
    visual_feat_file = osp.join(feat_dir, f'CLIP_frames_features_5fps_{split}.hdf5')
    lang_feat_file = osp.join(feat_dir, f'CLIP_language_tokens_features_{split}.hdf5')
    VideoFeatFile = h5py.File(visual_feat_file, 'a') 
    SentenceFeatFile = h5py.File(lang_feat_file, 'a') 
    
    processed_annos = dict()
    for anno_id, anno in tqdm(raw_annos.items()):
        movie_id = anno['movie']
        # Create dict for movie_id
        if movie_id not in processed_annos:
            movie_feat = VISUAL_FEAT[movie_id][:]
            duration_frame = movie_feat.shape[0]
            VideoFeatFile.create_dataset(movie_id, data=movie_feat)
            processed_annos[movie_id] = {
                'anno_ids': [],
                'sentences': [],
                'duration': anno['movie_duration'],
                'timestamps': [],
                'duration_frame': duration_frame
            }
        # Add anno in movie_id
        prosed_anno = processed_annos.get(movie_id)
        prosed_anno['anno_ids'].append(anno_id)
        prosed_anno['sentences'].append(anno['sentence'])
        prosed_anno['timestamps'].append(anno['ext_timestamps'])
        assert anno['movie_duration'] == prosed_anno['duration']
        sent_feat = LANG_FEAT[anno_id][:]
        SentenceFeatFile.create_dataset(anno_id, data=sent_feat)
        processed_annos[movie_id] = prosed_anno
    
    with open(anno_file, 'w') as F:
        json.dump(processed_annos, F)
    VideoFeatFile.close()
    SentenceFeatFile.close()

VISUAL_FEAT.close()
LANG_FEAT.close()