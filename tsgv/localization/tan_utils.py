import numpy as np


def get_2d_mask(num_clips, group_size, num_group):
    """
        mask2d: true means valid proposal grid
    """
    pooling_counts = [2 * group_size] + [group_size] * (num_group - 1)
    mask2d = np.zeros((num_clips, num_clips), dtype=bool)
    maskij = []

    stride, offset = 1, -1
    for c in pooling_counts:
        for _ in range(c): 
            # fill a diagonal line 
            offset += stride
            i, j = range(0, num_clips - offset, stride), range(offset, num_clips, stride)
            mask2d[i, j] = True
            maskij.append((i, j))
        stride *= 2
    return mask2d, maskij


def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()
    grids = np.stack(grids, 1)
    # [num_proposals, 2]
    scores = score2d[grids[:,0], grids[:,1]]
    # [num_proposals]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    # [num_proposals, 2]
    return moments, scores