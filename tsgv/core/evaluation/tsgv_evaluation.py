import numpy as np
from sklearn.metrics import average_precision_score


def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap


def average_precision(preds, gts):
    """Calculate average precision.
    Args:
        preds (np.array): 
        gts (np.array): 
    """
    assert len(preds) == len(gts), "Prediction and ground truth need to be of the same length"
    if np.sum(gts == 1) > 0:
        AP = average_precision_score(gts, preds)
        cAP = calibrated_average_precision_score(gts, preds)
        if np.isnan(cAP):
            cAP = 1.0
        """In rare circumstances, k = 0, gts[idx] == 1, then
        wtp / (wtp + fp) -> 0 / (0 + 0) causing overflow problems."""
    else:
        AP = 1.0
        cAP = 1.0
    return float(AP), float(cAP)


def frame_level_map(prediction, ground_truth, num_workers=8):
    """Calculate mean average precision for temporal sentence grounding.
    Args:
        prediction (list[np.array]): 
        ground_truth (list[np.array]): 
    """
    all_sample_ap = list()
    all_sample_cap = list()

    for sample_pred, sample_gt in zip(prediction, ground_truth):
        this_sample_ap, this_sample_cap = average_precision(sample_pred, sample_gt)
        all_sample_ap.append(this_sample_ap)
        all_sample_cap.append(this_sample_cap)

    mAP = sum(all_sample_ap) / len(all_sample_ap)
    mcAP = sum(all_sample_cap) / len(all_sample_cap)
    return mAP, all_sample_ap, mcAP, all_sample_cap


def iou(candidates, gt):
    """Calculate the Intersection over Union score of 
    two timestamps.

    Args:
        candidates (np.array): [*, 2]
        gt (np.array): [2]
    Return:
        iou [*]
    """
    candidates = candidates.astype(np.float32)
    gt = gt.astype(np.float32)
    start, end = np.split(candidates, [1], -1)
    # [*, 1] [*, 1]
    s, e = gt[0], gt[1]
    # [1] [1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    inter = np.clip(inter, a_min=0, a_max=None)
    iou = inter / union
    # [*, 1]
    return iou.squeeze(-1)


def nms(moments, scores, thresh, topn=None):
    """Non-maximum supression on timestamp proposals.
    """
    # Sort the moments in an score-descending order
    ranks = (-scores).argsort()
    scores = scores[ranks]
    moments = moments[ranks]
    # nms
    suppressed = np.zeros_like(ranks).astype(np.bool)
    topn_chk = 0
    for i in range(suppressed.size - 1):
        if suppressed[i]:
            continue
        topn_chk += 1
        if topn and topn_chk >= topn:
            suppressed[i+1:][:] = True
            break
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed], scores[~suppressed]


def recall_at_iou_at(proposals, 
                     labels,
                     recall_at=[1,5],
                     iou_at=[0.3,0.5,0.7]):
    """Evalation metric ``R@N,IoU=M``
    for temporal sentence grounding in videos.
    Args:
        proposals (list[np.array]): predicted timestamps with 
            confidence scores.
        labels (list[np.array]): ground truth timestamps.
        recall_at (list[int]): values of N.
        iou_at (list[float]): values of M.
    """
    recall_at = np.array(recall_at)
    iou_at = np.array(iou_at)
    recall_at.sort()
    iou_at.sort()
    # recall_at, iou_at must be in a ascending order.

    recall_x_iou = np.zeros((recall_at.size, iou_at.size))
    for idx, (proposal, label) in enumerate(zip(proposals, labels)):
        # proposal: [num_proposals, 3]
        # label: [2]
        moments, scores = proposal[:,:2], proposal[:,2]
        for i, r in enumerate(recall_at):
            mious = iou(moments[:r], label)[:,None]
            # [min(r, num_proposals), 1]
            mious = np.pad(
                mious, 
                ((0,max(0, r-mious.shape[0])), (0,iou_at.size-1)), 
                mode='edge')
            # [r, iou_at.size]
            bools = mious > iou_at
            recall_x_iou[i] += bools.any(axis=0)
    recall_x_iou /= len(proposals)
    return recall_x_iou


if __name__ == "__main__":
    # Test iou()
    print(iou(np.array([[1,2],[1.5,2.5]]), np.array([1,2])))
    print(iou(np.array([[1,2]]), np.array([1,2])))
    # Test recall_at_iou_at()
    proposals = []
    labels = []
    for i in range(100):
        label = 1 + 10 * np.random.rand(2)
        # [2]
        if label[0] > label[1]:
            label[0], label[1] = label[1], label[0]
        proposal = np.concatenate(
            [0.1*np.random.randn(10, 2)+label[None, :],
            np.random.rand(10, 1)],
            axis=1)
        # [10, 3]
        proposals.append(proposal)
        labels.append(label)
    recall_x_iou = recall_at_iou_at(proposals, labels)
    print(recall_x_iou)