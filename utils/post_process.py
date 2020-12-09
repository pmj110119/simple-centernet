import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def multi_pose_decode(heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps

    heat = _nms(heat)   # 先用一个3x3的maxpooling处理下，找局部峰值

    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _tranpose_and_gather_feature(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)  
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None: # 精校准（小数回归）
        reg = _tranpose_and_gather_feature(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feature(wh, inds)
    wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(0, 2, 1, 3).contiguous() # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)

        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K

        if hp_offset is not None: 
            hp_offset = _tranpose_and_gather_feature(hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5
        
        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3) # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
              (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
              (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
    return detections



def ctdet_decode(hmap, regs, w_h_, K=100):
    batch, cat, height, width = hmap.shape
    hmap=torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]

    batch = 1

    # 找到hmap峰值     （用一个maxpool得到）
    hmap = _nms(hmap)  # perform nms on heatmaps

    return hmap
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


