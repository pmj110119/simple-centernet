import os
import sys
import argparse
import numpy as np

import torch
import torch.utils.data
from torchvision import transforms as T
from torch.utils.data import DataLoader

from Model.CellDetect_dataset import COCO
from Model.resnet import get_pose_net
from Model.hourglass import get_hourglass
from utils.utils import load_model
from utils.image import transform_preds
from utils.summary import create_logger
from utils.post_process import _nms,_topk,multi_pose_decode
from Model.config import args
#from nms.nms import soft_nms
from torchvision import transforms
unloader = transforms.ToPILImage()
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def tensor2plt(tensor):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image)
    return image
def imshow(tensor,ind=1 , title=None):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image)
    #plt.figure(ind)
    if(ind==1):
        plt.subplot(131)
    elif(ind==2):
        plt.subplot(132)
    else:
        plt.subplot(133)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
def soft_nms(dicts):
    for label, boxes in dicts.items():
        boxesArray = np.asarray(boxes)

        x1 = boxesArray[:, 0]
        y1 = boxesArray[:, 1]
        x2 = boxesArray[:, 2]
        y2 = boxesArray[:, 3]
        scores = boxesArray[:,4]
        areas = (x2-x1+1)*(y2-y1+1)
        index = scores.argsort()[::-1]

        while index.size>0:
            box = boxes[index[0]]
            x1tmp = np.maximum(x1[index[0]], x1[index[1:]])
            y1tmp = np.maximum(y1[index[0]], y1[index[1:]])
            x2tmp = np.minimum(x2[index[0]], x2[index[1:]])
            y2tmp = np.minimum(y2[index[0]], y2[index[1:]])
            interArea = np.maximum(x2tmp - x1tmp,0.0)*np.maximum(y2tmp - y1tmp, 0.0)
            iou = interArea/(areas[index[0]] + areas[index[1:]] -interArea)
            for i in range(1,len(iou)+1):
                if iou[i-1]>0.2:
                    #boxes[index[i]][4] *= 1-iou[i-1]
                    ov = 1-iou[i-1]
                    sigma = 0.5
                    boxes[index[i]][4] *= np.exp(-(ov*ov)/sigma)
            index = np.delete(index,[0])

        dicts[label] = boxes
        return dicts

# cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
# cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
# cfg.pretrain_dir = os.path.join(cfg.ckpt_dir, 'checkpoint.t7')

# os.makedirs(cfg.log_dir, exist_ok=True)
# os.makedirs(cfg.ckpt_dir, exist_ok=True)

# cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]
def merge_outputs(detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        # if self.opt.nms or len(self.opt.test_scales) > 1:
        #     soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

def main():

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    max_per_image = 105


    # 加载之前训练的模型(指定轮数)
    pth_epoch = 350
    #poseNet = get_pose_net(num_layers=50,head_conv=32,num_classes=1,num_joints=12).to(device)
    poseNet = get_hourglass('large_hourglass',num_joints=12).to(device)

    #poseNet.load_state_dict(torch.load(os.path.join('./Checkpoint',str(pth_epoch)+'.pth')))
    pre = torch.load(os.path.join('./Checkpoint/36.pth'))
    poseNet.load_state_dict(pre)
    poseNet.eval()
    
    # dataloader
    dataset = COCO("./data",split='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    results ={}
    with torch.no_grad():   # 不进行反向传播
        for batch_idx, batches in enumerate(dataloader):
            
            detections = []
         
            # image = batch_img.to(device)
            # print(image.shape)
            outputs = poseNet(batches['image'].to(device))
            #hmap, regs, w_h_, hm_hp = zip(*output)
            hmap, w_h_, hps ,regs,  hm_hp,hp_offset = zip(*outputs)
            #dets = ctdet_decode(*output, K=100) # K = topk 前100个峰值
            
   
            hmap = torch.sigmoid(hmap[0])
            hm_hp = torch.sigmoid(hm_hp[0])

            #hmap_nms =  # K = topk 前100个峰值
            plt.subplot(231)
            plt.imshow(tensor2plt(hmap[0]))
            plt.subplot(232)
            plt.imshow(tensor2plt(_nms(hmap[0])))
            plt.subplot(233)
            plt.imshow(tensor2plt(batches['hmap'][0][0]))
            

        
       
            plt.subplot(234)
            plt.imshow(tensor2plt(hm_hp[0][0])) 
            plt.subplot(235)
            #hm_hp_plt = tensor2plt( _nms(hm_hp[0])[0])
            hm_hp_np = _nms(hm_hp[0])[0].cpu().numpy()
            row = np.argmax(hm_hp_np) // hm_hp_np.shape[1] 
            col = np.argmax(hm_hp_np) % hm_hp_np.shape[1]
            curve = np.zeros([128,128])
            curve[row][col] = 1.
            plt.imshow(curve)
            plt.subplot(236)
            plt.imshow(tensor2plt(batches['hm_hp'][0][0]))

            plt.pause(0.001)






            # #print(w_h_.shape)
            # #def multi_pose_decode(heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
            # detections = multi_pose_decode(hmap[0],w_h_[0],hps[0],reg=regs[0],hm_hp=hm_hp[0],K=1)
            # #print(detections[0][0].shape)
            # dets = detections.detach().cpu().numpy().reshape(1, -1, detections.shape[2])    # [1, 100, 24]
            # #print(dets.shape)
            # ret = []
            # c=np.array([640.,512.])
            # s=np.array([1280.])
            # # bbox = np.array(dets[0, :, :4].reshape(-1, 2))
            # # pts = np.array(dets[0, :, 5:29].reshape(-1, 2))
            # # print('bbox:',bbox)
            # # print('pts:',pts)
            # for i in range(dets.shape[0]):
            #     bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (512, 512))
            #     pts = transform_preds(dets[i, :, 5:29].reshape(-1, 2), c[i], s[i], (512, 512))  #  (5:29) =================
            #     top_preds = np.concatenate(
            #         [bbox.reshape(-1, 4), dets[i, :, 4:5],
            #         pts.reshape(-1, 24)], axis=1).astype(np.float32).tolist()      # 24 =================
            #     ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
            # dets = ret
            # # print(dets[0][1].size())
        
            # dets[0][1] = np.array(dets[0][1], dtype=np.float32).reshape(-1, 29)
            # dets[0][1][:, :4] /= 1
            # dets[0][1][:, 5:] /= 1
            # dets =  dets[0]
            # result = merge_outputs(dets)
            # print(dets[1].size())










            #detections = torch.cat([bboxes, scores, kps, clses], dim=2)
                # dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                # top_preds = {}
                # dets[:, :2] = transform_preds(dets[:, 0:2],
                #                             inputs[scale]['center'],
                #                             inputs[scale]['scale'],
                #                             (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                # dets[:, 2:4] = transform_preds(dets[:, 2:4],
                #                             inputs[scale]['center'],
                #                             inputs[scale]['scale'],
                #                             (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                # cls = dets[:, -1]
                # for j in range(dataset.num_classes):
                #     inds = (cls == j)
                #     top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                #     top_preds[j + 1][:, :4] /= scale

                # detections.append(top_preds)

            # bbox_and_scores = {}
            # for j in range(1, dataset.num_classes + 1):
            #     bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
            #     if len(dataset.test_scales) > 1:
            #         soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
            # scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, dataset.num_classes + 1)])

            # if len(scores) > max_per_image:
            #     kth = len(scores) - max_per_image
            #     thresh = np.partition(scores, kth)[kth]
            #     for j in range(1, dataset.num_classes + 1):
            #         keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
            #         bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            # results[img_id] = bbox_and_scores

    #eval_results = dataset.run_eval(results, cfg.ckpt_dir)
    #print(eval_results)


if __name__ == '__main__':
  main()
