import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius

COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class COCO(data.Dataset):
    def __init__(self, data_dir, split, split_ratio=1.0, img_size=512):
        super(COCO, self).__init__()
        self.num_classes = 1
        self.num_joints =12
        self.class_name = COCO_NAMES
        self.valid_ids = COCO_IDS
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.split = split
        self.data_dir = os.path.join(data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, 'images/%s' % split)
        if split == 'test':
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'test.json')
        else:
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'train.json')

        self.max_objs = 128
        self.padding = 127  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7

        print('==> initializing coco 2017 %s data.' % split)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()

        if 0 < split_ratio < 1:
            split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
            self.images = self.images[:split_size]

        self.num_samples = len(self.images)

        print('Loaded %d %s samples' % (self.num_samples, split))

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)
        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)
        pts = np.array([anno['keypoints'] for anno in annotations], dtype=np.float32).reshape(self.num_joints,3)
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy
        #print(img_path)
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0

        flipped = False
        if self.split == 'train':
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(256, width)
            h_border = get_border(256, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            center[0] = width - center[0] - 1

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        # -----------------------------------debug---------------------------------
        # for bbox, label in zip(bboxes, labels):
        #   if flipped:
        #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        #   bbox[:2] = affine_transform(bbox[:2], trans_img)
        #   bbox[2:] = affine_transform(bbox[2:], trans_img)
        #   bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.img_size['w'] - 1)
        #   bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.img_size['h'] - 1)
        #   cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        #   cv2.putText(img, self.class_name[label + 1], (int(bbox[0]), int(bbox[1])),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        # -----------------------------------debug---------------------------------

        img = img.astype(np.float32) / 255.

        if self.split == 'train':
            color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])
        # 中心点heatmap
        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        # 关键点heatmap
        hm_hp = np.zeros((self.num_joints, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)
        kps = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.float32)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((self.max_objs * self.num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * self.num_joints), dtype=np.int64)
        hp_mask = np.zeros((self.max_objs * self.num_joints), dtype=np.int64)


        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        ind = np.zeros((self.max_objs), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        # detections = []
        
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
    
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                ind[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1
                # groundtruth bounding box coordinate with class
                # detections.append([obj_c[0] - w / 2, obj_c[1] - h / 2,
                #                    obj_c[0] + w / 2, obj_c[1] + h / 2, 1, label])
                        # 处理关键点   索引-j

                for j in range(self.num_joints):
                    # 只有v>0，才处理这个关键点（否则全为初始值0）
                    if pts[j, 2] > 0:
                        pts[j, :2] = affine_transform(pts[j, :2], trans_fmap) # 仿射变换
                        if pts[j, 0] >= 0 and pts[j, 0] < self.fmap_size['w'] and \
                                pts[j, 1] >= 0 and pts[j, 1] < self.fmap_size['h']:  # 这里是魔改了下，原本是限制不超过最大bbox长宽，现在不需要bbox，所以设置成fmap长宽
                            kps[k, j * 2: j * 2 + 2] = pts[j, :2] - obj_c_int  # 相对于中心点的offset
                            kps_mask[k, j * 2: j * 2 + 2] = 1   
                            pt_int = pts[j, :2].astype(np.int32)  # 关键点坐标 int
                            #hp_offset[k * self.num_joints + j] = pts[j, :2] - pt_int # 代表小数精度损失的offset
                            #hp_ind[k * self.num_joints + j] = pt_int[1] * output_res + pt_int[0]
                            #hp_mask[k * self.num_joints + j] = 1
                             
                            draw_umich_gaussian(hm_hp[j], pt_int, radius)
        # detections = np.array(detections, dtype=np.float32) \
        #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'ind': ind, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id,'hm_hp':hm_hp, 'hps_mask':kps_mask, 
                'hps':kps}


    def __len__(self):
        return self.num_samples




# if __name__ == '__main__':
#   from tqdm import tqdm
#   import pickle

#   dataset = COCO('E:\\coco_debug', 'train')
#   for d in dataset:
#     b1 = d
#   #   pass


 