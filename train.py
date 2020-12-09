# python imports
import os
import glob
import warnings
# external imports
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
# model imports
from Model.config import args
from Model.resnet import get_pose_net
from Model.hourglass import get_hourglass
from Model.dataset import COCO
# utils imports
from utils.utils import _tranpose_and_gather_feature
from utils.losses import _neg_loss, _reg_loss, _reg_l1_loss





def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
def make_dirs():    # 对应文件夹不存在的话就创建文件夹
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def train():
    # 加载之前训练的模型(指定轮数)
    pth_epoch = 32
    # 创建需要的文件夹并指定gpu
    make_dirs() 
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    #poseNet = get_pose_net(num_layers=50,head_conv=64,num_joints=12)       # resnet+dconv 
    poseNet = get_hourglass('large_hourglass',num_joints=12)        # hourglass
    poseNet = poseNet.to(device)

    if(pth_epoch!=0):
        pre=torch.load(os.path.join('./Checkpoint',str(pth_epoch)+'.pth'))
        poseNet.load_state_dict(pre)

    poseNet.train()
    print("poseNet: ", count_parameters(poseNet))   # 模型参数个数
    opt = Adam(poseNet.parameters(), lr=args.lr)


    # 加载训练数据
    dataset = COCO("./data",split='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Training loop.
    for epoch in range(pth_epoch+1, args.n_iter + 1):
        for batch_idx, batch in enumerate(dataloader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=device, non_blocking=True)

            outputs = poseNet(batch['image'])
            hmap, w_h_, hps, regs, hm_hp, hp_offset = zip(*outputs)
           
            regs = [_tranpose_and_gather_feature(r, batch['ind']) for r in regs]
            w_h_ = [_tranpose_and_gather_feature(r, batch['ind']) for r in w_h_]

            # 计算loss
            hmap_loss = _neg_loss(hmap, batch['hmap'])
            reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
            w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
            hm_hp_loss = _neg_loss(hm_hp, batch['hm_hp'])
            #hp_loss = _reg_l1_loss(hps, batch['hps_mask'], batch['ind'], batch['hps']) # 关键点相对于中心点的offset
            loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss + 1* hm_hp_loss #+ 0* hp_loss
            print('[%d/%d-%d/%d] ' % (epoch, args.n_iter + 1, batch_idx, len(dataloader)) +
                    ' loss= %.5f hmap= %.5f reg= %.5f w_h= %.5f hm_hp= %.5f ' %
                    (loss.item(), hmap_loss.item(), reg_loss.item(), w_h_loss.item(), hm_hp_loss.item()) 
                )

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # 保存模型
        if(epoch% args.n_save_iter == 0):
            save_file_name = os.path.join(args.model_dir, '%d.pth' % epoch)
            torch.save(poseNet.state_dict(), save_file_name)
    f.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
