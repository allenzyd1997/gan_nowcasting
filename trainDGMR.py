
# coding=utf-8
import argparse
import torch.distributed as dist
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import numpy as np
import os
import cv2
import random
from generator import generator
from TemDiscriminator import TemDiscriminator
from SpaDiscriminator import SpaDiscriminator
from utils import w,Norm_1_numpy,Norm_1_torch

from avm import AverageMeter, accuracy


from dataset_radar import TrainDataset,TestDataset

from tqdm import tqdm
from PIL import Image
from pip import main

from torch import Tensor
from torchvision import datasets
from torchvision import transforms 


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "2"



parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--num_epoch', default=5, type=int,
                    help='the epoch num')
parser.add_argument('--rec_iter', default=70, type=int,
                    help='for each --rec_iter num iterations record the result')
parser.add_argument('--img_test_pth', default="./img/result_test", 
                    help='the path for saving the img generated in the test phase')     
parser.add_argument('--batch_size', default=8, type=int,
                    help='for each --rec_iter num iterations record the result')           
args = parser.parse_args()
dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)


cuda = True if torch.cuda.is_available() else False



BATCHSIZE=args.batch_size
VAL_BATCH=2
M=4
N=22
H=256
W=256
Lambda=20
num_epoch=args.num_epoch
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# 创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out

# 图像预处理

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
# ])

# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]
mean, std = [80.0], [0.0]
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)  # (x-mean) / std
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)  # (x-mean) / std
])

train_dataset = TrainDataset(path = '/data1/shuliang/Radar_900/train/', transforms=img_transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCHSIZE, shuffle=False, sampler=train_sampler
) 

val_dataset = TrainDataset(path = '/data1/shuliang/Radar_900/val/', transforms=val_transform)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=VAL_BATCH, shuffle=False, sampler=val_sampler
) 

test_dataset = TestDataset(path = '/data1/shuliang/Radar_900/test/')
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False, sampler=test_sampler
)

SDis=SpaDiscriminator()
TDis=TemDiscriminator()
TDis=torch.nn.parallel.DistributedDataParallel(TDis.cuda(), find_unused_parameters=True, device_ids=[args.local_rank])
G = generator(N)
G = torch.nn.parallel.DistributedDataParallel(G.cuda(), device_ids=[args.local_rank])
RELU = nn.ReLU()

# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCEWithLogitsLoss()  # 是单目标二分类交叉熵函数
# criterion = nn.MSELoss()
SDis_optimizer = torch.optim.Adam(SDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
TDis_optimizer = torch.optim.Adam(TDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),betas=(0.0, 0.999), lr=0.00005)




# if torch.cuda.is_available():
#     SDis = SDis.cuda()
#     TDis = TDis.cuda()
#     G = G.cuda()
    # device = torch.device("cuda")
    # G = G.to(device)
    # SDis = SDis.to(device)
    # TDis = TDis.to(device)
    
# ##########################进入训练##判别器的判断过程#####################
def cal_loss(a,b):
    a_length = a.size(1)
    loss = 0.0
    for ei in range(a_length):
        loss += torch.sum(((a[:, ei, :, :] - b[:, ei, :, :]) ** 2)) + \
                torch.sum((torch.abs(a[:, ei, :, :] - b[:, ei, :, :])))
    return loss / a_length

def train(G):
    for epoch in range(num_epoch):  # 进行多个epoch的训练
        print('第'+str(epoch)+'次迭代')

        if epoch % 30 == 1 and epoch > 10:
            torch.save(G.state_dict(), './' +str(epoch) +'_generator.pth') 

        loss_gen_av = AverageMeter()
        loss_tid_av = AverageMeter()

        p_bar = tqdm(dataloader)

        for i, img in enumerate(p_bar):
            if i > 20:
                break
            # training in the iteration
            img = img.cuda(non_blocking=True)
            img = torch.squeeze(img, -1)
            valid = Variable(Tensor(img.size(0)).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(img.size(0)).fill_(0.0), requires_grad=False)
            # init the valid and the fake matrix here 
            real_imgs = Variable(img.type(Tensor))

            fst_half = real_imgs[:, :M, : , : ]
            scd_half = real_imgs[:, M:, : , : ]

            # ------------------
            # Train Generator
            # ------------------
            g_optimizer.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE ,8 ,8 ,8))))
            fake_output = G(fst_half, z)
            TD_input_fake = torch.cat((fst_half, fake_output), dim=1)
            g_loss = criterion(TDis(TD_input_fake),valid)
            # g_loss = cal_loss(scd_half,fake_output)
            loss_gen_av.update(g_loss.item())
            g_loss.backward()
            g_optimizer.step()

            # ------------------------
            # Train Time Discriminator
            # ------------------------
            TDis_optimizer.zero_grad()
            TD_input_real = torch.cat((fst_half, scd_half), dim=1)
            TD_input_fake = torch.cat((fst_half, fake_output.detach()), dim=1)

            td_real_loss = criterion(TDis(TD_input_real),valid)
            td_fake_loss = criterion(TDis(TD_input_fake), fake)
            td_loss = (td_real_loss + td_fake_loss) / 2
            loss_tid_av.update(td_loss.item())
            td_loss.backward()
            TDis_optimizer.step()

            infor_per_iter = "Train Epoch: {epoch}/{epochs:4}. gen_loss: {g_loss:.4f}. TD_loss: {td_loss:.4f}".format(
                        epoch=epoch + 1,
                        epochs=num_epoch,
                        g_loss=loss_gen_av.avg,
                        td_loss = loss_tid_av.avg,
                        )
            p_bar.set_description(infor_per_iter)
            p_bar.update()

        loss_gen_av.save(infor_per_iter, "./loss_result/")
        p_bar.close()
        validation(G, val_dataloader)
    print('....................................')

def validation(model, dataloader):
    # Validate the generator performance
    criterion = nn.MSELoss()
    loss_am = AverageMeter()
    p_bar = tqdm(dataloader)
    for i, img in enumerate(p_bar):
        img = img.cuda(non_blocking=True)
        img = torch.squeeze(img, -1)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))
        fst_half = real_imgs[:, :M, : , : ]
        scd_half = real_imgs[:, M:, : , : ]
        z = Variable(Tensor(np.random.normal(0, 1, (VAL_BATCH ,8 ,8 ,8))))
        
        fake_output = model(fst_half, z)

        g_loss = criterion(scd_half,fake_output)
        loss_am.update(g_loss.item())
        p_bar.set_description("VAL Epoch: {iteration}/{iterations:4}. gen_loss: {g_loss:.4f}".format(
                    iteration = i,
                    iterations= len(dataloader),
                    g_loss=loss_am.avg,
                ))
        p_bar.update()
    p_bar.close()
    return loss_am

def test(model, dataloader):
    criterion = nn.MSELoss()
    loss_gen_av = AverageMeter()
    p_bar = tqdm(len(dataloader))
    for i, img in enumerate(dataloader):
        img = img.cuda(non_blocking=True)
        img = torch.squeeze(img, -1)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))
        fst_half = real_imgs[:, :M, : , : ]
        scd_half = real_imgs[:, M:, : , : ]

        z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE ,8 ,8 ,8))))
        
        fake_output = model(fst_half, z)

        g_loss = criterion(scd_half,fake_output)
        loss_gen_av.update(g_loss.item())

        if i % args.rec_iter == 1:
            pth = os.path.join(args.img_test_pth, str(i))
            if not os.path.isdir(pth):
                os.makedirs(pth)
            # 只选择Batch中的第一套图片做存储
            ri = real_imgs[0].cpu().detach().numpy()
            fo = fake_output[0].cpu().detach().numpy()
            for idx, img_it in enumerate(fo):
                # 还原图像
                img_gt = np.uint8(img_it * 80)
                mask = img_gt < 1
                img_gt = 255 * mask + (1 - mask) * img_gt
                # 存储图像
                sv_pth = os.path.join(pth,'fake_out_'+str(idx)+'.png')
                cv2.imwrite(sv_pth, img_gt)
            for idx, img_it in enumerate(ri):
                # 还原图像
                img_gt = np.uint8(img_it * 80)
                mask = img_gt < 2
                img_gt = 255 * mask + (1 - mask) * img_gt
                # 存储图像
                sv_pth = os.path.join(pth, "real_img_"+str(idx)+'.png')
                cv2.imwrite(sv_pth, img_gt)
        p_bar.set_description("Test Epoch: {iteration}/{iterations:4}. gen_loss: {g_loss:.4f}".format(
                    iteration = i,
                    iterations= 600,
                    g_loss=loss_gen_av.avg,
                    ) )
        p_bar.update()
    p_bar.close()
G.load_state_dict(torch.load("./model_pth/mse_trained_31epoch.pth"))
train(G) 
# test(G,test_dataloader)



# 保存模型
torch.save(G.state_dict(), './generator.pth')  
torch.save(SDis.state_dict(), './SpaDiscriminator.pth')
torch.save(TDis.state_dict(), './TemDiscriminator.pth')