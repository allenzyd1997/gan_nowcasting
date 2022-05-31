
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
args = parser.parse_args()
dist.init_process_group(backend='nccl')

torch.cuda.set_device(args.local_rank)


cuda = True if torch.cuda.is_available() else False



BATCHSIZE=4
M=4
N=22
H=256
W=256
Lambda=20
num_epoch=3
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
    transforms.Normalize(mean, std)  # (x-mean) / std
])
train_dataset = TrainDataset(path = '/data1/shuliang/Radar_900/train/', transforms=img_transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCHSIZE, shuffle=False, sampler=train_sampler
) 

test_dataset = TestDataset(path = '/data1/shuliang/Radar_900/test/')
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False, sampler=test_sampler
)



SDis=SpaDiscriminator()
TDis=TemDiscriminator()
G = generator(N)
G = torch.nn.parallel.DistributedDataParallel(G.cuda(), device_ids=[args.local_rank])
RELU = nn.ReLU()

# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
# criterion = nn.BCEWithLogitsLoss()  # 是单目标二分类交叉熵函数
criterion = nn.MSELoss()
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
for epoch in range(num_epoch):  # 进行多个epoch的训练
    print('第'+str(epoch)+'次迭代')

    loss_gen_av = AverageMeter()

    p_bar = tqdm(len(dataloader))
    for i, img in enumerate(dataloader):
        img = img.cuda(non_blocking=True)
        img = torch.squeeze(img, -1)
        valid = Variable(Tensor(img.size(0)).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(img.size(0)).fill_(0.0), requires_grad=False)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))

        fst_half = real_imgs[:, :M, : , : ]
        scd_half = real_imgs[:, M:, : , : ]

        z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE ,8 ,8 ,8))))
        
        fake_output = G(fst_half, z)
        g_loss = criterion(scd_half,fake_output)
        loss_gen_av.update(g_loss.item())

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. gen_loss: {g_loss:.4f}".format(
                    epoch=epoch + 1,
                    epochs=num_epoch,
                    g_loss=loss_gen_av.avg,
                    ) )
        p_bar.update()
    p_bar.close()
print('....................................')


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
        p_bar.set_description("Train Epoch: {iteration}/{iterations:4}. gen_loss: {g_loss:.4f}".format(
                    iteration = i,
                    iterations= 600,
                    g_loss=loss_gen_av.avg,
                    ) )
        p_bar.update()
    p_bar.close()

test(G,test_dataloader )



# 保存模型
torch.save(G.state_dict(), './generator.pth')  
torch.save(SDis.state_dict(), './SpaDiscriminator.pth')
torch.save(TDis.state_dict(), './TemDiscriminator.pth')