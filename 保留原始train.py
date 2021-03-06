
# coding=utf-8

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


from dataset_radar import TrainDataset

from tqdm import tqdm
from PIL import Image
from pip import main

from torch import Tensor
from torchvision import datasets
from torchvision import transforms 



os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "2"


cuda = True if torch.cuda.is_available() else False



BATCHSIZE=4
M=4
N=22
H=256
W=256
Lambda=20
num_epoch=5
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
radar_l = TrainDataset(path = '/data1/shuliang/Radar_900/train/', transforms=img_transform)

dataloader = torch.utils.data.DataLoader(
    dataset=radar_l, batch_size=BATCHSIZE, shuffle=False
)


SDis=SpaDiscriminator()
TDis=TemDiscriminator()
G = generator(N)
RELU = nn.ReLU()

# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCEWithLogitsLoss()  # 是单目标二分类交叉熵函数
SDis_optimizer = torch.optim.Adam(SDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
TDis_optimizer = torch.optim.Adam(TDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),betas=(0.0, 0.999), lr=0.00005)

if torch.cuda.is_available():
    SDis = SDis.cuda()
    TDis = TDis.cuda()
    G = G.cuda()
    # device = torch.device("cuda")
    # G = G.to(device)
    # SDis = SDis.to(device)
    # TDis = TDis.to(device)

# ##########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    print('第'+str(epoch)+'次迭代')

    loss_gen_av = AverageMeter()
    loss_sd_av  = AverageMeter()
    loss_td_av  = AverageMeter()
    loss_dis_av = AverageMeter()

    p_bar = tqdm(len(dataloader))

    for i, img in enumerate(dataloader):

        img = torch.squeeze(img, -1)
        valid = Variable(Tensor(img.size(0)).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(img.size(0)).fill_(0.0), requires_grad=False)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))

        fst_half = real_imgs[:, :M, : , : ]
        scd_half = real_imgs[:, M:, : , : ]

        z = Variable(Tensor(np.random.normal(0, 1, (BATCHSIZE ,8 ,8 ,8))))
        
        fake_output = G(fst_half, z).detach()
        
        fake_output_combine = torch.cat((fst_half, fake_output),dim=1)

        # generate a batch of images 
        # X_real_second_half = Variable(X_real_second_half).cuda()  # 将tensor变成Variable放入计算图中
        S = random.sample(range(0, N-M), 8)
        S.sort()
        scd_half_chosed = scd_half[:,S]
        fake_output_chosed = fake_output[:,S]
        # 十八张图中挑选八张用于Temporal的预测
        SD_out_real = SDis(scd_half_chosed)  # 将真实图片放入判别器中
        real_imgs = torch.squeeze(real_imgs)
        TD_out_real = TDis(real_imgs)

        SD_out_fake = SDis(fake_output_chosed)
        TD_out_fake = TDis(fake_output_combine)

        sd_loss_real = criterion(SD_out_real, valid)
        td_loss_real = criterion(TD_out_real, valid)
        sd_loss_fake = criterion(SD_out_fake, fake)
        td_loss_fake = criterion(TD_out_fake, fake)

        sd_loss = RELU(1-sd_loss_real) + RELU(1+sd_loss_fake)
        td_loss = RELU(1-td_loss_real) + RELU(1+td_loss_fake)

        d_loss = sd_loss + td_loss
        

        loss_dis_av.update(d_loss.item())
        loss_sd_av.update(sd_loss.item())
        loss_td_av.update(td_loss.item())

        SDis_optimizer.zero_grad()
        TDis_optimizer.zero_grad()
        d_loss.backward()
        SDis_optimizer.step()
        TDis_optimizer.step()


        SD_out_fake = SDis(fake_output_chosed)
        TD_out_fake = TDis(fake_output_combine)
        sd_loss_fake = criterion(SD_out_fake, fake)
        td_loss_fake = criterion(TD_out_fake, fake)        

        r_loss_sum = 0 
        for i in range(fake_output.shape[0]):
            result = torch.mul((fake_output[i] - scd_half[i]), scd_half[i])
            r_loss = (1 / H*W*N) * Lambda * Norm_1_torch(result)
            r_loss_sum += r_loss
        
        g_loss = td_loss_fake + td_loss_fake + r_loss_sum / fake_output.shape[0]
        # g_loss = td_loss_fake + td_loss_fake 

        loss_gen_av.update(g_loss.item())
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. gen_loss: {g_loss:.4f}. dis_loss: {d_loss:.4f}. td_loss: {td_loss:.4f}. sd_loss: {sd_loss:.4f}".format(
                    epoch=epoch + 1,
                    epochs=num_epoch,
                    g_loss=loss_gen_av.avg,
                    d_loss=loss_dis_av.avg,
                    td_loss=loss_td_av.avg,
                    sd_loss=loss_sd_av.avg)
                    ) 
        p_bar.update()


    p_bar.close()
print('....................................')
# 保存模型
torch.save(G.state_dict(), './generator.pth')  
torch.save(SDis.state_dict(), './SpaDiscriminator.pth')
torch.save(TDis.state_dict(), './TemDiscriminator.pth')