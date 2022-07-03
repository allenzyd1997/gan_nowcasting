
# coding=utf-8
import argparse
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import os
import cv2
import random
from generator import generator
from TemDiscriminator import TemDiscriminator
from SpaDiscriminator import SpaDiscriminator
from utils import w,Norm_1_numpy,Norm_1_torch,save

from avm import AverageMeter



from dataset_radar import TrainDataset,TestDataset

from tqdm import tqdm
from PIL import Image
from pip import main

from torch import Tensor
from torchvision import datasets
from torchvision import transforms 

from val_test import validation

criterion = nn.BCEWithLogitsLoss()  # 是单目标二分类交叉熵函数
mse = nn.MSELoss()
SDis_optimizer = torch.optim.Adam(SDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
TDis_optimizer = torch.optim.Adam(TDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),betas=(0.0, 0.999), lr=0.00005)
def train(G, SDis, TDis, args):
    
    criterion = nn.BCEWithLogitsLoss()  # 是单目标二分类交叉熵函数
    mse = nn.MSELoss()
    SDis_optimizer = torch.optim.Adam(SDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
    TDis_optimizer = torch.optim.Adam(TDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(),betas=(0.0, 0.999), lr=0.00005)

    for epoch in range(args.num_epoch):  # 进行多个epoch的训练
        print('第'+str(epoch)+'次迭代')
        if epoch % 10 == 1 and epoch > 5:
            torch.save(G.state_dict(), os.path.join(out_model_pth, str(epoch))+'g.pth' ) 
            torch.save(TDis.state_dict(), os.path.join(out_model_pth, str(epoch))+'td.pth' ) 
            torch.save(SDis.state_dict(), os.path.join(out_model_pth, str(epoch))+'sd.pth' ) 

        loss_gen_av = AverageMeter()
        loss_tid_av = AverageMeter()
        loss_spd_av = AverageMeter()
        loss_dis_av = AverageMeter()
        loss_reg_av = AverageMeter()

        p_bar = tqdm(dataloader)
        flag = epoch % args.interval == 0
        # flag决定了是否在这个epoch训练D并用D的loss训练G， 否则会用MSE
        # flag = True ; flag = False

        for i, img in enumerate(p_bar):
            # training in the iteration
            img = img.cuda(non_blocking=True)
            img = torch.squeeze(img, -1)
            valid = Variable(Tensor(img.size(0)).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(img.size(0)).fill_(0.0), requires_grad=False)
            # init the valid and the fake matrix here 
            real_imgs = Variable(img.type(Tensor))

            fst_half = real_imgs[:, :args.M, : , : ]
            scd_half = real_imgs[:, args.M:, : , : ]

            # ------------------
            # Train Generator
            # ------------------
            g_optimizer.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (args.mean_num, args.batch_size ,8 ,8 ,8))))
            fake_output = G(fst_half, z)
            S = random.sample(range(0, args.N-args.M), 8)
            S.sort()

            TD_input_fake = torch.cat((fst_half, fake_output), dim=1)
            SD_input_fake = fake_output[:, S]
            if flag: 
                r_loss_sum = 0 
                for j in range(fake_output.shape[0]):
                    result = torch.mul((fake_output[j] - scd_half[j]), scd_half[j])
                    r_loss = (1 / args.H*args.W*args.N) * args.L * Norm_1_torch(result)
                    r_loss_sum += r_loss

                r_loss_sum = r_loss_sum / fake_output.shape[0]

                loss_reg_av.update(r_loss_sum)

                g_loss = criterion(TDis(TD_input_fake),valid) + criterion(SDis(SD_input_fake),valid) + r_loss_sum
                loss_gen_av.update(g_loss.item())
                g_loss.backward()

            else:
                g_loss = mse(scd_half, fake_output)
                loss_gen_av.update(g_loss.item())
                g_loss.backward()

            g_optimizer.step()

            # ------------------------
            # Train Discriminator
            # ------------------------
            if flag :
                TDis_optimizer.zero_grad()
                TD_input_real = torch.cat((fst_half, scd_half), dim=1)
                TD_input_fake = torch.cat((fst_half, fake_output.detach()), dim=1)

                td_real_loss = criterion(TDis(TD_input_real),valid)
                td_fake_loss = criterion(TDis(TD_input_fake), fake)
                td_loss = RELU(1-td_real_loss) + RELU(1+td_fake_loss)
                loss_tid_av.update(td_loss.item())

                SDis_optimizer.zero_grad()
                SD_input_real = scd_half[:, S]
                SD_input_fake = fake_output.detach()[:, S]

                sd_real_loss = criterion(SDis(SD_input_real), valid)
                sd_fake_loss = criterion(SDis(SD_input_fake), fake)
                sd_loss = RELU(1-sd_real_loss) + RELU(1+sd_fake_loss)
                loss_spd_av.update(sd_loss.item())

                d_loss = td_loss + sd_loss 
                d_loss.backward()
                SDis_optimizer.step()
                TDis_optimizer.step()

                loss_dis_av.update(d_loss.item())

            infor_per_iter = "Train Epoch: {epoch}/{epochs:4}. Iteration: {iteration:4} gen_loss: {g_loss:.4f}. TD_loss: {td_loss:.4f}. SD_loss: {sd_loss:.4f}. D_loss: {d_loss:.4f}. R_loss: {r_loss:.4f}.".format(
                        epoch=epoch + 1,
                        epochs=args.num_epoch,
                        iteration=i,
                        g_loss=loss_gen_av.avg,
                        td_loss = loss_tid_av.avg,
                        sd_loss = loss_spd_av.avg,
                        d_loss  = loss_dis_av.avg,
                        r_loss  = loss_reg_av.avg,
                        )
            p_bar.set_description(infor_per_iter)
            p_bar.update()

              
            save(infor_per_iter, args.exp_name,"./loss_result/")

        p_bar.close()
        validation(G, val_dataloader, args)
    print('....................................')
