
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()

################## CONFIG PARAMETER ###################

parser.add_argument('--img_test_pth', default="./img/result_test", 
                    help='the path for saving the img generated in the test phase')  
parser.add_argument('--dataset_pth' , default='/data1/shuliang/Radar_900',
                    help='')

################## HYPER PARAMETER ####################
parser.add_argument('--num_epoch', default=50, type=int,
                    help='the epoch num')   
parser.add_argument('--batch_size', default=3, type=int,
                    help='batch size ')           
parser.add_argument('--val_batch', default=1, type=int,
                    help='validation batch size ')                      
parser.add_argument('--interval', default=4, type=int,
                    help='for each interval to use the gan training ')
parser.add_argument('--M', default=4, type=int,
                    help='the input images amount of generator')
parser.add_argument('--N', default=22, type=int,
                    help='the total images amout of each sequence')
parser.add_argument('--H', default=256, type=int,
                    help='Height of the image')
parser.add_argument('--W', default=256, type=int,
                    help='Width of the image')
parser.add_argument('--L', default=20, type=int,
                    help='regularization factor lambda')

################## OUTPUT PARAMETER ####################

parser.add_argument('--rec_iter', default=10, type=int,
                    help='for each --rec_iter num iterations record the result')   
parser.add_argument('--exp_name', default="no_name_exp")                                        


################## EXP_TEST PARAMETER ####################
parser.add_argument('--attention', action='store_true')

parser.add_argument('--mean_num', default=1, type=int,
                    help='how many z will be input to get the mean value')   

args = parser.parse_args()

print("实验： " + args.exp_name + " 开始")
cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# 创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')

out_model_pth =  os.path.join('./result_dir/', args.exp_name)
if not os.path.exists(out_model_pth):
    os.mkdir(out_model_pth)
    #我只在里面存模型， loss 还是放loss_reslult


# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)  # (x-mean) / std
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)  # (x-mean) / std
])


########## train val test dataloader的定义 ##########
train_dataset = TrainDataset(path = os.path.join( args.dataset_pth, 'train/'), transforms=img_transform)
dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=False
) 

val_dataset = TrainDataset(path = os.path.join(args.dataset_pth, 'val/'), transforms=val_transform)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=args.val_batch, shuffle=False,
) 

test_dataset = TestDataset(path = os.path.join(args.dataset_pth, 'test/'))
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=args.val_batch, shuffle=False,
)
####################网络结构定义#######################
SDis=SpaDiscriminator().cuda()
TDis=TemDiscriminator().cuda()
G = generator(args).cuda()
# TDis=torch.nn.parallel.DistributedDataParallel(TDis.cuda(), device_ids=[args.local_rank])

RELU = nn.ReLU()

criterion = nn.BCEWithLogitsLoss()  # 是单目标二分类交叉熵函数
mse = nn.MSELoss()
SDis_optimizer = torch.optim.Adam(SDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
TDis_optimizer = torch.optim.Adam(TDis.parameters(),betas=(0.0, 0.999), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),betas=(0.0, 0.999), lr=0.00005)


# ##########################进入训练##判别器的判断过程#####################

def train(G, args):
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
        validation(G, val_dataloader,args)
    print('....................................')

def validation(model, dataloader, args):
    # Validate the generator performance
    criterion = nn.MSELoss()
    loss_am = AverageMeter()
    p_bar = tqdm(dataloader)
    for i, img in enumerate(p_bar):
        img = img.cuda(non_blocking=True)
        img = torch.squeeze(img, -1)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))
        fst_half = real_imgs[:, :args.M, : , : ]
        scd_half = real_imgs[:, args.M:, : , : ]
        z = Variable(Tensor(np.random.normal(0, 1, (1, args.val_batch ,8 ,8 ,8))))
        
        fake_output = model(fst_half, z)

        g_loss = criterion(scd_half,fake_output)
        loss_am.update(g_loss.item())
        infor_per_iter = "VAL Epoch: {iteration}/{iterations:4}. gen_loss: {g_loss:.4f}".format(
                    iteration = i,
                    iterations= len(dataloader),
                    g_loss=loss_am.avg,
                )
        p_bar.set_description(infor_per_iter)
        p_bar.update()
    p_bar.close()
    save(infor_per_iter,args.exp_name, "./loss_result/")
    return loss_am

def test(model, dataloader):
    criterion = nn.MSELoss()
    loss_gen_av = AverageMeter()
    p_bar = tqdm(len(dataloader))
    rec = 0 
    for i, (img, apth) in enumerate(dataloader):
        img = img.cuda(non_blocking=True)
        img = torch.squeeze(img, -1)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))
        fst_half = real_imgs[:, :args.M, : , : ]
        scd_half = real_imgs[:, args.M:, : , : ]

        z = Variable(Tensor(np.random.normal(0, 1, (1, args.val_batch ,8 ,8 ,8))))
        
        fake_output = model(fst_half, z)

        g_loss = criterion(scd_half,fake_output)
        loss_gen_av.update(g_loss.item())

        # if i % args.rec_iter == 1:
        if True:
            pth = os.path.join(args.img_test_pth, str(i))
            if not os.path.isdir(pth):
                os.makedirs(pth)
            # 只选择Batch中的第一套图片做存储
            ri = real_imgs[0].cpu().detach().numpy()
            fo = fake_output[0].cpu().detach().numpy()
            for idx, img_it in enumerate(ri):
                # 还原图像
                img_gt = np.uint8(img_it * 80)
                mask = img_gt < 1
                img_gt = 255 * mask + (1 - mask) * img_gt
                # 存储图像
                sv_pth = os.path.join(pth, "real_img_"+str(idx)+'.png')
                cv2.imwrite(sv_pth, img_gt)
            for idx, img_it in enumerate(fo):
                # 还原图像
                img_gt = np.uint8(img_it * 80).reshape(args.H,args.W,1)
                mask = img_gt < 1
                img_gt = 255 * mask + (1 - mask) * img_gt
                # 存储图像
                sv_pth = os.path.join(pth,'fake_out_'+str(idx)+'.png')
                cv2.imwrite(sv_pth, img_gt)

        p_bar.set_description("Test Epoch: {iteration}/{iterations:4}. gen_loss: {g_loss:.4f}".format(
                    iteration = i,
                    iterations= 600,
                    g_loss=loss_gen_av.avg,
                    ) )
        p_bar.update()
    p_bar.close()

# G.load_state_dict(torch.load("./model_pth/SD+TD41.pth"))
TRAIN_OR_NOT = True
if TRAIN_OR_NOT:
    train(G,args)
    torch.save(G.state_dict(), os.path.join(out_model_pth, '_final_')+'g.pth')  
    torch.save(SDis.state_dict(), os.path.join(out_model_pth, '_final_')+'sd.pth')
    torch.save(TDis.state_dict(), os.path.join(out_model_pth, '_final_')+'td.pth')
else:
    test(G,test_dataloader,args)


