

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np



class file_handler():
    def __init__(self, path, epoch_n, int_n, cuda_n, name = ''):
        
        self.path = path
        self.name = name 

        self.epoch_n = epoch_n 
        self.int_n = int_n 
        self.cuda_n = cuda_n

        self.gen_loss_ep = [] 
        self.gen_loss_it = []
        self.val_loss_ep = []
        
        # file will be handled by the following function 
        # self.gen_loss_ep  == generator loss by epoch 
        # self.gen_loss_it  == generator loss by iteration 
        # self.val_loss_ep  == validation loss by epoch 
        self.handle_file()
        self.iter_n = len(self.gen_loss_it) / self.epoch_n


    def handle_file(self):
        self.data_dict = {} 
        file = open(self.path, "r")
        cuda_counter = 0 
        tem_saver = 0.0
        for idx, line in enumerate(file.readlines()):
            if len(line) < 0:
                continue 
            else:
                if line.startswith("Train"):
                    # ['Train Epoch: 1/   5', 'Iteration:0', 'gen_loss: 0.5956', 'TD_loss: 0.7092\n']
                    epoch_num = int(line[line.find('h:')+2 :line.find('/')] )
                    gen_loss  = float(line[line.find('gen_loss:')+ len('gen_loss:') :   line.find('gen_loss:')+ len('gen_loss:') + len('0.1111')])
                    cuda_counter += 1 
                    tem_saver += gen_loss
                    if cuda_counter == self.cuda_n:
                        cuda_counter = 0
                        self.gen_loss_it.append(tem_saver / self.cuda_n)
                        tem_saver = 0.0

                elif line.startswith("VAL"):
                    if cuda_counter == 0:
                        self.gen_loss_ep.append(self.gen_loss_it[-1])
                    val_loss  = float(line[line.find('gen_loss:')+ len('gen_loss:') :   line.find('gen_loss:')+ len('gen_loss:') + len('0.1111')])
                    cuda_counter += 1
                    tem_saver += val_loss
                    if cuda_counter == self.cuda_n:
                        cuda_counter = 0
                        self.val_loss_ep.append(tem_saver / self.cuda_n)
                        tem_saver = 0.0 
        file.close()
    
    def eraseInt(self, data):
        new_data = [] 
        if self.int_n == 1 :
            return data 
        for idx, d in  enumerate(data):
            if idx % self.int_n == 0 :
                continue 
            else:
                new_data.append(d)
        return new_data 



    
def plot(data, path = './img/plot_figure/1.png'):
    x = list(range(len(data)))
    figure = plt.figure(figsize=(10,10))
    # ax = figure.add_subplot(1,1,1)
    plt.plot(data)
    plt.title("data")
    plt.savefig(path, dpi=300)

def plot_val(files, pth = './img/plot_figure/val.png', fig_name = "i123"):
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch ')
    ax.set_ylabel('val_loss')
    ax.axis([0, 50, 0, 0.3])
    for file in files :
        ax.plot(file.val_loss_ep, label = file.name)
    ax.legend()
    ax.set_title(fig_name)
    plt.savefig(pth, dpi = 300 )

def plot_gen(files, pth = './img/plot_figure/loss.png', fig_name = 'loss'):
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch ')
    ax.set_ylabel('gen_loss')
    # ax.axis([0, 50, 0, 0.3])
    for file in files :
        ax.plot(file.eraseInt(file.gen_loss_ep), label = file.name)
    ax.legend()
    ax.set_title(fig_name)
    plt.savefig(pth, dpi = 300 )



                
if __name__ == "__main__":
    att_int4 = file_handler(path= "./loss_result/att_int4result.csv", epoch_n = 50, int_n = 4, cuda_n = 4, name = 'att i4')
    att_int8 = file_handler(path= "./loss_result/att_int8result.csv", epoch_n = 36, int_n = 8, cuda_n = 1, name = 'att i8')
    att_avg4 = file_handler(path= "./loss_result/att_avg4result.csv", epoch_n = 36, int_n = 4, cuda_n = 1, name = 'avg i4')
    mse_rslt = file_handler(path= "./loss_result/mse_result.csv", epoch_n = 50, int_n = 1, cuda_n = 4, name = 'mse')
    sd_td    = file_handler(path= "./loss_result/SD_TD.csv", epoch_n = 50, int_n = 4, cuda_n = 4, name = 'sd td')
    i8_total = file_handler(path= "./loss_result/3090/loss_result/int_8_total/result.csv", epoch_n = 36, int_n = 8, cuda_n = 1, name= 'i8')
    i10total = file_handler(path= "./loss_result/3090/loss_result/int_10_total/result.csv", epoch_n = 36, int_n = 10, cuda_n = 1, name = 'i10')
    i4_total  = file_handler(path= "./loss_result/3090/loss_result/lr_total/result.csv", epoch_n = 50, int_n = 4, cuda_n = 1, name = 'i4 dgmr')


    # plot_gen([att_int4, att_int8, i8_total, i4_total], pth = './img/plot_figure/att_com_tr.png', fig_name= 'Attention Comparison Train Loss')
    # plot_val([att_int4, att_int8, i8_total, i4_total], pth = './img/plot_figure/att_com_va.png', fig_name= 'Attention Comparison Val Loss')

    # plot_gen([i4_total, sd_td, mse_rslt], pth = './img/plot_figure/ablation_tr.png', fig_name= 'Ablation Comparison Train Loss')
    # plot_val([i4_total, sd_td, mse_rslt], pth = './img/plot_figure/ablation_vl.png', fig_name= 'Ablation Comparison Val Loss')
    
    plot_gen([ i4_total, i8_total, i10total], pth = './img/plot_figure/int_tr.png', fig_name= 'Interval Comparison Train Loss')
    plot_val([ i4_total, i8_total, i10total], pth = './img/plot_figure/int_vl.png', fig_name= 'Interval Comparison VAL Loss')
