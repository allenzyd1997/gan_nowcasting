import numpy as np
import matplotlib.pyplot as plt

loss_file_pth = "./loss_result/test_result.csv"

class loss_ploter():
    def __init__(self, path = loss_file_pth):
        self.path = path
        self.gen_loss_dict = {}
        self.td_loss_dict = {} 
        self.handle_file()


    def handle_file(self):
        self.data_dict = {} 
        file = open(self.path, "r")
        for idx, line in enumerate(file.readlines()):
            if len(line) < 0:
                continue 
            else:
                if line.startswith("Train"):
                    line = line.split(". ")
                    # ['Train Epoch: 1/   5', 'Iteration:0', 'gen_loss: 0.5956', 'TD_loss: 0.7092\n']
                    epoch_num = int(line[0][line[0].find(':')+2 : line[0].find('/')])
                    iter_num  = int(line[1][line[1].find(':')+1 :])
                    gen_loss  = float(line[2][line[2].find(':')+1 : ])
                    td_loss   = float(line[3][line[3].find(':')+1 : -1])
                    
                    self.add_dict(epoch_num, iter_num, self.gen_loss_dict, gen_loss)
                    self.add_dict(epoch_num, iter_num, self.td_loss_dict, td_loss)

        self.avg_dict(self.gen_loss_dict)
        self.avg_dict(self.td_loss_dict)
        file.close()

    def add_dict(self, epoch_num, iter_num, dict, val):
            if epoch_num not in dict.keys():
                dict[epoch_num] = {}

            cd = dict[epoch_num] 

            if iter_num not in cd.keys():
                cd[iter_num] = val
            else:
                cd[iter_num] += val
    
    def avg_dict(self, dict):
        for epoch in dict.keys():
            for iter in dict[epoch].keys():
                dict[epoch][iter] = float( '%.4f' % (dict[epoch][iter] / 4))

    def gen_epoch(self, dict):
        epochs = dict.keys()
        data   = [] 
        for epoch in epochs:
            iters = list(dict[epoch].keys())
            liv   = dict[epoch][iters[-1]]
            data.append(liv)
        return data 

    def gen_iter(self, dict, interve = 1):
        epochs = dict.keys()
        data   = [] 
        for epoch in epochs:
            iters = dict[epoch].keys()
            for idx, iter in enumerate(iters):
                if idx % interve == 0:
                    data.append(dict[epoch][iter])
        return data
    
    def plot(self, data, path = './img/plot_figure/1.png'):
        x = list(range(len(data)))
        figure = plt.figure(figsize=(10,10))
        # ax = figure.add_subplot(1,1,1)
        plt.plot(data)
        plt.title("data")
        plt.savefig(path, dpi=300)
        figure.show()
                

if __name__ == "__main__":
    ploter = loss_ploter(path= loss_file_pth)
    data   = ploter.gen_iter(ploter.gen_loss_dict, interve = 1)
    ploter.plot(data, './img/plot_figure/1.png')


