import torch.nn as nn


def getList(file, child=True):
    listname = []
    for line in open(file, 'r').readlines():
        ls = line.split(',')
        if child:
            for cl in ls[1:]:
                if '\n' in cl:
                    cl = cl[:-1]
                listname.append(cl)
        else:
            listname.append(ls[0])
    return listname



def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def mean_squared_error(y, t):
    '''Calculate the weight difference between two epoch'''
    return 0.5 * np.sum((y - t) ** 2)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x