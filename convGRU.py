import torch
import torch.nn as nn

class ConvGRU_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvGRU_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding, bias=self.bias)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding, bias=self.bias)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, input_, prev_state):  # x [batch, hidden_dim, width, height]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class ConvGRU(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, kernel_size):
        super(ConvGRU, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        self.kernel_size = kernel_size
        self.H  = []

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvGRU_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_,): 

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])

        return self.H[-1]  # (hidden, output)
    def setHidden(self, hidden):
        self.H = [hidden]
        


