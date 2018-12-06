import torch
from torch import nn
from models.convlstm.convlstm import ConvLSTMCell

class LSTMSequentialEncoder(nn.Module):
    def __init__(self, height, width, input_dim, hidden_dim, kernel_size=(3,3), bias=False):
        super(LSTMSequentialEncoder, self).__init__()

        self.cell = ConvLSTMCell(input_size=(height, width),
                     input_dim=input_dim,
                     hidden_dim=hidden_dim,
                     kernel_size=kernel_size,
                     bias=bias)


    def forward(self, input, hidden=None, state=None):

        b, t, c, h, w = input.shape

        if hidden is None:
            hidden = torch.zeros((b, c, h, w))
        if state is None:
            state = torch.zeros((b, c, h, w))

        for iter in range(t):

            hidden, state = self.cell.forward(input[:,iter,:,:,:], (hidden, state))

        return hidden, state


if __name__=="__main__":


    b, t, c, h, w = 2, 10, 3, 320, 320

    model = LSTMSequentialEncoder(height=h, width=w, input_dim=c, hidden_dim=3)


    x = torch.randn((b,t,c,h,w))

    hidden, state = model.forward(x)


