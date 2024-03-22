import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.sodegcn import SGNN

# ICN
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self,
                 num_input, num_channel, splitting=True, kernel=3, dropout=0.05,  dilation=1):
        super(Interactor, self).__init__()
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = 0.3
        self.hidden_size = num_channel

        # 设置步长为1
        pad_l = self.dilation * (self.kernel_size-1)
        pad_r = self.dilation * (self.kernel_size-1)

        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []

        modules_P += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_input, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_input, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)
        self.downsample = nn.Conv2d(num_input, num_channel, (1, 1))


    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        d = self.downsample(x_odd).mul(torch.exp(self.phi(x_even)))
        c = self.downsample(x_even).mul(torch.exp(self.psi(x_odd)))

        x_even_update = c + self.U(d)
        x_odd_update = d - self.P(c)

        return (x_even_update, x_odd_update)

#用一层然后拼接
class InteractorNet(nn.Module):
    def __init__(self, num_input, num_channel, kernel, dropout, dilation):
        super(InteractorNet, self).__init__()
        self.level = Interactor(num_input=num_input, num_channel=num_channel, splitting=True, kernel=kernel, dropout=dropout, dilation=dilation)
        self.relu = nn.ReLU()
        # 下采样
        self.downsample = nn.Conv2d(num_input, num_channel, (1, 1))

    def zip_up_the_pants(self, even, odd):
        # even odd shape is (B, C, N, T)
        even_len = even.shape[3]
        odd_len = odd.shape[3]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[:, :, :, i].unsqueeze(3))
            _.append(odd[:, :, :, i].unsqueeze(3))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(3))
        return torch.cat(_, 3)  # B, L, D

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        res = x if self.downsample is None else self.downsample(x)
        (x_even_update, x_odd_update) = self.level(x)
        x_concat = self.zip_up_the_pants(x_even_update, x_odd_update)
        return self.relu((x_concat + res).permute(0, 2, 3, 1))


class SODEGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_features,
                 num_timesteps_input,
                 num_timesteps_output,
                 A_sp_hat,
                 hid_channels,
                 sp_channels,
                 ksize=3,
                 num_levels=3,
                 num_of_split=4,
                 dropout=0.05):
        """
        :param num_nodes: number of nodes in the graph
        :param num_features: number of features at each node in each time step
        :param num_timesteps_input: number of past time steps fed into the network
        :param num_timesteps_output: desired number of future time steps output by the network
        :param A_sp_hat: nomarlized adjacency spatial matrix
        :param A_se_hat: nomarlized adjacency semantic matrix
        :param num_levels: Layers of the ST block
        """
        super(SODEGCN, self).__init__()
        # adjacency graph
        self.hid_channels = hid_channels
        self.sp_channels = sp_channels
        self.ksize = ksize
        self.num_levels = num_levels
        self.dropout = dropout
        self.sp_blocks = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.stode_convs = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=num_features, out_channels=hid_channels, kernel_size=(1, 1), stride=1)
        self.ksize = [2, 3, 5, 6]
        for i in range(num_levels):
            self.filter_convs.append(InteractorNet(num_input=hid_channels, num_channel=hid_channels, kernel=self.ksize[i], dropout=dropout, dilation=1))
            self.stode_convs.append(SGNN(num_nodes=num_nodes, num_features=hid_channels, sp_channels=sp_channels,
                                         num_timesteps_input=num_timesteps_input, num_timesteps_output=num_timesteps_output, num_of_split=num_of_split, adj=A_sp_hat, time=6))
            self.skip_convs.append(nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=(1, num_timesteps_input)))


        self.skip0 = nn.Conv2d(in_channels=num_features, out_channels=hid_channels, kernel_size=(1, num_timesteps_input), stride=1, bias=True)
        # self.skip0 = nn.Conv2d(in_channels=num_features, out_channels=hid_channels, kernel_size=(1, 1), stride=1, bias=True)
        self.skipE = nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=(1, num_timesteps_input), stride=1, bias=True)
        # self.skipE = nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=(1, 1), stride=1, bias=True)
        self.end_conv_1 = nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=(1, 1), bias=True)
        self.ln = nn.LayerNorm([num_nodes, 1])
        self.end_conv_2 = nn.Conv2d(in_channels=hid_channels, out_channels=num_timesteps_output, kernel_size=(1, 1), bias=True)



    def forward(self, x):
        # input shape is (b, n, t, c)
        input = x
        outs = []
        # chage shape from (b, n, t, c) to (b, c, n, t) then (b, n , t, c)
        x = self.start_conv(input.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training).permute(0, 3, 1, 2))
        for i in range(self.num_levels):
            # resx (b, n, t, c)
            resx = x
            # tc
            # filter (b, n, t, c)
            filter = self.filter_convs[i](x)
            # filter = x
            # gate = filter
            filter = torch.tanh(filter)
            # gate = torch.sigmoid(gate)

            x = filter
            x = F.dropout(x, self.dropout, training=self.training)
            # sc (b, c, n, t)
            skipx = self.skip_convs[i](x.permute(0, 3, 1, 2))
            skip = skip + skipx
            # gc
            x = self.stode_convs[i](x)
            # res
            x = x + resx
            # layer norm
            x = F.layer_norm(x, (x.shape[3], ))

        # finally skip (b, c, n, t)
        skip = self.skipE(x.permute(0, 3, 1, 2)) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        # x = self.ln(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = torch.squeeze(x).permute(0, 2, 1)
        return x
