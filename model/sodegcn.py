import sys
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.

#shared temporal weight
class joint_time(nn.Module):
    def __init__(self, temporal_dim):
        super(joint_time, self).__init__()
        if temporal_dim == 12:
            self.w3 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
            self.w4 = nn.Parameter(torch.randn(temporal_dim, temporal_dim))
        else:
            self.w3 = nn.Parameter(torch.randn(1, 1))
            self.w4 = nn.Parameter(torch.randn(1, 1))

    def forward(self, x, type):
        if type == 'node':
            att_left = torch.einsum('bntc, to -> bnoc', x, self.w3).transpose(1, 3)
            att_right = torch.einsum('bntc, tp -> bnpc', x, self.w4).transpose(1, 3)
            all_att = F.sigmoid(torch.matmul(att_left, att_right.transpose(2, 3)))
            return torch.matmul(all_att, x.transpose(1, 3)).transpose(1, 3)
        else:
            att_left = torch.einsum('bnmt, to -> bnmo', x, self.w3)
            att_right = torch.einsum('bnmt, tp -> bnmp', x, self.w4)
            all_att = F.sigmoid(torch.matmul(att_left.transpose(2, 3), att_right))
            return torch.matmul(x, all_att)

# first gnn
class ODEFunc(nn.Module):

    def __init__(self, time_func, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6

        # self.w = nn.Parameter(torch.eye(feature_dim))
        self.w = nn.Parameter(torch.ones((2 * feature_dim, feature_dim)))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.time = time_func
        self.fc = nn.Linear(feature_dim, 2*feature_dim)

    def forward(self, t, x):
        b, n, t, c = x.shape
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        ax = torch.einsum('ij, kjlm -> kilm', self.adj, x)
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        w = (1 + self.beta) * w - self.beta * torch.mm(torch.mm(w, torch.t(w)), w)
        xw = torch.einsum('ijkl, lm->ijkm', x, w)
        f = alpha * 0.5 * ax - x + xw - x + self.fc(self.x0.reshape(-1, self.x0.shape[3])).reshape(b, n, t, -1)
        return f


class ODEFunc_edge(nn.Module):

    def __init__(self, time_func, feature_dim, temporal_dim, adj):
        super(ODEFunc_edge, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.time = time_func
        self.fc = nn.Linear(12, 2*12)

    def forward(self, t, x):
        b, n, n, t = x.shape
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        ax = torch.einsum('ij, kjlm -> kilm', self.adj, x)
        f = alpha * 0.5 * ax - x + self.fc(self.x0.reshape(-1, self.x0.shape[3])).reshape(b, n, n, -1)
        return f




# 2-st order
class initial_velocity(nn.Module):

    def __init__(self, adj, in_channels, out_channels):
        super(initial_velocity, self).__init__()
        self.graph = adj
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()

    def reset(self):
        stdv = 1./ math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)
    def forward(self, x):
        ax = torch.einsum('ij, kjlm -> kilm', self.graph, x)
        axw = F.relu(torch.einsum('kjlm, mn -> kjln', ax, self.theta))
        return torch.cat((x, axw), dim=3)


#second gnn
class SODEFunc(nn.Module):

    def __init__(self, time_func, in_features, out_features, adj):
        super(SODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.time = time_func
        self.alpha_train = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.w = nn.Parameter(torch.eye(int(out_features)))
        self.d = nn.Parameter(torch.zeros(int(out_features)) + 1)

    def forward(self, t, x):
        cutoff = int(x.shape[3]/2)
        z = x[:, :, :, :cutoff]
        v = x[:, :, :, cutoff:]
        self.nfe += 1
        alpha = F.sigmoid(self.alpha_train).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        av = torch.einsum('ij, kjlm -> kilm', self.adj, v)
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        vw = torch.einsum('ijkl, lm -> ijkm', v, w)
        # f = alpha * 0.5 * (av - v) + self.time(v, 'node') - v + vw - v
        # second order gnn
        f = alpha * 0.5 * av - v + vw - v
        return torch.cat((v, f), dim=3)

# edge_sode
class SODEFunc_edge(nn.Module):

    def __init__(self, time_func, in_features, out_features, adj):
        super(SODEFunc_edge, self).__init__()
        self.adj = adj
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.time = time_func
        self.alpha_train = nn.Parameter(0.8 * torch.ones(adj.shape[1]))

    def forward(self, t, x):
        b, n, n, t = x.shape
        cutoff = int(x.shape[3]/2)
        z = x[:, :, :, :cutoff]
        v = x[:, :, :, cutoff:]
        self.nfe += 1
        alpha = F.sigmoid(self.alpha_train).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        av = torch.einsum('ij, kjlm -> kilm', self.adj, v)
        # f = alpha * 0.5 * (av - v) + self.time(v, 'edge') - v
        f = alpha * 0.5 * av - v
        return torch.cat((v, f), dim=3)


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1][:, :, :, :int(x.shape[3]/2)]
        return z

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
               + ")"


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=True)
        self._norm_fact = 1/math.sqrt(dim_k//num_heads)

    def forward(self, x):
        # x shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        # dim_k of each head
        dk = self.dim_k // nh
        # dim_v of each head
        dv = self.dim_v //nh

        # (batch, nh, n, dk)
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)
        # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)
        # (batch, nh, n, dv)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)

        # (batch, nh, n, n)
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        # (batch, nh, n, n)
        dist = torch.softmax(dist, dim=-1)

        # (batch, nh, n, dv)
        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


# Define the SODEGNN model.
class SGNN(nn.Module):
    def __init__(self, num_nodes, num_features, sp_channels, num_timesteps_input, num_timesteps_output, num_of_split, adj, time):
        super(SGNN, self).__init__()
        self.time_12 = joint_time(num_timesteps_input)
        self.time_1 = joint_time(num_timesteps_input/num_timesteps_input)
        self.num_nodes = num_nodes
        self.adj = adj
        self.T = time
        self.sp_channels = sp_channels
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input

        self.num_of_split = num_of_split
        self.len_of_split = int(num_timesteps_input/num_of_split)
        self.hidsize = num_timesteps_input * sp_channels

        self.len_4 = int(self.hidsize/num_of_split)

        self.encoder1 = nn.Linear(num_features, sp_channels)
        self.initalv_node = initial_velocity(adj, sp_channels, sp_channels)
        self.initalv_edge = initial_velocity(adj, num_timesteps_input, num_timesteps_input)
        #  24, 24 hidszie must be exact divison by num_of_split
        self.initalv_node_1 = initial_velocity(adj, self.len_4, self.len_4)
        # global message
        self.odeblock_1_node = ODEblock(SODEFunc(self.time_12, sp_channels, sp_channels, adj), t=torch.tensor([0, self.T]))
        # self.odeblock_1_node = ODEblock(ODEFunc(self.time_12, sp_channels, sp_channels, adj), t=torch.tensor([0, self.T]))
        self.odeblock_1_edge = ODEblock(SODEFunc_edge(self.time_12, sp_channels, sp_channels, adj), t=torch.tensor([0, self.T]))
        # self.odeblock_1_edge = ODEblock(ODEFunc_edge(self.time_12, sp_channels, sp_channels, adj), t=torch.tensor([0, self.T]))


        # local message
        for i in range(4):
            exec(
                f'self.odeblock_2_{i}_node = ODEblock(SODEFunc(self.time_1, '
                f'{int(num_timesteps_input*sp_channels/num_of_split)}, {int(num_timesteps_input*sp_channels/num_of_split)}, adj), t=torch.tensor([0,1,2,3]))'
            )
        # for i in range(4):
        #     exec(
        #         f'self.odeblock_2_{i}_node = ODEblock(ODEFunc(self.time_1, '
        #         f'{int(num_timesteps_input*sp_channels/num_of_split)}, {int(num_timesteps_input*sp_channels/num_of_split)}, adj), t=torch.tensor([0,1,2,3]))'
        #     )

        self.fcn_3_1 = nn.Linear(self.len_of_split, 1)
        self.fcn_12_4 = nn.Linear(num_timesteps_input, self.num_of_split)
        # attention
        self.att_12_4 = MultiHeadSelfAttention(num_timesteps_input * sp_channels, sp_channels * sp_channels, num_timesteps_input * sp_channels)
        self.fcn_4_1 = nn.Linear(4, 1)
        self.fcn_6_1 = nn.Linear(6, 1)
        self.fc_final1 = nn.Linear(sp_channels, num_features)
        self.fc_final2 = nn.Linear(sp_channels, num_features)
        self.fc_final3 = nn.Linear(num_features, num_features)
        self.clip = nn.Parameter(torch.randn(1, ))
        self.res_weight = nn.Parameter(torch.randn(1, ))
        self.edge_weight = nn.Parameter(torch.randn(1, ))
        self.weights = nn.Parameter(torch.randn(2, ))
        self.fc_edge = nn.Linear(num_nodes, num_features)


    def reset(self):
        self.encoder1.reset_parameters()
        self.encoder2.reset_parameters()

    def forward(self, x):
        # batch_size, num_nodes, timesteps, channels
        b, n, t, c = x.shape
        res = x

        # Generate dynamic edges
        # edge shape is (b, n, n, t)
        edge = torch.mean(x, dim=-1)
        edge = edge.repeat(1, n, 1).reshape(b, n, n, t) + edge.repeat(1, n, 1).reshape(b, n, n, t).transpose(1, 2)

        x_0 = x
        x_1 = x

        # Encode each node based on its feature.1
        x_0 = F.dropout(x_0, 0.3, training=self.training)
        x_0 = self.encoder1(x_0)
        x_1 = F.dropout(x_1, 0.3, training=self.training)
        x_1 = self.encoder1(x_1)

        # Global message passing
        self.odeblock_1_node.set_x0(x_0)
        x_0 = self.initalv_node(x_0)
        x_0 = self.odeblock_1_node(x_0)

        # Edge message passing
        self.odeblock_1_edge.set_x0(edge)
        edge = self.initalv_edge(edge)
        edge = self.odeblock_1_edge(edge)

        x_3_4 = []

        # Attention Module
        # x_1 shape is (b, n, t, c)
        x_1 = self.att_12_4(x_1.reshape(b, n, t * self.sp_channels)).reshape(b, n, self.num_of_split, int((t * self.sp_channels)/self.num_of_split))

        # Local message passing
        # split 12 into len 3 of 4 tensor
        for i in range(self.num_of_split):
            x = x_1[..., i, :].unsqueeze(-2)
            exec(f'self.odeblock_2_{i}_node.set_x0(x)')
            x = self.initalv_node_1(x)
            x = eval(f'self.odeblock_2_{i}_node(x)')
            x = x.transpose(0, 3).squeeze(0)
            x_3_4.append(x)

        x_1 = torch.stack(x_3_4, dim=-2).reshape(-1, self.num_nodes, t, self.sp_channels)

        # Message Filter
        x_1 = torch.where((x_0 + self.clip[0] - x_1) < 0, x_0 + self.clip[0], x_1)
        x_1 = torch.where((x_0 - self.clip[0] - x_1) > 0, x_0 - self.clip[0], x_1)
        x_g = self.fc_final1(x_1.reshape(-1, self.sp_channels)).reshape(-1, self.num_nodes, self.num_timesteps_input, self.num_features)
        x_l = self.fc_final2(x_0)
        x_e = self.fc_edge(edge.permute(0, 2, 3, 1))

        # Aggregation Layer
        # out_global = x_g * torch.sigmoid(x_l) + x_g * torch.sigmoid(x_e) + \
        #              x_l * torch.sigmoid(x_e) + x_l * torch.sigmoid(x_g) + \
        #              x_e * torch.sigmoid(x_l) + x_e * torch.sigmoid(x_g)
        # z = out_global/6
        out_global = x_g + x_l + x_e
        z = out_global/3




        # Update Layer
        z = self.weights[0] * F.sigmoid(self.fc_final3(res)) + self.weights[1] * z
        z = F.relu(z)
        return z