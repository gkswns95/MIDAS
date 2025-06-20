import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from models.utils import reshape_tensor
from typing import List

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self, params):
        super(RITS, self).__init__()
        self.build(params)

    def build(self, params):
        self.params = params

        self.n_features = params["n_features"]
        self.dataset = params["dataset"]

        self.rnn_dim = params["rnn_dim"]

        # if params["single_team"]:
        #     self.n_players = params["team_size"]
        # else:
        #     self.n_players = params["team_size"] * 2
        self.n_players = params["team_size"] * 2
        
        self.x_dim = self.n_features * self.n_players

        self.rnn_cell = nn.LSTMCell(self.x_dim * 2, self.rnn_dim)

        self.temp_decay_h = TemporalDecay(input_size=self.x_dim, output_size=self.rnn_dim, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.x_dim, output_size=self.x_dim, diag=True)

        self.hist_reg = nn.Linear(self.rnn_dim, self.x_dim)
        self.feat_reg = FeatureRegression(self.x_dim)

        self.weight_combine = nn.Linear(self.x_dim * 2, self.x_dim)

    def forward(self, ret: Dict[str, torch.Tensor], device="cuda:0") -> Dict[str, torch.Tensor] :
        input = ret["input"]
        target = ret["target"]
        mask = ret["mask"]
        delta = ret["delta"]

        # device = input.device
        bs, seq_len = input.shape[:2]

        input = input.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) # [bs, time, players * feats]
        target = target.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) 
        mask = mask.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)
        delta = delta.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)

        h = Variable(torch.zeros((bs, self.rnn_dim))).to(device)
        c = Variable(torch.zeros((bs, self.rnn_dim))).to(device)

        total_loss = 0.0
        pred = torch.zeros(input.shape).to(device)
        x_h_ = torch.zeros(input.shape).to(device)
        z_h_ = torch.zeros(input.shape).to(device)
        c_h_ = torch.zeros(input.shape).to(device)

        for t in range(seq_len):
            x_t = input[:, t, :]  # [bs, x_dim]
            # y_t = target[:, t, :]
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]

            gamma_h = self.temp_decay_h(d_t)
            gamma_x = self.temp_decay_x(d_t)

            h = h * gamma_h

            x_h = self.hist_reg(h)  # [bs, x_dim]

            x_c = m_t * x_t + (1 - m_t) * x_h

            z_h = self.feat_reg(x_c)

            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))

            c_h = beta * z_h + (1 - beta) * x_h

            c_c = m_t * x_t + (1 - m_t) * c_h

            inputs = torch.cat([c_c, m_t], dim=1)

            h, c = self.rnn_cell(inputs, (h, c))

            pred[:, t, :] = c_c
            x_h_[:, t, :] = x_h
            z_h_[:, t, :] = z_h
            c_h_[:, t, :] = c_h

        output_list = [x_h_, z_h_, c_h_]
        target_xy = reshape_tensor(target, sports=self.dataset, single_team=self.params["single_team"])
        mask_xy = reshape_tensor(mask, sports=self.dataset, single_team=self.params["single_team"])
        for out in output_list:
            pred_xy = reshape_tensor(out, sports=self.dataset, single_team=self.params["single_team"])  # [bs, total_players, -1]
            total_loss += torch.sum(torch.abs(pred_xy - target_xy) * (1 - mask_xy)) / torch.sum((1 - mask_xy))

        ret.update({
            "loss": total_loss, 
            "pred": pred, 
            "target": target,
            "input" : input,
            "mask" : mask})
        
        return ret