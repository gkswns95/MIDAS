import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from models.utils import reshape_tensor
from typing import List

from set_transformer.model import SetTransformer

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


class STRITS(nn.Module):
    def __init__(self, params):
        super(STRITS, self).__init__()
        self.build(params)

    def build(self, params):
        self.params = params

        self.n_features = params["n_features"]
        self.dataset = params["dataset"]

        self.rnn_dim = params["rnn_dim"]
        self.team_size = params["team_size"]
    
        if params["single_team"]:
            self.n_players = params["team_size"]
        else:
            self.n_players = params["team_size"] * 2
        
        self.x_dim = self.n_features * self.n_players

        if self.params["ppe"] or self.params["fpe"] or self.params["fpi"]:
            self.temp_decay_h = TemporalDecay(input_size=self.n_features, output_size=self.rnn_dim, diag=False)
            self.temp_decay_x = TemporalDecay(input_size=self.n_features, output_size=self.n_features, diag=True)
        
            self.hist_reg = nn.Linear(self.rnn_dim, self.n_features)
            self.feat_reg = FeatureRegression(self.n_features)
            self.weight_combine = nn.Linear(self.n_features * 2, self.n_features)
        
            self.pe_z_dim = params["pe_z_dim"]
            self.pi_z_dim = params["pi_z_dim"]

            in_dim = self.n_features
            if params["ppe"]:
                self.ppe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                in_dim += self.pe_z_dim
            if params["fpe"]:
                self.fpe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                in_dim += self.pe_z_dim
            if params["fpi"]:
                self.fpi_st = SetTransformer(self.n_features, self.pi_z_dim, embed_type="i")
                in_dim += self.pi_z_dim
            
            self.rnn_cell = nn.LSTMCell(in_dim, self.rnn_dim)

    def forward(self, ret: Dict[str, torch.Tensor], device="cuda:0") -> Dict[str, torch.Tensor] :
        input = ret["input"]
        target = ret["target"]
        mask = ret["mask"]
        delta = ret["delta"]

        # device = input.device
        bs, seq_len = input.shape[:2]

        input = input.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) # [bs, time, players * feats (=x_dim)]
        target = target.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) 
        mask = mask.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)
        delta = delta.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)

        h = Variable(torch.zeros((bs * self.n_players, self.rnn_dim))).to(device)
        c = Variable(torch.zeros((bs * self.n_players, self.rnn_dim))).to(device)

        pred = torch.zeros(input.shape).to(device)
        x_h_ = torch.zeros(input.shape).to(device)
        z_h_ = torch.zeros(input.shape).to(device)
        c_h_ = torch.zeros(input.shape).to(device)

        st_embs = self.set_transformer_emb(input) # [bs, time, player, st_emb_dim]

        total_loss = 0.0
        for t in range(seq_len):
            x_t = input[:, t, :]  # [bs, x_dim]
            m_t = mask[:, t, :]
            d_t = delta[:, t, :]
            st_embs_t = st_embs[:, t, :, :] # [bs, players, st_emb_dim]

            x_t = x_t.reshape(bs * self.n_players, -1) # [bs * players, feats]
            m_t = m_t.reshape(bs * self.n_players, -1)
            d_t = d_t.reshape(bs * self.n_players, -1)
            st_embs_t = st_embs_t.reshape(bs * self.n_players, -1) # [bs * players, st_emb_dim]

            gamma_h = self.temp_decay_h(d_t) # [bs * players, rnn_dim]
            gamma_x = self.temp_decay_x(d_t) # [bs * players, feats]

            h = h * gamma_h # [bs * players, rnn_dim]

            x_h = self.hist_reg(h)  # [bs * players, feats]

            x_c = m_t * x_t + (1 - m_t) * x_h

            z_h = self.feat_reg(x_c) # [bs * players, feats]

            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1)) # [bs * players, feats]

            c_h = beta * z_h + (1 - beta) * x_h

            c_c = m_t * x_t + (1 - m_t) * c_h # [bs * players, feats]
            # inputs = self.set_transformer_emb(c_c.reshape(bs, self.n_players, -1)) # [bs * players, emb_dim]
            inputs = torch.cat([c_c, st_embs_t], dim=-1)
            h, c = self.rnn_cell(inputs, (h, c)) # [bs * players, rnn_dim]

            pred[:, t, :] = c_c.reshape(bs, -1)
            x_h_[:, t, :] = x_h.reshape(bs, -1)
            z_h_[:, t, :] = z_h.reshape(bs, -1)
            c_h_[:, t, :] = c_h.reshape(bs, -1)

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
    
    def set_transformer_emb(self, x):
        '''
        x : [bs, time, x_dim]
        '''

        bs, seq_len = x.shape[:2]

        x = x.transpose(0, 1) # [bs, time, x_dim] to [time, bs, x_dim]

        team1_x = x[..., : self.n_features * self.team_size].reshape(-1, self.team_size, self.n_features)
        if self.params["single_team"]:
            x = team1_x  # [time * bs, players, feats]
        else:
            team2_x = x[..., self.n_features * self.team_size :].reshape(-1, self.team_size, self.n_features)
            x = torch.cat([team1_x, team2_x], 1)  # [time * bs, players, feats]

        input_list = []
        if self.params["ppe"]:
            team1_z = self.ppe_st(team1_x)
            if self.params["single_team"]:
                self.ppe_z = team1_z # [time * bs, players, pe_z]
            else:
                team2_z = self.ppe_st(team2_x) # [time * bs, team_size, pe_z]
                self.ppe_z = torch.cat([team1_z, team2_z], dim=1)  # [time * bs, players, pe_z]
            input_list += [self.ppe_z]
        if self.params["fpe"]:
            self.fpe_z = self.fpe_st(x)  # [time * bs, players, pe_z]
            input_list += [self.fpe_z]
        if self.params["fpi"]:
            self.fpi_z = self.fpi_st(x).unsqueeze(1).expand(-1, self.n_players, -1) # [time * bs, players, pi_z]
            input_list += [self.fpi_z]
        
        x_ = torch.cat(input_list, -1) # [time * bs, players, emb_dim]
        # x_ = self.set_transformer_fc(x_) # [time * bs, players, -1]
        x_ = x_.reshape(seq_len, bs, self.n_players, -1).transpose(0, 1) # [bs, time, players, emb_dim]

        return x_