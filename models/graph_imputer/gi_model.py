import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

EPS = torch.finfo(torch.float).eps  # numerical logs

class EdgeModel(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)
            )

    def forward(self, x):
        return self.model(x)
    
class NodeModel(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim)    
        )
    def forward(self, x):
        return self.model(x)

class GraphNets(nn.Module):
    def __init__(self, in_dim, h_dim, std_dim):
        super().__init__()
        
        self.h_dim = h_dim
        self.std_dim = std_dim

        # Build edge and node model
        self.edge_model = EdgeModel(in_dim * 2, h_dim=h_dim)
        self.node_model = NodeModel(h_dim, h_dim=h_dim)

        self.mean_fc = nn.Linear(self.h_dim, self.std_dim)
        self.std_fc = nn.Sequential(nn.Linear(self.h_dim, self.std_dim), nn.Softplus())

    def forward(self, x):
        '''
        x : [bs, players, *]
        '''
        bs, n_players, _ = x.shape

        node_i = x.repeat_interleave(n_players, dim=1)  # [bs, players ** 2, *]
        node_j = node_i.clone()
        edge_feats = torch.cat([node_i, node_j], dim=-1)
        edge_out = self.edge_model(edge_feats)

        # Aggregation
        edge_agg = torch.zeros(bs, n_players, self.h_dim).to(x.device)
        for p in range(n_players):
            edge_agg[:, p, :] = torch.sum(edge_out[:, p :: n_players, :], dim=1)

        node_out = self.node_model(edge_agg)

        mean = self.mean_fc(node_out)
        std = self.std_fc(node_out)

        return mean, std

class GImodel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.team_size = params["team_size"]
        self.n_features = params["n_features"]

        if params["single_team"]:
            self.n_players = self.team_size
        else:
            self.n_players = self.team_size * 2
        
        self.rnn_dim = params["rnn_dim"]
        self.std_dim = params["std_dim"]
        self.kld_weight = params["kld_weight"]

        # LSTM
        self.rnn = nn.LSTM(self.n_features, self.rnn_dim, num_layers=2, batch_first=True)

        # Encoder
        enc_in_dim = self.rnn_dim + self.n_features
        self.gn_enc = GraphNets(in_dim=enc_in_dim, h_dim=self.rnn_dim, std_dim=self.std_dim)
    
        # Prior
        prior_in_dim = self.rnn_dim
        self.gn_prior = GraphNets(in_dim=prior_in_dim, h_dim=self.rnn_dim, std_dim=self.std_dim)
        self.phi_z = nn.Sequential(nn.Linear(self.std_dim, self.std_dim), nn.ReLU())

        # Decoder
        dec_in_dim = self.rnn_dim + self.std_dim
        self.gn_dec = GraphNets(in_dim=dec_in_dim, h_dim=self.rnn_dim, std_dim=self.std_dim)
        
        self.dec_out_fc = nn.Linear(self.std_dim, self.n_features)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=mean.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)

    def _calculate_loss(self, recons, input, enc_mu, enc_std, prior_mu, prior_std):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = (2 * torch.log(prior_std + EPS) - 2 * torch.log(enc_std + EPS)
                    + (enc_std.pow(2) + (enc_mu - prior_mu).pow(2)) / prior_std.pow(2) - 1)
        kld_loss = 0.5 * torch.sum(kld_loss)
        return recons_loss + self.kld_weight * kld_loss

    def forward(self, ret: Dict[str, torch.Tensor], mode="train", device="cuda:0") -> Dict[str, torch.Tensor]:    
        input = ret["input"] # [bs, time, x_dim]
        target = ret["target"]
        mask = ret["mask"]

        bs, seq_len = input.shape[:2]

        input = input.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) # [bs, time, players * feat_dim(=x_dim)]
        target = target.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)
        mask = mask.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)

        output_list = []
        delta_x_list = []
        
        h = torch.zeros(2, bs * self.n_players, self.rnn_dim).to(device)
        c = torch.zeros(2, bs * self.n_players, self.rnn_dim).to(device)

        for i in range(seq_len):
            if i == 0:
                input_t_prev = torch.zeros_like(input[:, 0, :])
            else:
                input_t_prev = output_list[-1]

            # Player-wise LSTM
            rnn_input = input_t_prev.reshape(bs * self.n_players, 1, -1)  # [bs * players, 1, feat_dim]
            rnn_out, (h, c) = self.rnn(rnn_input, (h, c))
            # rnn_out, (h, c) = self.rnn(rnn_input, (h, c))[:, -1, :]
            node_feats = rnn_out[: ,-1, :].reshape(bs, self.n_players, -1) # [bs, players, h_dim]
            
            # Encoder
            if mode == "train":
                target_t = target[:, i, :]
                input_t_reshaped = target_t.reshape(bs, self.n_players, -1)
                enc_input = torch.cat([input_t_reshaped, node_feats], dim=2)  # [bs, players, h_dim + node_feat_dim]
                enc_mean, enc_std = self.gn_enc(enc_input)  # [bs, players, h_dim]
            # Prior
            pri_mean, pri_std = self.gn_prior(node_feats)  # [bs, players, h_dim]
            
            if mode == "train":
                z_t = self._reparameterized_sample(enc_mean, enc_std)  # [bs, players, std_dim]
            else:
                z_t = self._reparameterized_sample(pri_mean, pri_std)  # [bs, players, std_dim]

            z_t = self.phi_z(z_t)

            # Decoder
            dec_input_t = torch.cat([z_t, node_feats], dim=2)
            dec_mean, dec_std = self.gn_dec(dec_input_t)  # [bs, players, h_dim]
            
            dec_out = self._reparameterized_sample(dec_mean, dec_std)  # [bs, players, std_dim]
            delta_x_t = self.dec_out_fc(dec_out).reshape(bs, -1)
            if i == 0:
                prev_input = input[:, 0, :]
            else:
                prev_input = output_list[-1]
            
            # Replace the missing values at time t with the predicted values.
            replaced_out = torch.where((mask[:, i, :] == 0), prev_input + delta_x_t, input[:, i, :])

            output_list.append(replaced_out)
            delta_x_list.append(delta_x_t)

        delta_x = torch.stack(delta_x_list).transpose(0, 1)  # [bs, time, players, feat_dim]
        outputs = torch.stack(output_list).transpose(0, 1)  # [bs, time, players, feat_dim]

        input_delta = torch.cat([torch.zeros(bs, 1, input.shape[-1], device=device), torch.diff(input, dim=1)], dim=1)

        ret["pred"] = outputs
        if mode == "train":
            ret["loss"] = self._calculate_loss(
                delta_x, 
                input_delta, 
                enc_mean,
                enc_std, 
                pri_mean, 
                pri_std)

        return ret