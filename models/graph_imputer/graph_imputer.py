import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

EPS = torch.finfo(torch.float).eps  # numerical logs


class LSTM(nn.Module):
    def __init__(self, n_players, n_features, hid, device):
        super().__init__()
        self.n_features = n_features
        self.n_players = n_players
        self.hid = hid
        self.device = device

        self.rnn1 = nn.LSTMCell(self.n_features, self.hid)
        self.rnn2 = nn.LSTMCell(self.hid, self.hid)

    def init_states(self, bs):
        self.h1 = torch.zeros(bs * self.n_players * 2, self.hid).to(self.device)
        self.c1 = torch.zeros(bs * self.n_players * 2, self.hid).to(self.device)
        self.h2 = torch.zeros(bs * self.n_players * 2, self.hid).to(self.device)
        self.c2 = torch.zeros(bs * self.n_players * 2, self.hid).to(self.device)

    def forward(self, x):
        bs, _ = x.shape  # [batch size, features]

        h1, c1 = self.rnn1(x, (self.h1, self.c1))
        h2, c2 = self.rnn2(self.h1, (self.h2, self.c2))

        self.h1, self.c1 = h1.detach(), c1.detach()
        self.h2, self.c2 = h2.detach(), c2.detach()
        # return self.h2
        return h2


class EdgeModel(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim * 2, hid)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(hid, hid)

    # def forward(self, src, dest, edge_attr, u, batch):
    def forward(self, x):
        # out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.mlp1(x)
        out = self.relu(out)
        out = self.mlp2(out)
        return out


class NodeModel(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.mlp1 = nn.Linear(in_dim, hid)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(hid, hid)

    # def forward(self, src, dest, edge_attr, u, batch):
    def forward(self, x):
        # out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.mlp1(x)
        out = self.relu(out)
        out = self.mlp2(out)
        return out


class GraphNets(nn.Module):
    def __init__(self, n_players, n_features, in_dim, hid, device):
        super().__init__()
        self.n_features = n_features
        self.n_players = n_players
        self.in_dim = in_dim
        self.hid = hid
        self.device = device

        self.edgemodel = EdgeModel(self.in_dim, self.hid)
        self.nodemodel = NodeModel(self.hid, self.hid)

    def forward(self, x):
        bs, _, _ = x.shape
        node_i = x.repeat_interleave(self.n_players * 2, dim=1)  # [b, (num_player * 2) ** 2, hid]
        node_j = node_i.clone()
        edge_feats = torch.cat([node_i, node_j], dim=2)  # [b, (num_player * 2) ** 2, hid * 2]
        edge_out = self.edgemodel(edge_feats)

        # Aggregation
        edge_agg = torch.zeros(bs, self.n_players * 2, self.hid).to(self.device)
        for i in range(self.n_players * 2):
            edge_agg[:, i, :] = torch.sum(edge_out[:, i :: self.n_players * 2, :], dim=1)

        node_out = self.nodemodel(edge_agg)
        return node_out


class GraphImputer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.n_players = params["n_players"]
        self.n_features = params["n_features"]
        self.hid = params["rnn_dim"]
        self.var_dim = params["var_dim"]
        self.kld_weight = params["kld_weight_float"]
        self.bs = params["batch_size"]
        self.device = params["device"]

        # LSTM
        self.lstm = LSTM(self.n_players, self.n_features, self.hid, self.device)

        # Encoder
        self.gn_enc = GraphNets(self.n_players, self.n_features, self.n_features + self.hid, self.hid, self.device)
        self.gn_enc_mean = nn.Linear(self.hid, self.var_dim)
        self.gn_enc_std = nn.Sequential(nn.Linear(self.hid, self.var_dim), nn.Softplus())

        # Prior
        self.gn_prior = GraphNets(self.n_players, self.n_features, self.hid, self.hid, self.device)
        self.gn_prior_mean = nn.Linear(self.hid, self.var_dim)
        self.gn_prior_std = nn.Sequential(nn.Linear(self.hid, self.var_dim), nn.Softplus())

        self.phi_z = nn.Sequential(nn.Linear(self.var_dim, self.var_dim), nn.ReLU())

        # Decoder
        self.gn_dec = GraphNets(self.n_players, self.n_features, self.var_dim + self.hid, self.hid, self.device)
        self.gn_dec_mean = nn.Linear(self.hid, self.var_dim)
        self.gn_dec_std = nn.Sequential(nn.Linear(self.hid, self.var_dim), nn.Softplus())

        self.final_layer = nn.Linear(self.hid, self.n_features)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)

    def _calculate_loss(self, recons, input, enc_mu, enc_std, prior_mu, prior_std):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = (
            2 * torch.log(prior_std + EPS)
            - 2 * torch.log(enc_std + EPS)
            + (enc_std.pow(2) + (enc_mu - prior_mu).pow(2)) / prior_std.pow(2)
            - 1
        )
        kld_loss = 0.5 * torch.sum(kld_loss)
        return recons_loss + self.kld_weight * kld_loss

    def forward(self, x, masked_x, mask):
        bs, seq_len, feat_dim = x.shape  # [batch size, time steps, features]

        # h_list = []
        outputs_diff = []
        outputs = []
        for i in range(seq_len):
            if i == 0:
                inputs = x[:, i, :]
                self.lstm.init_states(bs)
            else:
                inputs = outputs[-1]
            inputs = inputs.reshape(bs * self.n_players * 2, self.n_features)  # [16, 22, 6]
            node_feats = self.lstm(inputs)
            node_feats = node_feats.reshape(bs, self.n_players * 2, self.hid)  # [16, 22, 64]

            # if i == 0:
            #     h = torch.zeros_like(node_feats)
            # else:
            #     h = h_list[-1]

            # enc_input = torch.concat([inputs.view(bs, self.n_players * 2, self.n_features), h], dim = 2) #[16, 22, 70]
            enc_input = torch.concat(
                [inputs.view(bs, self.n_players * 2, self.n_features), node_feats], dim=2
            )  # [16, 22, 70]
            # Encoder
            enc_out = self.gn_enc(enc_input)  # [batch_size, num_player * 2, hid]
            enc_out_mean = self.gn_enc_mean(enc_out)  # [batch_size, num_player * 2, var_dim]
            enc_out_std = self.gn_enc_std(enc_out)

            # Prior
            # prior_out = self.gn_prior(h) #[batch_size, num_player * 2, hid]
            prior_out = self.gn_prior(node_feats)  # [batch_size, num_player * 2, hid]
            prior_out_mean = self.gn_prior_mean(prior_out)  # [batch_size, num_player * 2, var_dim]
            prior_out_std = self.gn_prior_std(prior_out)

            z = self._reparameterized_sample(enc_out_mean, enc_out_std)  # [batch_size, num_player * 2, var_dim]
            z_out = self.phi_z(z)

            # Decoder
            # dec_out = self.gn_dec(torch.cat([z_out, h], dim = 2)) #[batch_size, num_player * 2, hid]
            dec_out = self.gn_dec(torch.cat([z_out, node_feats], dim=2))  # [batch_size, num_player * 2, hid]
            dec_out_mean = self.gn_dec_mean(dec_out)  # [batch_size, num_player * 2, hid]
            dec_out_std = self.gn_dec_std(dec_out)

            out = self.final_layer(dec_out)  # delta_x_t [batch_size, num_player * 2, n_features]

            out = out.view(bs, self.n_players * 2 * self.n_features)
            zero_mask = mask[:, i, :] == 0
            if i == 0:
                prev_values = x[:, 0, :]
            else:
                # prev_values = x[:, i-1, :]
                prev_values = outputs[-1]

            replace_out = torch.where(zero_mask, out + prev_values, x[:, i, :])

            # Update
            # h_list.append(node_feats)
            outputs_diff.append(out)
            outputs.append(replace_out)

        # h_list = torch.stack(h_list) #[time steps, batch size, n_players * 2, hid]
        outputs_diff = torch.stack(outputs_diff).transpose(
            0, 1
        )  # [time_steps, batch_size, n_players * 2, n_features] to [batch_size, time_steps, n_players * 2, n_features]
        outputs = torch.stack(outputs).transpose(
            0, 1
        )  # [time_steps, batch_size, n_players * 2, n_features] to [batch_size, time_steps, n_players * 2, n_features]

        input_diff = torch.concat([torch.zeros(bs, 1, feat_dim).to(self.device), torch.diff(x, dim=1)], dim=1)
        # input_diff = torch.concat([torch.zeros(bs, 1, feat_dim).to(self.device), torch.diff(x_hat, dim = 1)], dim = 1)

        # loss = self._calculate_loss(outputs, input_diff, enc_out_mean, enc_out_std, prior_out_mean, prior_out_std)
        loss = self._calculate_loss(outputs_diff, input_diff, enc_out_mean, enc_out_std, prior_out_mean, prior_out_std)

        # outputs = outputs + torch.concat([x[:, :1, :], x[:, :seq_len-1, :]], dim = 1) #delta_x_t + x_t-1

        return outputs, loss

    def sample(self, x, masked_x, mask):
        bs, seq_len, feat_dim = x.shape  # [batch size, time steps, features]

        # h_list = []
        outputs_diff = []
        outputs = []
        for i in range(seq_len):
            if i == 0:
                inputs = x[:, i, :]
                self.lstm.init_states(bs)
            else:
                inputs = outputs[-1]
                # inputs = masked_x[:, i, :]

            inputs = inputs.reshape(bs * self.n_players * 2, self.n_features)
            node_feats = self.lstm(inputs)
            node_feats = node_feats.reshape(bs, self.n_players * 2, self.hid)

            # if i == 0:
            #     h = torch.zeros_like(node_feats)
            # else:
            #     h = h_list[-1]

            # Prior
            # prior_out = self.gn_prior(h) #[batch_size, num_player * 2, hid]
            prior_out = self.gn_prior(node_feats)  # [batch_size, num_player * 2, hid]
            prior_out_mean = self.gn_prior_mean(prior_out)  # [batch_size, num_player * 2, var_dim]
            prior_out_std = self.gn_prior_std(prior_out)

            z = self._reparameterized_sample(prior_out_mean, prior_out_std)  # [batch_size, num_player * 2, var_dim]
            z_out = self.phi_z(z)

            # Decoder
            # dec_out = self.gn_dec(torch.cat([z_out, h], dim = 2)) #[batch_size, num_player * 2, hid]
            dec_out = self.gn_dec(torch.cat([z_out, node_feats], dim=2))  # [batch_size, num_player * 2, hid]
            dec_out_mean = self.gn_dec_mean(dec_out)  # [batch_size, num_player * 2, hid]
            dec_out_std = self.gn_dec_std(dec_out)

            out = self.final_layer(dec_out)  # delta_x_t
            # outputs.append(out)
            out = out.view(bs, self.n_players * 2 * self.n_features)
            zero_mask = mask[:, i, :] == 0
            if i == 0:
                prev_values = x[:, 0, :]
            else:
                # prev_values = x[:, i-1, :]
                prev_values = outputs[-1]
            replace_out = torch.where(zero_mask, out + prev_values, x[:, i, :])

            # Update
            # h_list.append(node_feats)
            outputs_diff.append(out)
            outputs.append(replace_out)

        # h_list = torch.stack(h_list) #[time steps, batch size, n_players * 2, hid]
        outputs_diff = torch.stack(outputs_diff).transpose(
            0, 1
        )  # [time_steps, batch_size, n_players * 2, n_features] to [batch_size, time_steps, n_players * 2, n_features]
        outputs = torch.stack(outputs).transpose(
            0, 1
        )  # [time_steps, batch_size, n_players * 2, n_features] to [batch_size, time_steps, n_players * 2, n_features]
        # outputs = outputs.view(bs, seq_len, feat_dim) #[batch_size, time_steps, feat_dim]

        input_diff = torch.cat([torch.zeros(bs, 1, feat_dim).to(self.device), torch.diff(x, dim=1)], dim=1)
        # input_diff = torch.concat([torch.zeros(bs, 1, feat_dim).to(self.device), torch.diff(x, dim = 1)], dim = 1)
        # loss = F.mse_loss(outputs_diff, input_diff)
        loss = F.mse_loss(outputs, x)

        return outputs, loss


class BidirectionalGraphImputer(nn.Module):
    # def __init__(self, n_players, n_features, hid, var_dim, missing_mode, dataset, kld_weight, device):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "n_players",
            "pe_z_dim",
            "pi_z_dim",
            "rnn_dim",
            "hybrid_rnn_dim",
            "n_layers",
            "n_heads",
            "dropout",
            "pred_xy",
            "cartesian_accel",
            "transformer",
            "ppe",
            "fpe",
            "fpi",
            "bidirectional",
            "dynamic_missing",
            "upper_bound",
            "stochastic",
            "seperate_learning",
            "avg_length_loss",
            "var_dim",
            "kld_weight_float",
            "weighted",
            "missing_prob_float",
            "m_pattern",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)
        self.device = self.params["device"]

        # if params["target_type"] == "imputation":
        #     self.model_type = "regressor"
        # else:
        #     self.model_type = "classifier"

        self.missing_mode = "block" if self.params["dynamic_missing"] else "block_all_feat"
        if self.params["dynamic_missing"]:
            if self.params["m_pattern"] == "camera":
                self.missing_mode = "camera"
            else:
                self.missing_mode = "playerwise"
        else:
            self.missing_mode = "uniform"

        self.dataset = params["dataset"]
        self.n_features = params["n_features"]
        self.weighted = params["weighted"]
        self.missing_prob = params["missing_prob_float"]
        print(f"Missing mode : {self.missing_mode} | {self.missing_prob}")
        print(f"KLD Weight : {params['kld_weight_float']}")
        self.forward_gi = GraphImputer(self.params)
        self.backward_gi = GraphImputer(self.params)

    def forward(self, data, device="cuda:0"):
        if len(data) == 2:  # Soccer
            player_data, ball_data = data
        else:
            player_data = data
            ball_data = []
        data_dict = {"target": player_data, "ball": ball_data}
        ret = dict()
        bs, seq_len, feat_dim = player_data.shape
        player_data = player_data.to(self.device)

        # masking
        mask = generate_mask(
            data=data_dict,
            sports=self.params["dataset"],
            mode=self.missing_mode,
            missing_rate=random.randint(1, 9) * 0.1,
        )  # [bs, time, players]

        # import pickle
        # with open(f'mask_{self.missing_mode}.pkl', 'wb') as f:
        #     pickle.dump(mask, f)
        # print('mask saved')

        if self.missing_mode == "camera":
            time_gap = time_interval(mask, list(range(seq_len)), mode="camera")
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, n_players]
            time_gap = torch.tensor(time_gap, dtype=torch.float32)
            mask = torch.repeat_interleave(mask, self.n_features, dim=-1).squeeze(0)  # [bs, time, x_dim]
            time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).squeeze(0)

        else:
            time_gap = time_interval(mask, list(range(seq_len)), mode="block")
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, n_players]
            time_gap = torch.tensor(time_gap, dtype=torch.float32).unsqueeze(0)
            mask = torch.repeat_interleave(mask, self.n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]
            time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).expand(
                bs, -1, -1
            )  # [bs, time, x_dim]

        mask = mask.to(self.device)
        masked_x = player_data * mask

        reversed_x = torch.flip(player_data, dims=[1])
        reversed_masked_x = torch.flip(masked_x, dims=[1])
        reversed_mask = torch.flip(mask, dims=[1])

        forward_out, forward_loss = self.forward_gi(player_data, masked_x, mask)
        backward_out, backward_loss = self.backward_gi(reversed_x, reversed_masked_x, reversed_mask)
        backward_out = torch.flip(backward_out, dims=[1])

        if self.weighted:
            weights = torch.arange(1, seq_len + 1) / seq_len
            weights = weights.unsqueeze(0).unsqueeze(2).to(self.device)
            reversed_weights = torch.flip(weights, dims=[1]).to(self.device)
            out = forward_out * weights + backward_out * reversed_weights
        else:
            out = 0.5 * forward_out + 0.5 * backward_out
        loss = forward_loss + backward_loss

        pred_t = reshape_tensor(out, upscale=True, dataset_type=self.dataset)  # [bs, total_players, 2]
        target_t = reshape_tensor(player_data, upscale=True, dataset_type=self.dataset)

        # pos_pe = torch.norm(pred_t - target_t, dim=-1).sum()

        ret["target"] = player_data
        ret["pred"] = out
        ret["mask"] = mask
        ret["total_loss"] = loss
        ret["pred_pe"] = calc_pos_error(
            ret["pred"], ret["target"], ret["mask"], n_features=self.n_features, aggfunc="mean", dataset=self.dataset
        )
        ret["pred_pe_sum"] = calc_pos_error(
            ret["pred"], ret["target"], ret["mask"], n_features=self.n_features, aggfunc="sum", dataset=self.dataset
        )
        # ret['pos_pe'] = pos_pe.item()

        return ret

    @torch.no_grad()
    def evaluate(self, data, device="cuda:0"):
        if len(data) == 2:  # Soccer
            player_data, ball_data = data
        else:
            player_data = data
            ball_data = []

        data_dict = {"target": player_data, "ball": ball_data}
        ret = dict()
        bs, seq_len, feat_dim = player_data.shape
        player_data = player_data.to(self.device)
        if self.missing_prob == "random":
            missing_probs = np.arange(10) * 0.1
            missing_rate = missing_probs[random.randint(1, 9)]
        else:
            missing_rate = self.missing_prob

        missing_probs = np.arange(10) * 0.1
        mask = mask = generate_mask(
            data=data_dict,
            sports=self.params["dataset"],
            mode=self.missing_mode,
            missing_rate=missing_rate,
        )  # [bs, time, players]

        if self.missing_mode == "camera":
            time_gap = time_interval(mask, list(range(seq_len)), mode="camera")
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, n_players]
            time_gap = torch.tensor(time_gap, dtype=torch.float32)
            mask = torch.repeat_interleave(mask, self.n_features, dim=-1).squeeze(0)  # [bs, time, x_dim]
            time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).squeeze(0)

        else:
            time_gap = time_interval(mask, list(range(seq_len)), mode="block")
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, time, n_players]
            time_gap = torch.tensor(time_gap, dtype=torch.float32).unsqueeze(0)
            mask = torch.repeat_interleave(mask, self.n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]
            time_gap = torch.repeat_interleave(time_gap, self.n_features, dim=-1).expand(
                bs, -1, -1
            )  # [bs, time, x_dim]

        mask = mask.to(self.device)
        masked_x = player_data * mask

        reversed_x = torch.flip(player_data, dims=[1])
        reversed_masked_x = torch.flip(masked_x, dims=[1])
        reversed_mask = torch.flip(mask, dims=[1])

        forward_out, forward_loss = self.forward_gi.sample(player_data, masked_x, mask)
        backward_out, backward_loss = self.backward_gi.sample(reversed_x, reversed_masked_x, reversed_mask)
        backward_out = torch.flip(backward_out, dims=[1])
        # out = 0.5 * forward_out + 0.5 * backward_out #Fusion하는 부분 weight 다르게 수정
        if self.weighted:
            weights = torch.arange(1, seq_len + 1) / seq_len
            weights = weights.unsqueeze(0).unsqueeze(2).to(self.device)
            reversed_weights = torch.flip(weights, dims=[1]).to(self.device)
            out = forward_out * weights + backward_out * reversed_weights
        else:
            out = 0.5 * forward_out + 0.5 * backward_out

        loss = forward_loss + backward_loss

        pred_t = reshape_tensor(out, upscale=True, dataset_type=self.dataset)  # [bs, total_players, 2]
        target_t = reshape_tensor(player_data, upscale=True, dataset_type=self.dataset)

        # pos_pe = torch.norm(pred_t - target_t, dim=-1).sum()

        ret["target"] = player_data
        ret["pred"] = out
        ret["mask"] = mask
        ret["total_loss"] = loss
        ret["pred_pe"] = calc_pos_error(
            ret["pred"], ret["target"], ret["mask"], n_features=self.n_features, aggfunc="sum", dataset=self.dataset
        )
        ret["pred_pe_sum"] = calc_pos_error(
            ret["pred"], ret["target"], ret["mask"], n_features=self.n_features, aggfunc="sum", dataset=self.dataset
        )
        # ret['pos_pe'] = pos_pe.item()
        return ret
