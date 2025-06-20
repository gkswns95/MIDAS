import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter

from models.midas.utils import deriv_accum_pred
from models.utils import calc_coherence_loss, reshape_tensor
from set_transformer.model import SetTransformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


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


class MIDASImputer(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        self.params = params

        self.dataset = params["dataset"]
        self.n_features = params["n_features"]  # number of features per player
        self.team_size = params["team_size"]  # number of players per team
        if params["single_team"]:
            self.n_players = self.team_size
        else:
            self.n_players = self.team_size * 2

        self.pe_z_dim = params["pe_z_dim"]
        self.pi_z_dim = params["pi_z_dim"]
        self.rnn_dim = params["rnn_dim"]

        n_heads = params["n_heads"]
        n_layers = params["n_layers"]
        dropout = params["dropout"] if "dropout" in params else 0

        if params["ppe"] or params["fpe"] or params["fpi"]:  # use set transformer for PI/PE embedding
            ip_rnn_in_dim = self.n_features
            if params["ppe"]:
                self.ppe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                ip_rnn_in_dim += self.pe_z_dim
            if params["fpe"]:
                self.fpe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                ip_rnn_in_dim += self.pe_z_dim
            if params["fpi"]:
                self.fpi_st = SetTransformer(self.n_features, self.pi_z_dim, embed_type="i")
                ip_rnn_in_dim += self.pi_z_dim

            if params["transformer"]:
                # self.pos_encoder = PositionalEncoding(self.rnn_dim, dropout)
                # trans_encoder_layer = TransformerEncoderLayer(self.rnn_dim, n_heads, self.rnn_dim * 2, dropout)
                self.transformer_in_fc = nn.Linear(ip_rnn_in_dim, self.rnn_dim)
                self.pos_encoder = PositionalEncoding(self.rnn_dim, dropout)
                trans_encoder_layer = TransformerEncoderLayer(self.rnn_dim, n_heads, self.rnn_dim * 2, dropout)
                self.transformer_encoder = TransformerEncoder(trans_encoder_layer, n_layers)
                self.transformer_out_fc = nn.Linear(self.rnn_dim, self.rnn_dim * 2)
            else:
                self.ip_rnn = nn.LSTM(
                    input_size=ip_rnn_in_dim,
                    hidden_size=self.rnn_dim,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=params["bidirectional"],
                )
            ip_rnn_out_dim = self.rnn_dim * 2 if params["bidirectional"] else self.rnn_dim
            ip_out_dim = 6 if params["cartesian_accel"] else 4
            self.ip_fc = nn.Linear(ip_rnn_out_dim, ip_out_dim)

        else:  # use random or naive player ordering
            assert not params["transformer"]

            ip_rnn_out_dim = self.rnn_dim * 2 if params["bidirectional"] else self.rnn_dim
            ip_out_dim = 6 if params["cartesian_accel"] else 4

            if self.params["uniagent"]:
                self.ip_in_fc = nn.Sequential(nn.Linear(self.n_features, self.pe_z_dim), nn.ReLU())
            else:
                self.ip_in_fc = nn.Sequential(nn.Linear(self.n_players * self.n_features, self.pi_z_dim), nn.ReLU())
                ip_out_dim *= self.n_players

            self.ip_rnn = nn.LSTM(
                input_size=self.pi_z_dim,
                hidden_size=self.rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=params["bidirectional"],
            )
            self.ip_fc = nn.Linear(ip_rnn_out_dim, ip_out_dim)

        if self.params["dynamic_hybrid"]:
            self.temp_decay = TemporalDecay(input_size=1, output_size=1, diag=False)
            self.hybrid_rnn_dim = params["hybrid_rnn_dim"]

            if params["fpe"] and params["fpi"]:
                hybrid_rnn_in_dim = ip_out_dim + self.pe_z_dim + self.pi_z_dim + 6
            elif params["fpe"] or params["uniagent"]:
                hybrid_rnn_in_dim = ip_out_dim + self.pe_z_dim + 6
            else:
                hybrid_rnn_in_dim = 6 + self.pi_z_dim + 6

            self.hybrid_rnn = nn.LSTM(
                input_size=hybrid_rnn_in_dim,
                hidden_size=self.hybrid_rnn_dim,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=True,
            )
            hybrid_rnn_out_dim = self.hybrid_rnn_dim * 2 if params["bidirectional"] else self.hybrid_rnn_dim
            self.hybrid_fc = nn.Sequential(
                nn.Linear(hybrid_rnn_out_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Softmax(dim=-1),
            )

    def dynamic_hybrid_pred(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        ip_pos = data["pos_pred"].permute(2, 0, 1, 3)  # [time, bs, players, 2]
        ip_vel = data["vel_pred"].permute(2, 0, 1, 3)
        dap_f = data["dap_f"].permute(2, 0, 1, 3)
        dap_b = data["dap_b"].permute(2, 0, 1, 3)

        if self.params["cartesian_accel"]:
            ip_accel = data["accel_pred"].permute(2, 0, 1, 3)
            preds = torch.cat([ip_pos, ip_vel, ip_accel, dap_f, dap_b], dim=-1)  # [time, bs, players, 10]
        else:
            preds = torch.cat([ip_pos, ip_vel, dap_f, dap_b], dim=-1)  # [time, bs, players, 8]

        if self.params["fpe"] and self.params["fpi"]:  # FPE + FPI
            z = torch.cat([self.fpe_z, self.fpi_z], -1).reshape(self.seq_len, self.bs, self.n_players, -1)
        elif self.params["fpe"]:  # FPE only
            z = self.fpe_z.reshape(self.seq_len, self.bs, self.n_players, -1)  # [time, bs, players, z]
        elif self.params["fpi"]:  # FPI only
            z = self.fpi_z.reshape(self.seq_len, self.bs, self.n_players, -1)  # [time, bs, players, z]
        elif self.params["uniagent"]:
            z = self.z.reshape(self.seq_len, self.bs, self.n_players, -1)  # [time, bs, players, pe_z]
        else:
            z = self.z.unsqueeze(2).expand(-1, -1, self.n_players, -1)  # [time, bs, players, pi_z]

        gamma_f = self.temp_decay(data["deltas_f"].unsqueeze(-1)).transpose(0, 1)  # [time, bs, players, 1]
        gamma_b = self.temp_decay(data["deltas_b"].unsqueeze(-1)).transpose(0, 1)  # [time, bs, players, 1]

        rnn_in = torch.cat([preds, z, gamma_f, gamma_b], dim=-1).flatten(1, 2)  # [time, bs * players, -1]
        out, _ = self.hybrid_rnn(rnn_in)  # [time, bs * players, hrnn * 2]
        lambdas = self.hybrid_fc(out).unsqueeze(-1)  # [time, bs * players, 3, 1]

        ip_pos_ = ip_pos.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]
        dap_f_ = dap_f.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]
        dap_b_ = dap_b.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]

        preds_pos = torch.cat([ip_pos_, dap_f_, dap_b_], dim=2)  # [time, bs * players, 3, 2]
        hybrid_pos = torch.sum(lambdas * preds_pos, dim=2)  # [time, bs * players, 2]

        mask = data["pos_mask"]  # [bs, players, time, 2]
        target_pos = data["pos_target"]
        hybrid_pos = hybrid_pos.reshape(self.seq_len, self.bs, self.n_players, 2).permute(1, 2, 0, 3)
        hybrid_pos = mask * target_pos + (1 - mask) * hybrid_pos  # [bs, players, time, 2]

        lambdas = lambdas.reshape(self.seq_len, self.bs, self.n_players, 3).permute(1, 2, 0, 3)

        return hybrid_pos, lambdas  # [bs, players, time, 2], [bs, players, time, 3]

    def forward(self, ret: Dict[str, torch.Tensor], device="cuda:0") -> Dict[str, torch.Tensor]:
        total_loss = 0

        if not self.params["transformer"]:
            self.ip_rnn.flatten_parameters()
            if self.params["dynamic_hybrid"]:
                self.hybrid_rnn.flatten_parameters()

        if self.params["cuda"]:
            input = ret["input"].to(device)
            target = ret["target"].to(device)
            mask = ret["mask"].to(device)

        self.bs, self.seq_len = input.shape[:2]
        input = input.reshape(self.bs, self.seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)
        input = input.transpose(0, 1)  # [bs, time, players * 6] to [time, bs, players * feats]

        if self.params["ppe"] or self.params["fpe"] or self.params["fpi"]:
            team1_x = input[..., : self.n_features * self.team_size].reshape(-1, self.team_size, self.n_features)
            if self.params["single_team"]:
                 x = team1_x  # [time * bs, players, feats]
            else:
                team2_x = input[..., self.n_features * self.team_size :].reshape(-1, self.team_size, self.n_features)
                x = torch.cat([team1_x, team2_x], 1)  # [time * bs, players, feats]
            
            rnn_in = [x]
            if self.params["ppe"]:
                team1_z = self.ppe_st(team1_x)  # [time * bs, team_size, pe_z]
                if self.dataset in ["soccer", "basketball"]:
                    team2_z = self.ppe_st(team2_x)  # [time * bs, team_size, pe_z]
                    self.ppe_z = torch.cat([team1_z, team2_z], dim=1)  # [time * bs, players, pe_z]
                else:
                    self.ppe_z = team1_z  # [time * bs, players, pe_z]
                rnn_in += [self.ppe_z]
            if self.params["fpe"]:
                self.fpe_z = self.fpe_st(x)  # [time * bs, players, pe_z]
                rnn_in += [self.fpe_z]
            if self.params["fpi"]:
                self.fpi_z = self.fpi_st(x).unsqueeze(1).expand(-1, self.n_players, -1)  # [time * bs, players, pi_z]
                rnn_in += [self.fpi_z]

            rnn_in = torch.cat(rnn_in, -1).reshape(self.seq_len, self.bs * self.n_players, -1)

            if self.params["transformer"]:
                rnn_in = self.transformer_in_fc(rnn_in) # [time, bs, * players, rnn]
                rnn_in = self.pos_encoder(rnn_in)
                out = self.transformer_encoder(rnn_in)  
                out = self.transformer_out_fc(out) # [time, bs * players, rnn * 2]
            else:
                out = self.ip_rnn(rnn_in)[0]  # [time, bs * players, rnn * 2]

            out = self.ip_fc(out).reshape(self.seq_len, self.bs, -1).transpose(0, 1)  # [bs, time, players * ip_out]

        elif self.params["uniagent"]:
            input = input.reshape(self.seq_len, self.bs * self.n_players, -1)  # [time, bs * players, feats]
            self.z = self.ip_in_fc(input)  # [time, bs * players, pe_z]
            out = self.ip_rnn(self.z)[0]  # [time, bs * players, rnn * 2]
            out = self.ip_fc(out).reshape(self.seq_len, self.bs, -1).transpose(0, 1)  # [bs, time, players * ip_out]

        else:
            self.z = self.ip_in_fc(input)  # [time, bs, pi_z]
            out = self.ip_rnn(self.z)[0]  # [time, bs, rnn * 2]
            out = self.ip_fc(out).transpose(0, 1)  # [bs, time, players * ip_out]

        ip_out_types = ["pos", "vel"]
        if self.params["cartesian_accel"]:
            ip_out_types += ["cartesian_accel"]

        ip_out_list = []

        for mode in ip_out_types:
            # [bs, time, players * ip_out] to [bs, players, time, 2]
            pred_xy = reshape_tensor(out, mode=mode, sports=self.dataset, single_team=self.params["single_team"]).transpose(1, 2)
            target_xy = reshape_tensor(target, mode=mode, sports=self.dataset, single_team=self.params["single_team"]).transpose(1, 2)
            mask_xy = reshape_tensor(mask, mode=mode, sports=self.dataset, single_team=self.params["single_team"]).transpose(1, 2)

            pred_xy = mask_xy * target_xy + (1 - mask_xy) * pred_xy  # [bs, players, time, 2]
            loss = torch.abs((pred_xy - target_xy) * (1 - mask_xy)).sum() / (1 - mask_xy).sum()
            total_loss += loss

            mode = mode.split("_")[-1]  # cartesian_accel to accel
            ret[f"{mode}_pred"] = pred_xy
            ret[f"{mode}_target"] = target_xy
            ret[f"{mode}_mask"] = mask_xy
            ret[f"{mode}_loss"] = loss

            ip_out_list.append(pred_xy)

        if self.params["deriv_accum"]:
            # DAP-F and DAP-B with output sizes [bs, players, time, 2]
            use_accel = self.params["cartesian_accel"]
            ret["dap_f"] = deriv_accum_pred(ret, use_accel=use_accel, single_team=self.params["single_team"], fb="f", dataset=self.dataset)
            ret["dap_b"] = deriv_accum_pred(ret, use_accel=use_accel, single_team=self.params["single_team"], fb="b", dataset=self.dataset)
            
            ret["dap_f_loss"] = torch.abs(ret["dap_f"] - ret["pos_target"]).sum() / (1 - ret["pos_mask"]).sum()
            ret["dap_b_loss"] = torch.abs(ret["dap_b"] - ret["pos_target"]).sum() / (1 - ret["pos_mask"]).sum()

            total_loss += ret["dap_f_loss"] + ret["dap_b_loss"]
            # total_loss += ret["dap_b_loss"]

            if self.params["dynamic_hybrid"]:
                hybrid_pos, lambdas = self.dynamic_hybrid_pred(ret)
                ret["hybrid_d"] = hybrid_pos  # [bs, players, time, 2] (STRNN-DBHP-D)
                ret["lambdas"] = lambdas  # [bs, players, time, 3]

                sum_loss = torch.abs((ret["hybrid_d"] - ret["pos_target"]) * (1 - ret["pos_mask"])).sum()
                ret["hybrid_d_loss"] = sum_loss / (1 - ret["pos_mask"]).sum()
                total_loss += ret["hybrid_d_loss"]

            if self.params["coherence_loss"]:
                p_d = ret["pos_pred"]  # [bs, time, players, 2]
                v_d = ret["vel_pred"]
                a_d = ret["cartesian_accel_pred"]
                m = ret["pos_mask"][..., 0].unsqueeze(-1)  # [bs, time, players, 1]
                ret["coherence_loss"] = calc_coherence_loss(p_d, v_d, a_d, m, add_va=True)
                total_loss += ret["coherence_loss"]

            # Reshape Predictions
            ret["dap_f"] = ret["dap_f"].transpose(1, 2).flatten(2, 3)  # [bs, time, x]
            ret["dap_b"] = ret["dap_b"].transpose(1, 2).flatten(2, 3)
            if self.params["dynamic_hybrid"]:
                ret["hybrid_d"] = ret["hybrid_d"].transpose(1, 2).flatten(2, 3)
                ret["lambdas"] = ret["lambdas"].transpose(1, 2).flatten(2, 3)  # [bs, time, players * 3]

        ret["pred"] = torch.cat(ip_out_list, dim=-1).transpose(1, 2).flatten(2, 3)  # [bs, time, players, ip_out]
        ret["total_loss"] = total_loss

        return ret
