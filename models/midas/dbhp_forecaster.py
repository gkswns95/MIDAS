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


class DBHPForecaster(nn.Module):
    def __init__(self, params: dict):
        super().__init__()

        assert params["missing_pattern"] == "forecast"

        self.params = params

        self.dataset = params["dataset"]
        self.n_features = params["n_features"]  # number of features per player
        self.team_size = params["team_size"]  # number of players per team
        self.n_players = self.team_size if self.dataset == "afootball" else self.team_size * 2

        self.pe_z_dim = params["pe_z_dim"]
        self.pi_z_dim = params["pi_z_dim"]
        self.rnn_dim = params["rnn_dim"]

        n_layers = params["n_layers"]
        dropout = params["dropout"] if "dropout" in params else 0

        dp_rnn_in_dim = self.n_features
        if params["ppe"]:
            self.ppe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
            dp_rnn_in_dim += self.pe_z_dim
        if params["fpe"]:
            self.fpe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
            dp_rnn_in_dim += self.pe_z_dim
        if params["fpi"]:
            self.fpi_st = SetTransformer(self.n_features, self.pi_z_dim, embed_type="i")
            dp_rnn_in_dim += self.pi_z_dim

        self.dp_rnn = nn.LSTM(
            input_size=dp_rnn_in_dim,
            hidden_size=self.rnn_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False,
        )
        self.dp_fc = nn.Linear(self.rnn_dim, self.n_features)

        if self.params["dynamic_hybrid"]:
            self.temp_decay = TemporalDecay(input_size=1, output_size=1, diag=False)
            if params["fpe"] and params["fpi"]:
                hybrid_rnn_in_dim = self.pe_z_dim + self.pi_z_dim + 9
            elif params["fpe"]:
                hybrid_rnn_in_dim = self.pe_z_dim + 9
            else:
                hybrid_rnn_in_dim = self.pi_z_dim + 9

            self.hybrid_rnn = nn.LSTM(
                input_size=hybrid_rnn_in_dim,
                hidden_size=params["hybrid_rnn_dim"],
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=False,
            )
            self.hybrid_fc = nn.Sequential(
                nn.Linear(params["hybrid_rnn_dim"], 16),
                nn.ReLU(),
                nn.Linear(16, 2),
                nn.Softmax(dim=-1),
            )

    def dynamic_hybrid_pred(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        dp_pos = data["pos_pred"].permute(2, 0, 1, 3)  # [time, bs, players, 2]
        dp_vel = data["vel_pred"].permute(2, 0, 1, 3)
        dp_accel = data["accel_pred"].permute(2, 0, 1, 3)
        dap_f = data["dap_f"].permute(2, 0, 1, 3)
        preds = torch.cat([dp_pos, dp_vel, dp_accel, dap_f], dim=-1)  # [time, bs, players, 8]

        if self.params["fpi"]:  # FPE + FPI
            z = torch.cat([self.fpe_z, self.fpi_z], -1).reshape(self.seq_len, self.bs, self.n_players, -1)
        else:  # FPE only
            z = self.fpe_z.reshape(self.seq_len, self.bs, self.n_players, -1)  # [time, bs, players, z]
        gamma_f = self.temp_decay(data["deltas_f"].unsqueeze(-1)).transpose(0, 1)  # [time, bs, players, 1]

        rnn_in = torch.cat([preds, z, gamma_f], dim=-1).flatten(1, 2)  # [time, bs * players, z+9]
        out, _ = self.hybrid_rnn(rnn_in)  # [time, bs * players, hrnn]
        lambdas = self.hybrid_fc(out).unsqueeze(-1)  # [time, bs * players, 2, 1]

        dp_pos_ = dp_pos.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]
        dap_f_ = dap_f.flatten(1, 2).unsqueeze(2)  # [time, bs * players, 1, 2]

        preds_pos = torch.cat([dp_pos_, dap_f_], dim=2)  # [time, bs * players, 2, 2]
        hybrid_pos = torch.sum(lambdas * preds_pos, dim=2)  # [time, bs * players, 2]

        mask = data["pos_mask"]  # [bs, players, time, 2]
        target_pos = data["pos_target"]
        hybrid_pos = hybrid_pos.reshape(self.seq_len, self.bs, self.n_players, 2).permute(1, 2, 0, 3)
        hybrid_pos = mask * target_pos + (1 - mask) * hybrid_pos  # [bs, players, time, 2]

        lambdas = lambdas.reshape(self.seq_len, self.bs, self.n_players, 2).permute(1, 2, 0, 3)

        return hybrid_pos, lambdas  # [bs, players, time, 2], [bs, players, time, 2]

    def forward(self, ret: Dict[str, torch.Tensor], device="cuda:0") -> Dict[str, torch.Tensor]:
        total_loss = 0.0
        if not self.params["transformer"]:
            self.dp_rnn.flatten_parameters()

        if self.params["cuda"]:
            input = ret["input"].to(device)
            target = ret["target"].to(device)
            mask = ret["mask"].to(device)

        input = input.transpose(0, 1)  # [bs, time, -1] to [time, bs, -1]
        self.seq_len, self.bs = input.shape[:2]

        assert self.params["ppe"] or self.params["fpe"] or self.params["fpi"]

        team1_x = input[..., : self.n_features * self.team_size].reshape(-1, self.team_size, self.n_features)
        if self.dataset in ["soccer", "basketball"]:
            team2_x = input[..., self.n_features * self.team_size :].reshape(-1, self.team_size, self.n_features)
            x = torch.cat([team1_x, team2_x], 1)  # [time * bs, players, feats]
        else:
            x = team1_x  # [time * bs, players, feats]

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
        out = self.dp_rnn(rnn_in)[0]  # [time, bs * players, rnn]
        out = self.dp_fc(out).reshape(self.seq_len, self.bs, -1).transpose(0, 1)  # [bs, time, x]

        pred = mask * target + (1 - mask) * out  # [bs, time, x], STRNN-DP

        feature_types = ["pos", "vel", "cartesian_accel"][: self.n_features // 2]
        for mode in feature_types:
            pred_ = reshape_tensor(pred, mode=mode, sports=self.dataset, single_team=self.params["single_team"]).transpose(1, 2)  # [bs, players, time, 2]
            target_ = reshape_tensor(target, mode=mode, sports=self.dataset, single_team=self.params["single_team"]).transpose(1, 2)
            mask_ = reshape_tensor(mask, mode=mode, sports=self.dataset, single_team=self.params["single_team"]).transpose(1, 2)
            loss = torch.abs((pred_ - target_) * (1 - mask_)).sum() / (1 - mask_).sum()
            total_loss += loss

            mode = mode.split("_")[-1]  # cartesian_accel to accel
            ret[f"{mode}_pred"] = pred_
            ret[f"{mode}_target"] = target_
            ret[f"{mode}_mask"] = mask_
            ret[f"{mode}_loss"] = loss

        if self.params["deriv_accum"]:
            # DAP-F with output sizes [bs, players, time, 2]
            use_accel = self.params["cartesian_accel"]
            ret["dap_f"] = deriv_accum_pred(ret, use_accel=use_accel, fb="f", dataset=self.dataset)
            ret["dap_f_loss"] = torch.abs(ret["dap_f"] - ret["pos_target"]).sum() / (1 - ret["pos_mask"]).sum()
            total_loss += ret["dap_f_loss"]

            if self.params["dynamic_hybrid"]:
                hybrid_pos, lambdas = self.dynamic_hybrid_pred(ret)
                ret["hybrid_d"] = hybrid_pos  # [bs, players, time, 2] (STRNN-DBHP-D)
                ret["lambdas"] = lambdas  # [bs, players, time, 2]

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
            ret["hybrid_d"] = ret["hybrid_d"].transpose(1, 2).flatten(2, 3)
            ret["lambdas"] = ret["lambdas"].transpose(1, 2).flatten(2, 3)  # [bs, time, players * 2]

        ret["pred"] = pred
        ret["total_loss"] = total_loss

        return ret
