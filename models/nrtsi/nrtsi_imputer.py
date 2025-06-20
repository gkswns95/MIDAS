""" Define the Transformer model """

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from set_transformer.model import SetTransformer


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5, attn_dropout=0.0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        return x


class Adapter(nn.Module):
    """Adapter Network"""

    def __init__(self, d_model):
        super(Adapter, self).__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, d_model // 8)
        self.fc2 = nn.Linear(d_model // 8, d_model)
        self.weight_init()

    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight, 1e-4)
        nn.init.xavier_uniform_(self.fc2.weight, 1e-4)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x += residual
        return x


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        super(EncoderLayer, self).__init__()

        self.dropout = 0.0
        self.use_layer_norm = 0
        self.adapter = 0

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=self.dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=self.dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        if self.adapter:
            multi_res_adapter_1 = dict()
            multi_res_adapter_2 = dict()
            i = 1
            while i < 256:
                multi_res_adapter_1[str(int(math.log(i + 1, 2)))] = Adapter(d_model)
                multi_res_adapter_2[str(int(math.log(i + 1, 2)))] = Adapter(d_model)
                i *= 2
            self.multi_res_adapter_1 = nn.ModuleDict(multi_res_adapter_1)
            self.multi_res_adapter_2 = nn.ModuleDict(multi_res_adapter_2)

    def forward(self, enc_input, gap, slf_attn_mask=None):
        residual = enc_input
        if self.use_layer_norm:
            enc_input = self.layer_norm_1(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if self.adapter:
            enc_output = self.multi_res_adapter_1[str(math.floor(math.log(gap + 1, 2)))](enc_output)
        enc_output += residual
        residual = enc_output
        if self.use_layer_norm:
            enc_output = self.layer_norm_2(enc_output)
        enc_output = self.pos_ffn(enc_output)
        if self.adapter:
            enc_output = self.multi_res_adapter_2[str(math.floor(math.log(gap + 1, 2)))](enc_output)
        enc_output += residual
        return enc_output, enc_slf_attn


class TimeEncoding(nn.Module):
    """time encoding from paper Set Functions for Time Series"""

    def __init__(self, max_time_scale, time_enc_dim):
        super(TimeEncoding, self).__init__()
        self.max_time = max_time_scale
        self.n_dim = time_enc_dim
        self._num_timescales = self.n_dim // 2

    def get_timescales(self):
        timescales = self.max_time ** np.linspace(0, 1, self._num_timescales)
        return timescales[None, None, :]

    def forward(self, times):
        """times has shape [bs, T, 1]"""
        timescales = torch.tensor(self.get_timescales()).to(times.device)
        scaled_time = times.float() / timescales

        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return signal


class MercerTimeEncoding(nn.Module):
    """Self-attention with Functional Time Representation Learning"""

    def __init__(self, time_dim, expand_dim):
        super().__init__()
        self.time_dim = time_dim
        self.expand_dim = expand_dim
        self.init_period_base = nn.Parameter(torch.linspace(0, 8, time_dim))
        self.basis_expan_var = torch.empty(time_dim, 2 * expand_dim)
        nn.init.xavier_uniform_(self.basis_expan_var)
        self.basis_expan_var = nn.Parameter(self.basis_expan_var)
        self.basis_expan_var_bias = nn.Parameter(torch.zeros([time_dim]))

    def forward(self, t):
        """t has shape [batch size, seq_len, 1]"""
        expand_t = t.repeat(1, 1, self.time_dim)
        period_var = 10**self.init_period_base
        period_var = period_var[:, None].repeat(1, self.expand_dim)  # [time_dim, expand_dim]
        expand_coef = torch.range(1, self.expand_dim)[None, :].float().cuda()  # [1, expand_dim]
        freq_var = 1 / period_var
        freq_var = freq_var * expand_coef
        sin_enc = torch.sin(expand_t[:, :, :, None] * freq_var[None, None, :, :])
        cos_enc = torch.cos(expand_t[:, :, :, None] * freq_var[None, None, :, :])
        time_enc = torch.cat([sin_enc, cos_enc], dim=-1) * self.basis_expan_var[None, None, :, :]
        time_enc = time_enc.sum(-1) + self.basis_expan_var_bias[None, None, :]
        return time_enc


class NRTSIImputer(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(self, params):
        super().__init__()

        self.params = params
        self.n_features = params["n_features"]
        self.team_size = params["team_size"]

        if params["single_team"]:
            self.n_players = params["team_size"]
        else:
            self.n_players = params["team_size"] * 2

        if params["missing_pattern"] == "playerwise":
            self.x_dim = params["n_features"]
            self.y_dim = params["n_features"]
        else:
            self.x_dim = self.n_players * params["n_features"]

            self.y_dim = self.x_dim * 2 if params["stochastic"] else self.x_dim

        self.max_time_scale = params["n_max_time_scale"]
        self.time_enc_dim = params["time_enc_dim"]
        self.time_dim = params["time_dim"]
        self.expand_dim = params["expand_dim"]
        self.n_head = params["n_heads"]
        self.n_layers = params["n_layers"]
        self.d_model = params["model_dim"]
        self.d_inner = params["inner_dim"]
        self.d_k = params["att_dim"]
        self.d_v = params["att_dim"]

        self.use_gap_encoding = 0
        self.use_layer_norm = 0
        self.adapter = 0
        self.confidence = 0
        self.mercer = 0
        self.use_mask = 1

        if not self.mercer:
            self.position_enc = TimeEncoding(self.max_time_scale, self.time_enc_dim)
            td = self.time_enc_dim
        else:
            self.position_enc = MercerTimeEncoding(time_dim=self.time_dim, expand_dim=self.expand_dim)
            td = self.time_dim
        self.dropout = nn.Dropout(p=0.0)

        self.layer_stack = nn.ModuleList(
            [EncoderLayer(self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v) for _ in range(self.n_layers)]
        )
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.output_proj = nn.Linear(self.d_model, self.y_dim)

        if self.use_gap_encoding:
            self.input_proj = nn.Linear(self.x_dim + 2 * td, self.d_model)
        else:
            self.input_proj = nn.Linear(self.x_dim + td + 1, self.d_model)

        # if params["ppe"] or params["fpe"] or params["fpi"]:  # use set transformer for PI/PE embedding
        #     self.pe_z_dim = params["pe_z_dim"]
        #     self.pi_z_dim = params["pi_z_dim"]

        #     in_dim = self.n_features
        #     if params["ppe"]:
        #         self.ppe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
        #         in_dim += self.pe_z_dim
        #     if params["fpe"]:
        #         self.fpe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
        #         in_dim += self.pe_z_dim
        #     if params["fpi"]:
        #         self.fpi_st = SetTransformer(self.n_features, self.pi_z_dim, embed_type="i")
        #         in_dim += self.pi_z_dim

        #     # in_dim *= self.n_players

        #     self.set_transformer_fc = nn.Linear(in_dim, self.n_features)


    def forward(self, obs_data, obs_time, imput_time, gap, src_mask=None, return_attns=False):
        """
        obs_data has shape [bs, obs_len, x_dim]
        obs_time has shape [bs, obs_len, 1]
        imput_time has shape [bs, imput_len, 1]
        gap is a scalar
        """

        device = obs_data.device

        num_obs = obs_data.shape[1]
        num_imp = imput_time.shape[1]
        if self.use_mask:
            mask = (
                torch.cat(
                    [
                        torch.cat([torch.ones(num_obs, num_obs), torch.ones(num_obs, num_imp)], dim=1),
                        torch.cat([torch.ones(num_imp, num_obs), torch.eye(num_imp)], dim=1),
                    ],
                    dim=0,
                )
                .unsqueeze(0)
                .to(device)
            )
        else:
            mask = None

        # Add Set Transformer embeddings in obs
        # if self.params["ppe"] or self.params["fpe"] or self.params["fpi"]:
        #     obs = self.set_transformer_emb(obs_data)

        if self.use_gap_encoding:
            obs_time_encoding = self.position_enc(obs_time).float()
            obs = torch.cat([obs_data, obs_time_encoding, torch.zeros_like(obs_time_encoding).float()], dim=-1)
            missing_data = torch.zeros(size=(imput_time.shape[0], imput_time.shape[1], obs_data.shape[-1])).to(device)
            gap_embedding = (
                torch.tensor([gap])[None, None, :].repeat(imput_time.shape[0], imput_time.shape[1], 1).to(device)
            )
            imput = torch.cat(
                [missing_data.float(), self.position_enc(imput_time).float(), self.position_enc(gap_embedding).float()],
                dim=-1,
            )
        else:
            obs = torch.cat([obs_data, self.position_enc(obs_time).float(), torch.ones_like(obs_time).float()], dim=-1)
            missing_data = torch.zeros(size=(imput_time.shape[0], imput_time.shape[1], obs_data.shape[-1])).to(device)
            imput = torch.cat(
                [missing_data.float(), self.position_enc(imput_time).float(), torch.zeros_like(imput_time).float()],
                dim=-1,
            )

        combined = torch.cat([obs, imput], dim=1) # [bs, time, *]


        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.input_proj(combined)  # [bs, time, d_model]
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, gap, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        # print(enc_output.shape) # [bs, time, *]

        if self.use_layer_norm:
            enc_output = self.layer_norm(enc_output)
        output = self.output_proj(enc_output[:, num_obs:, :])
        if return_attns:
            return output, enc_slf_attn_list
        return output

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

        input_list = [x]
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
        x_ = self.set_transformer_fc(x_) # [time * bs, players, -1]
        x_ = x_.reshape(seq_len, bs, -1).transpose(0, 1) # [bs, time, x_dim]

        return x_ 