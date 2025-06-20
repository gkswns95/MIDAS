import copy

import torch.nn as nn
import torch

from typing import Dict
from einops import rearrange, repeat
from .Attention_layers import AttentionLayer, SelfAttentionLayer, EmbeddedAttention

# from tsl.nn import utils
# from tsl.nn.blocks.encoders import MLP

class EmbeddedAttentionLayer(nn.Module):
    """
    Spatial embedded attention layer
    """
    def __init__(self,
                 model_dim, adaptive_embedding_dim, feed_forward_dim=2048, dropout=0):
        super().__init__()

        self.attn = EmbeddedAttention(model_dim, adaptive_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        # emb: (..., length, model_dim)
        residual = x
        out = self.attn(x, emb)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer
    """
    def __init__(self, seq_len, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super(ProjectedAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

        self.seq_len = seq_len

    def forward(self, x):
        # x: [b s n d]
        batch = x.shape[0]
        projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
                              repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]

        message_out = self.out_attn(projector, x, x)  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        message_in = self.in_attn(x, projector, message_out)  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class ImputeFormerModel(nn.Module):
    """
    Spatiotempoarl Imputation Transformer
    """
    def __init__(self, params):
        super(ImputeFormerModel, self).__init__()

        self.params = params

        self.n_features = params["n_features"]
        self.dataset = params["dataset"]

        if params["single_team"]:
            self.n_players = params["team_size"]
        else:
            self.n_players = params["team_size"] * 2
        
        self.x_dim = self.n_features * self.n_players

        self.num_nodes = params["n_nodes"]
        self.in_steps = params["window_size"]
        self.out_steps = params["window_size"]
        self.input_dim = params["n_features"]
        self.output_dim = params["n_features"]
        self.input_embedding_dim = params["input_embedding_dim"]
        self.learnable_embedding_dim = params["learnable_embedding_dim"]
        self.model_dim = (
                self.input_embedding_dim
                + self.learnable_embedding_dim)
        self.num_temporal_heads = params["n_temporal_heads"]
        self.num_layers = params["n_layers"]

        self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
        self.dim_proj = params["proj_dim"]

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(params["window_size"], self.num_nodes, self.learnable_embedding_dim))) # [time, players, emb_dim]

        # self.readout = MLP(self.model_dim, self.model_dim, self.output_dim, n_layers=2)

        self.readout = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.ReLU(),
            nn.Linear(self.model_dim, self.output_dim))

        self.attn_layers_t = nn.ModuleList(
            [ProjectedAttentionLayer(self.num_nodes, self.dim_proj, self.model_dim, self.num_temporal_heads, self.model_dim, params["dropout"])
             for _ in range(self.num_layers)])

        self.attn_layers_s = nn.ModuleList(
            [EmbeddedAttentionLayer(self.model_dim, self.learnable_embedding_dim, params["feed_forward_dim"])
                for _ in range(self.num_layers)])


    def forward(self, ret: Dict[str, torch.Tensor], device="cuda:0") -> Dict[str, torch.Tensor] :
        x = ret["input"] # [bs, time, x_dim]
        m = ret["mask"]
        y = ret["target"]

        bs, seq_len = x.shape[:2]

        x = x.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) # [bs, time, players * feats]
        m = m.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)
        y = y.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)

        x = x.reshape(bs, seq_len, self.n_players, -1) # [bs, time, players, feats]
        m = m.reshape(bs, seq_len, self.n_players, -1)
        y = y.reshape(bs, seq_len, self.n_players, -1)

        original_x = copy.deepcopy(x)

        x = self.input_proj(x)  # [bs, time, players, emb_dim]

        ### debug ###
        # self.learnable_embedding = nn.init.xavier_uniform_(
        #     nn.Parameter(torch.empty(seq_len, self.num_nodes, self.learnable_embedding_dim, device=device))) # [time, players, emb_dim]

        node_emb = self.learnable_embedding.expand(bs, *self.learnable_embedding.shape)

        x = torch.cat([x, node_emb], dim=-1) # [bs, time, players, model_dim (emb_dim + learnable_emb_dim)]

        x = x.permute(0, 2, 1, 3)  # [bs, players, time, -1]
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)

        x = x.permute(0, 2, 1, 3)
        out = self.readout(x) # [bs, time, players, feats]

        # Compute losses
        pred = m * y + (1 - m) * out  # [bs, time, players, 2]
        loss = torch.abs((pred - y) * (1 - m)).sum() / (1 - m).sum()
        loss_f = self.Freg(pred, y, m)
        ret["total_loss"] = loss + self.params["f1_loss_float"] * loss_f

        ret.update({
            "pred" : pred.flatten(2, 3),
            "target": y.flatten(2, 3),
            "input" : original_x.flatten(2, 3),
            "mask" : m.flatten(2, 3)})

        return ret
    
    def Freg(self, y_hat, y, mask):
        # mask: indicating whether the data point is masked for evaluation
        # calculate F-reg on batch.eval_mask (True is masked as unobserved)
        y_tilde = torch.where(mask.bool(), y_hat, y)
        y_tilde = torch.fft.fftn(y_tilde)
        y_tilde = rearrange(y_tilde, 'b s n c -> b (s n c)')
        f1loss = torch.mean(torch.sum(torch.abs(y_tilde), axis=1) / y_tilde.numel())
        return f1loss