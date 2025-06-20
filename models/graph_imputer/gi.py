import random

import numpy as np
import torch
import torch.nn as nn

from models.utils import *
from models.graph_imputer.gi_model import GImodel

class GraphImputer(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "cartesian_accel",
            "rnn_dim",
            "std_dim",
            "kld_weight",
            "weighted",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):
        self.gi_f = GImodel(self.params)
        self.gi_b = GImodel(self.params)

    def forward(self, data, mode="train", device="cuda:0"):
        if self.params["single_team"]:
            total_players = self.params["team_size"]
        else:
            total_players = self.params["team_size"] * 2
        
        if "player_order" not in self.params:
            self.params["player_order"] = None

        if self.params["player_order"] == "shuffle":
            player_data, _ = shuffle_players(data[0], n_players=total_players)
            player_orders = None
        elif self.params["player_order"] == "xy_sort":  # sort players by x+y values
            player_data, player_orders = sort_players(data[0], n_players=total_players)
        else:
            player_data, player_orders = data[0], None  # [bs, time, x] = [bs, time, players * feats]
        
        ball_data = data[1] if self.params["dataset"] == "soccer" else None  # [bs, time, 2]
        ret = {"target": player_data, "ball": ball_data}

        if mode == "train":
            missing_prob = np.arange(10) * 0.1
            missing_rate = missing_prob[random.randint(1, 9)]
        else:
            missing_rate = self.params["missing_rate"]

        mask, missing_rate = generate_mask(
            data=ret,
            sports=self.params["dataset"],
            mode=self.params["missing_pattern"],
            missing_rate=missing_rate,
            n_players=total_players,
            single_team=self.params["single_team"],
        )  # [bs, time, players]

        mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, team_size]
        mask = torch.repeat_interleave(mask, 6, dim=-1)

        if self.params["cuda"]:
            mask = mask.to(device)

        ret["mask"] = mask
        ret["input"] = player_data * mask
        ret["missing_rate"] = missing_rate
        reversed_ret = self.reverse(ret)

        # Training or Evaluating
        ret_f = self.gi_f(ret, mode=mode)
        ret_b = self.gi_b(reversed_ret, mode=mode)

        ret = self.merge_ret(ret_f, ret_b, mode=mode)

        aggfunc = "mean" if mode == "train" else "sum"
        ret["pred_pe"] = calc_pos_error(
            ret["pred"], 
            ret["target"], 
            ret["mask"], 
            aggfunc=aggfunc, 
            sports=self.params["dataset"],
            single_team=self.params["single_team"])

        if player_orders is not None:
            ret["mask"] = sort_players(ret["mask"], player_orders, total_players, mode="restore")
            ret["input"] = sort_players(ret["input"], player_orders, total_players, mode="restore")
            ret["target"] = sort_players(ret["target"], player_orders, total_players, mode="restore")
            ret["pred"] = sort_players(ret["pred"], player_orders, total_players, mode="restore")

        return ret

    def reverse(self, ret):
        reversed_ret = dict()
        for key in ret:
            if key != "missing_rate":
                reversed_ret[key] = torch.flip(ret[key], dims=[1])
        
        return reversed_ret

    def merge_ret(self, ret_f, ret_b, mode="train"):
        ret = dict()
        seq_len = ret_f["pred"].shape[1]
        device = ret_f["pred"].device

        if self.params["weighted"]:
            weights = torch.arange(1, seq_len + 1) / seq_len
            weights = weights.unsqueeze(0).unsqueeze(2).to(device)
            reversed_weights = torch.flip(weights, dims=[1]).to(device)
            out = ret_f["pred"] * weights + ret_b["pred"] * reversed_weights
        else:
            out = 0.5 * ret_f["pred"] + 0.5 * ret_b["pred"]
        
        ret["pred"] = out
        ret["mask"] = ret_f["mask"]
        ret["target"] = ret_f["target"]
        ret["missing_rate"] = ret_f["missing_rate"]

        if mode == "train":
            ret["total_loss"] = ret_f["loss"] + ret_b["loss"]

        return ret