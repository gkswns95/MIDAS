import random

import numpy as np
import torch
import torch.nn as nn

from models.utils import *
from models.ImputeFormer.imputeformer_imputer import ImputeFormerModel

class IMPUTEFORMER(nn.Module):
    def __init__(self, params, parser=None):
        super(IMPUTEFORMER, self).__init__()
        self.model_args = [
            "n_nodes",
            "input_embedding_dim",
            "feed_forward_dim",
            "learnable_embedding_dim",
            "n_temporal_heads",
            "n_layers",
            "proj_dim",
            "f1_loss_float",
            "dropout",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):
        self.model = ImputeFormerModel(self.params)

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
            # missing_rate=self.params["missing_rate"],
        )  # [bs, time, players]

        mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, team_size]
        mask = torch.repeat_interleave(mask, 6, dim=-1)
       
        if self.params["cuda"]:
            mask = mask.to(device)

        ret["mask"] = mask
        ret["input"] = player_data * mask  # masking missing values
        ret["missing_rate"] = missing_rate

        ret = self.model(ret, device=device)

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