import random

import numpy as np
import torch
import torch.nn as nn

from models.csdi.csdi_imputer import CSDIImputer
from models.utils import *

class CSDI(nn.Module):
    def __init__(self, params, parser=None, device="cuda:0"):
        super(CSDI, self).__init__()
        self.model_args = [
            "n_layers",
            "n_channels",
            "n_heads",
            "n_steps",
            "diffusion_embedding_dim",
            "timeemb_dim",
            "featureemb_dim",
            "is_unconditional",
            "cartesian_accel",
            "ppe",
            "fpe",
            "fpi",
            "pe_z_dim",
            "pi_z_dim",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build(device)

    def build(self, device):
        self.model = CSDIImputer(self.params, device=device) 

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
        
        if self.params["missing_pattern"] == "dynamic":
            if mode == "train":
                if self.params["dataset"] == "soccer":
                    missing_pattern = random.choice(["uniform", "playerwise", "camera"])
                else:
                    missing_pattern = random.choice(["uniform", "playerwise"])
            else:
                missing_pattern = "playerwise"
        else:
            missing_pattern = self.params["missing_pattern"]
        # missing_pattern = self.params["missing_pattern"]

        mask, missing_rate = generate_mask(
            data=ret,
            sports=self.params["dataset"],
            mode=missing_pattern,
            missing_rate=missing_rate,
            n_players=total_players,
            single_team=self.params["single_team"],
            # missing_rate=self.params["missing_rate"],
        )  # [bs, time, players]

        bs, seq_len = player_data.shape[:2]
        mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, team_size]
        mask = torch.repeat_interleave(mask, 6, dim=-1)
        timepoints = torch.arange(seq_len).repeat(bs, 1) # [bs, time]

        if self.params["cuda"]:
            mask, timepoints = mask.to(device), timepoints.to(device)

        ret["mask"] = mask
        ret["input"] = player_data
        ret["missing_rate"] = missing_rate
        ret["timepoints"] = timepoints

        if mode != "test":
            is_train = 1 if mode == "train" else 0
            ret = self.model(ret, is_train=is_train, device=device)
        else:
            ret = self.model.evaluate(ret, n_samples=1)

        if player_orders is not None:
            ret["mask"] = sort_players(ret["mask"], player_orders, total_players, mode="restore")
            ret["input"] = sort_players(ret["input"], player_orders, total_players, mode="restore")
            ret["target"] = sort_players(ret["target"], player_orders, total_players, mode="restore")
            if mode == "test":
                # masking missing data for evaluating naive-baseline models in testing phase.
                ret["input"] = ret["input"] * ret["mask"] 
                ret["pred"] = sort_players(ret["pred"], player_orders, total_players, mode="restore")

        return ret