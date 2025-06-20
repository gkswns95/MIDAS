import random

import numpy as np
import torch
import torch.nn as nn

from models.naomi.naomi_imputer import NAOMIImputer
from models.naomi.naomi_imputer_alpha import NAOMIImputerAlpha
from models.naomi.utils import *
from models.utils import *

class NAOMI(nn.Module):
    def __init__(self, params, parser=None):
        super(NAOMI, self).__init__()
        self.model_args = [
            "rnn_dim",
            "dec1_dim",
            "dec2_dim",
            "dec4_dim",
            "dec8_dim",
            "dec16_dim",
            "n_layers",
            "n_highest",
            "cartesian_accel",
            "stochastic",
            "ppe",
            "fpe",
            "fpi",
            "pe_z_dim",
            "pi_z_dim",
            "agent_wise",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):
        if self.params["missing_pattern"] == "playerwise":
            self.model = NAOMIImputerAlpha(self.params)  # not implemented yet.
        else:
            self.model = NAOMIImputer(self.params)

    def forward(self, data, mode="train", teacher_forcing=False, device="cuda:0"):
        ret = {"loss": 0, "pos_error": 0}
        n_features = self.params["n_features"]

        if self.params["single_team"]:
            total_players = self.params["team_size"]
        else:
            total_players = self.params["team_size"] * 2
            
        if self.params["player_order"] == "shuffle":
            player_data, _ = shuffle_players(data[0], n_players=total_players)
            player_orders = None
        elif self.params["player_order"] == "xy_sort":  # sort players by x+y values
            player_data, player_orders = sort_players(data[0], n_players=total_players)
        else:
            player_data, player_orders = data[0], None  # [bs, time, players * 6]

        ret["target"] = player_data.clone()

        bs, seq_len = player_data.shape[:2]

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

        mask, missing_rate = generate_mask(
            data=ret,
            sports=self.params["dataset"],
            mode=missing_pattern,
            missing_rate=missing_rate,
            n_players=total_players,
            single_team=self.params["single_team"],
            # missing_rate=self.params["missing_rate"],
        )  # [bs, time, players]

        mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, players]
        mask = torch.repeat_interleave(mask, 6, dim=-1).to(device)  # [bs, time, players * 6]

        masked_input = (player_data * mask).reshape(bs, seq_len, total_players, -1)[..., : n_features].flatten(2, 3) # [bs, time, x] = [bs, time, players * feats]
        target_data = player_data.reshape(bs, seq_len, total_players, -1)[..., : n_features].flatten(2, 3)
        mask = mask.reshape(bs, seq_len, total_players, -1)[..., : n_features].flatten(2, 3)

        masked_input = masked_input.transpose(0, 1).to(device)  # [bs, time, x] to [time, bs, x]
        target_data = target_data.transpose(0, 1)
        mask = mask.transpose(0, 1).to(device)

        if self.params["missing_pattern"] == "playerwise": # Not implemented yet.
            masked_input = masked_input.reshape(seq_len, bs, total_players, -1)  # [time, bs, players, feats]
            target_data = target_data.reshape(seq_len, bs, total_players, -1)
            mask = mask.reshape(seq_len, bs, total_players, -1)

            has_value = torch.ones_like(mask, dtype=torch.float32)  # [time, bs, players, feats]
            has_value = has_value * mask
            has_value = has_value[..., 0, None]  # [time, bs, players, 1]

            player_data = torch.cat([has_value, masked_input], dim=-1)  # [time, bs, players, 1+feats]

        elif self.params["missing_pattern"] == "uniform":  
            has_value = torch.ones(seq_len, bs, 1)
            if self.params["cuda"]:
                has_value = has_value.to(device)
            has_value = has_value * mask[..., 0, None]  # [time, bs, 1]
            player_data = torch.cat([has_value, masked_input], dim=-1)  # [time, bs, 1+feats]

        if teacher_forcing:
            batch_loss, pos_error = self.model(player_data, target_data)
            ret["total_loss"] = batch_loss
            ret["pred_pe"] = pos_error
        else:
            data_list = []
            for j in range(seq_len):
                data_list.append(player_data[j : j + 1])
            pred = self.model.sample(data_list)  # [time, bs, feats]

            pred = pred.transpose(0, 1)  # [bs, time, feats]
            if self.params["missing_pattern"] == "playerwise":
                target_ = target_data.flatten(2, 3).transpose(0, 1)
                mask_ = mask.flatten(2, 3).transpose(0, 1)
            else:
                target_ = target_data.transpose(0, 1)
                mask_ = mask.transpose(0, 1)

            batch_loss = self.model.calc_mae_loss(pred, target_)

            aggfunc = "mean" if mode == "train" else "sum"
            pos_error = calc_pos_error(
                pred,
                target_, 
                mask_, 
                n_features=n_features, 
                aggfunc=aggfunc, 
                sports=self.params["dataset"],
                single_team=self.params["single_team"])

            ret["total_loss"] = batch_loss
            ret["pred_pe"] = pos_error

            ret["mask"] = mask.transpose(0,1)
            ret["input"] = masked_input.transpose(0,1)
            ret["target"] = target_data.transpose(0,1)
            ret["pred"] = pred
            ret["missing_rate"] = missing_rate

        if player_orders is not None:
            ret["mask"] = sort_players(ret["mask"], player_orders, total_players, mode="restore")
            ret["input"] = sort_players(ret["input"], player_orders, total_players, mode="restore")
            ret["target"] = sort_players(ret["target"], player_orders, total_players, mode="restore")
            ret["pred"] = sort_players(ret["pred"], player_orders, total_players, mode="restore")

        return ret

    # def forward2(self, input_dict, device="cuda:0"):
    #     ret = {}

    #     masked_input = input_dict["input"].transpose(0, 1)  # [time, bs, feats]
    #     target = input_dict["target"].transpose(0, 1)
    #     mask = input_dict["mask"].transpose(0, 1)

    #     seq_len, bs = target.shape[:2]

    #     has_value = torch.ones(seq_len, bs, 1)
    #     if self.params["cuda"]:
    #         has_value = has_value.to(device)
    #     has_value = has_value * mask[..., 0, None]  # [time, bs, 1]
    #     player_data = torch.cat([has_value, masked_input], dim=-1)  # [time, bs, 1+feats]

    #     data_list = []
    #     for j in range(seq_len):
    #         data_list.append(player_data[j : j + 1])
    #     pred = self.model.sample(data_list)  # [time, bs, feats]

    #     ret["pred"] = pred.transpose(0, 1)  # [bs, time, feats]
    #     ret["target"] = target.transpose(0, 1)
    #     ret["mask"] = mask.transpose(0, 1)

    #     return ret
