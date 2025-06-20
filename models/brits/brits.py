import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.brits.rits import RITS
from models.brits.st_rits import STRITS
from models.utils import *

class BRITS(nn.Module):
    def __init__(self, params, parser=None):
        super(BRITS, self).__init__()
        self.model_args = [
            "rnn_dim",
            "dropout",
            "cartesian_accel",
            "ppe",
            "fpe",
            "fpi",
            "pe_z_dim",
            "pi_z_dim",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.build()

    def build(self):        
        if self.params["ppe"] or self.params["fpe"] or self.params["fpi"]:
            self.rits_f = STRITS(self.params)
            self.rits_b = STRITS(self.params)
        else:
            self.rits_f = RITS(self.params)
            self.rits_b = RITS(self.params)

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

        time_gap = time_interval(mask, list(range(mask.shape[1])))
        mask = torch.tensor(mask, dtype=torch.float32)  # [bs, time, team_size]
        mask = torch.repeat_interleave(mask, 6, dim=-1)
        time_gap = torch.repeat_interleave(time_gap, 6, dim=-1)
        # mask = torch.repeat_interleave(mask, self.params["n_features"], dim=-1)
        # time_gap = torch.repeat_interleave(time_gap, self.params["n_features"], dim=-1)

        if self.params["cuda"]:
            mask, time_gap = mask.to(device), time_gap.to(device)

        ret["mask"] = mask
        ret["input"] = player_data * mask  # masking missing values
        ret["missing_rate"] = missing_rate
        ret["delta"] = time_gap

        ret_f = self.rits_f(ret, device=device)
        ret_b = self.reverse(self.rits_b(self.reverse(ret), device=device))
        ret = self.merge_ret(ret_f, ret_b, mode)

        if player_orders is not None:
            ret["mask"] = sort_players(ret["mask"], player_orders, total_players, mode="restore")
            ret["input"] = sort_players(ret["input"], player_orders, total_players, mode="restore")
            ret["target"] = sort_players(ret["target"], player_orders, total_players, mode="restore")
            ret["pred"] = sort_players(ret["pred"], player_orders, total_players, mode="restore")

        return ret

    def merge_ret(self, ret_f, ret_b, mode):
        loss_f = ret_f["loss"]
        loss_b = ret_b["loss"]
        loss_c = self.get_consistency_loss(ret_f["pred"], ret_b["pred"])

        loss = loss_f + loss_b + loss_c

        pred = (ret_f["pred"] + ret_b["pred"]) / 2
        target = ret_f["target"]
        mask = ret_f["mask"]

        aggfunc = "mean" if mode == "train" else "sum"
        pos_error = calc_pos_error(
            pred, 
            target, 
            mask, 
            aggfunc=aggfunc, 
            sports=self.params["dataset"],
            single_team=self.params["single_team"])

        ret_f["total_loss"] = loss
        ret_f["pred_pe"] = pos_error
        ret_f["pred"] = pred

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            device = tensor_.device
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            indices = indices.to(device)

            return tensor_.index_select(1, indices)

        reversed_ret = dict()
        for key in ret:
            if not key.endswith("_loss") and not key.endswith("_rate") and not key.endswith("ball"):
                reversed_ret[key] = reverse_tensor(ret[key])

        return reversed_ret
    # def reverse(self, ret):
    #     def reverse_tensor(tensor_):
    #         device = tensor_.device
    #         if tensor_.dim() <= 1:
    #             return tensor_
    #         indices = range(tensor_.size()[1])[::-1]
    #         indices = Variable(torch.LongTensor(indices), requires_grad=False)

    #         indices = indices.to(device)

    #         return tensor_.index_select(1, indices)

    #     for key in ret:
    #         if not key.endswith("_loss") and not key.endswith("_rate") and not key.endswith("ball"):
    #             ret[key] = reverse_tensor(ret[key])

    #     return ret