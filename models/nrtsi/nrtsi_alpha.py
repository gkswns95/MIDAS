import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.nrtsi.nrtsi_imputer import NRTSIImputer
from models.nrtsi.utils import *
from models.utils import *


class NRTSI(nn.Module):
    def __init__(self, params, parser=None):
        super(NRTSI, self).__init__()
        self.model_args = [
            "n_players",
            "n_max_time_scale",
            "time_enc_dim",
            "att_dim",
            "model_dim",
            "inner_dim",
            "time_dim",
            "expand_dim",
            "n_layers",
            "n_heads",
            "n_max_level",
            "cartesian_accel",
            "use_mask",
            "dynamic_missing",
            "xy_sort",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.model_type = "nrtsi"

        self.build()

    def build(self):
        self.model = NRTSIImputer(self.params)

    def forward(self, data, model, gap_models=None, mode="train", teacher_forcing=False, device="cuda:0"):
        ret = {"loss": 0, "pos_dist": 0}

        n_features = self.params["n_features"]
        dataset = self.params["dataset"]

        if dataset == "soccer":
            total_players = 22
        elif dataset == "basketball":
            total_players = 10
        else:  # e.g. "football"
            total_players = 6

        if self.params["xy_sort"]:
            input_data, sort_indices = xy_sort_tensor_v2(
                data[0], n_featrues=n_features, n_players=total_players
            )  # [bs, time, x_dim]
            target_data = input_data.clone()
        else:
            if dataset == "football":  # randomly permute player order for NFL dataset.
                data[0], sort_indices = shuffle_players(data[0], total_players)
                data[1] = data[0].clone()
            input_data = data[0]  # [bs, time, x_dim]
            target_data = data[1]

        if mode == "train":
            min_gap, max_gap = data[2], data[3]

        bs, seq_len = input_data.shape[0], input_data.shape[1]

        missing_mode = "block" if self.params["dynamic_missing"] else "block_all_feat"
        missing_probs = np.arange(10) * 0.1
        mask_data = generate_mask(
            mode=missing_mode, window_size=seq_len, missing_rate=missing_probs[random.randint(1, 9)], sports=dataset
        )
        mask_data = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)
        mask_data = torch.repeat_interleave(mask_data, n_features, dim=-1).expand(bs, -1, -1)  # [bs, time, x_dim]

        if self.params["cuda"]:
            input_data, target_data, mask_data = input_data.to(device), target_data.to(device), mask_data.to(device)

        mask = mask_data.clone()

        loss = Variable(torch.tensor(0.0), requires_grad=True).to(device)

        num_levels = 1e-6
        pos_dist, missing_frames = 1e-6, 1e-6

        input_data = input_data.reshape(bs, seq_len, total_players, -1)  # [bs, time, total_players, feat_dim]
        target_data = target_data.reshape(bs, seq_len, total_players, -1)
        mask_data = mask_data.reshape(bs, seq_len, total_players, -1)
        mask = mask_data.clone()

        if mode == "train":
            for p in range(total_players):
                init_obs = True
                p_input = input_data[:, :, p, :]  # [bs, time, feat_dim]
                p_target = target_data[:, :, p, :]
                p_mask = mask_data[:, :, p, :]
                p_mask_ = mask[:, :, p, :]
                obs_list_count = (p_mask[0, :, 0] == 1).nonzero().reshape(-1).tolist()
                while len(obs_list_count) < seq_len:
                    next_list_count, gap = get_next_to_impute(p_mask, self.params["n_max_level"])
                    obs_list = torch.from_numpy(np.array(obs_list_count)).long()
                    if self.params["cuda"]:
                        obs_list = obs_list.to(device)
                    obs_list_count += next_list_count
                    p_mask[:, next_list_count, :] = 1
                    if min_gap < gap and gap <= max_gap:
                        if teacher_forcing or init_obs:
                            obs_data = p_input[:, obs_list, :]  # [bs, n_obs, feat_dim]
                            init_obs = False
                        gt_data = p_target[:, next_list_count, :]  # [bs, n_imp, feat_dim]

                        obs_list = obs_list[None, :, None].expand(bs, -1, -1)  # [bs, obs_len, 1]
                        next_list = torch.from_numpy(np.array(next_list_count)).long()[None, :, None]  # [1, imp_len, 1]
                        next_list = next_list.expand(bs, -1, -1)  # [bs, imp_len, 1]
                        if self.params["cuda"]:
                            next_list = next_list.to(device)

                        imputations = self.model(obs_data, obs_list, next_list, gap)  # [bs, n_imp, feat_dim]

                        if dataset == "football":
                            gt_data_ = reshape_tensor(
                                gt_data, upscale=False, n_features=n_features, dataset_type=dataset
                            ).flatten(2, 3)
                            loss += nll_gauss(gt_data_, imputations)
                        else:
                            loss += torch.mean(torch.abs(imputations - gt_data))
                            imputations_ = (
                                reshape_tensor(imputations, upscale=True, n_features=n_features, dataset_type=dataset)
                                .detach()
                                .cpu()
                            )  # [bs, n_imp, 1, 2]
                            gt_data_ = (
                                reshape_tensor(gt_data, upscale=True, n_features=n_features, dataset_type=dataset)
                                .detach()
                                .cpu()
                            )
                            mask_ = (
                                reshape_tensor(p_mask_, upscale=False, n_features=n_features, dataset_type=dataset)
                                .detach()
                                .cpu()
                            )  # [bs, time, 1, 2]

                            pos_dist += torch.sum(torch.norm(imputations_ - gt_data_, dim=-1))
                            missing_frames += (1 - mask_[:, next_list_count]).sum() / 2

                        num_levels += 1

                        if not teacher_forcing:
                            obs_data = torch.cat([imputations, obs_data], dim=1)  # [bs, n_imp + n_obs, feat_dim]

            ret["loss"] = loss / num_levels
            # ret["mse_loss"] = mse_loss / num_levels
            if dataset != "football":
                ret["pos_dist"] = pos_dist / missing_frames

        else:  # e.g. "test"
            pred = target_data.clone()
            for p in range(total_players):
                init_obs = True
                p_input = input_data[:, :, p, :]  # [bs, time, feat_dim]
                p_target = target_data[:, :, p, :]
                p_mask = mask_data[:, :, p, :]
                p_mask_ = mask[:, :, p, :]
                obs_list_count = (p_mask[0, :, 0] == 1).nonzero().reshape(-1).tolist()
                while len(obs_list_count) < seq_len:
                    next_list_count, gap = get_next_to_impute(p_mask, self.params["n_max_level"])
                    max_gap = gap_to_max_gap(gap)  # load best model
                    assert gap_models is not None
                    model.load_state_dict(gap_models[max_gap])

                    obs_list = torch.from_numpy(np.array(obs_list_count)).long()
                    if self.params["cuda"]:
                        obs_list = obs_list.to(device)
                    obs_list_count += next_list_count
                    p_mask[:, next_list_count, :] = 1

                    if init_obs:
                        obs_data = p_input[:, obs_list, :]
                        init_obs = False
                    gt_data = p_target[:, next_list_count, :]

                    obs_list = obs_list[None, :, None].expand(bs, -1, -1)  # [bs, obs_len, 1]
                    next_list = torch.from_numpy(np.array(next_list_count)).long()[None, :, None]  # [1, imp_len, 1]
                    next_list = next_list.expand(bs, -1, -1)  # [bs, imp_len, 1]
                    if self.params["cuda"]:
                        next_list = next_list.to(device)

                    imputations = self.model(obs_data, obs_list, next_list, gap)  # [bs, n_imp, y_dim]

                    if dataset == "football":
                        gt_data = reshape_tensor(
                            gt_data, upscale=False, n_features=n_features, dataset_type=dataset
                        ).flatten(2, 3)
                        imputations = sample_gauss(imputations, gt_data, gap=gap)

                    pred[:, next_list_count, p, :] = imputations
                    target[:, next_list_count, p, :] = gt_data

                    obs_data = torch.cat([imputations, obs_data], dim=1)  # [bs, n_imp + n_obs, y_dim]

            pred = pred.flatten(2, 3)  # [bs, time, x_dim]
            target = target_data.flatten(2, 3)
            mask = mask.flatten(2, 3)

            pred_ = reshape_tensor(
                pred, upscale=True, n_features=n_features, dataset_type=dataset
            )  # [bs, time, total_players * 2]
            target_ = reshape_tensor(target, upscale=True, n_features=n_features, dataset_type=dataset)

            ret["pos_dist"] = torch.norm(pred_ - target_, dim=-1).sum().item()

            if self.params["xy_sort"]:
                ret["pred"] = xy_sort_tensor_v2(
                    pred, sort_indices, n_featrues=n_features, n_players=total_players, mode="restore"
                )
                ret["target"] = xy_sort_tensor_v2(
                    target, sort_indices, n_featrues=n_features, n_players=total_players, mode="restore"
                )
                ret["mask"] = xy_sort_tensor_v2(
                    mask, sort_indices, n_featrues=n_features, n_players=total_players, mode="restore"
                )
            else:
                ret["pred"] = pred
                ret["target"] = target
                ret["mask"] = mask

        return ret
