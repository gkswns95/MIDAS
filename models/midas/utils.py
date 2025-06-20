from typing import Dict

import torch

from models.utils import get_dataset_config


def ffill(t: torch.Tensor) -> torch.Tensor:  # [bs * agents, time, feats]
    idx0 = torch.tile(torch.arange(t.shape[0]), (t.shape[1], 1)).T.flatten()  # [bs * agents * time]
    idx1 = torch.arange(t.shape[1]).unsqueeze(0).expand(t.shape[0], t.shape[1]).clone()
    idx1[t[..., 0] == 0] = 0
    idx1 = (idx1.cummax(axis=1)[0]).flatten()  # [bs * agents * time]
    return t[tuple([idx0, idx1])].reshape(t.shape)  # [bs * agents, time, feats]


def deriv_accum_pred(data: Dict[str, torch.Tensor], use_accel=False, single_team=False, fb="f", dataset="soccer") -> torch.Tensor:
    n_players, ps = get_dataset_config(dataset, single_team)
    if dataset == "afootball":
        ps = (1, 1)

    scale_tensor = torch.FloatTensor(ps).to(data["pos_pred"].device)
    mask = data["pos_mask"].flatten(0, 1)  # [bs * players, time, 2]
    ip_pos = data["pos_pred"].flatten(0, 1) * scale_tensor
    ip_vel = data["vel_pred"].flatten(0, 1)
    if use_accel:
        ip_accel = data["accel_pred"].flatten(0, 1)

        if fb == "f":
            va = (ip_vel * 0.1 + ip_accel * 0.01).roll(1, dims=1)  # [bs * players, time, 2]
            cumsum_va = ((1 - mask) * va).cumsum(axis=1)
            cumsum_va_by_segment = (1 - mask) * (cumsum_va - ffill(mask * cumsum_va))
            dap_pos = mask * ip_pos + (1 - mask) * (ffill(mask * ip_pos) + cumsum_va_by_segment)

        else:
            ip_pos = torch.flip(ip_pos, dims=(1,))  # [bs * players, time, 2]
            ip_vel = torch.flip(ip_vel, dims=(1,)).roll(2, dims=1)
            ip_accel = torch.flip(ip_accel, dims=(1,)).roll(1, dims=1)
            ip_vel[:, 1] = ip_vel[:, 2].clone()
            mask = torch.flip(mask, dims=(1,))

            va = -ip_vel * 0.1 + ip_accel * 0.01
            cumsum_va = ((1 - mask) * va).cumsum(axis=1)
            cumsum_va_by_segment = (1 - mask) * (cumsum_va - ffill(mask * cumsum_va))
            dap_pos = torch.flip(mask * ip_pos + (1 - mask) * (ffill(mask * ip_pos) + cumsum_va_by_segment), dims=(1,))
    else:
        if fb == "f":
            cumsum_v = ((1 - mask) * ip_vel * 0.1).cumsum(axis=1)  # [bs * players, time, 2]
            cumsum_v_by_segment = (1 - mask) * (cumsum_v - ffill(mask * cumsum_v))
            dap_pos = mask * ip_pos + (1 - mask) * (ffill(mask * ip_pos) + cumsum_v_by_segment)

        else:
            ip_pos = torch.flip(ip_pos, dims=(1,))  # [bs * players, time, 2]
            ip_vel = torch.flip(ip_vel, dims=(1,)).roll(1, dims=1)
            mask = torch.flip(mask, dims=(1,))

            cumsum_v = ((1 - mask) * -ip_vel * 0.1).cumsum(axis=1)
            cumsum_v_by_segment = (1 - mask) * (cumsum_v - ffill(mask * cumsum_v))
            dap_pos = torch.flip(mask * ip_pos + (1 - mask) * (ffill(mask * ip_pos) + cumsum_v_by_segment), dims=(1,))

    return dap_pos.reshape(-1, n_players, dap_pos.shape[1], 2) / scale_tensor  # [bs, players, time, 2]