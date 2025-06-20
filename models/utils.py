import math
import random
from copy import deepcopy
from typing import Dict, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from shapely.geometry import Point, Polygon
from sklearn.impute import KNNImputer
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve
from torch.autograd import Variable


def get_dataset_config(dataset, single_team=False):
    """
    players : number of total players contained each dataset
    ps : (width, height) pitch sizes
    """
    if dataset == "soccer":
        players = 11 if single_team else 22
        ps = (108, 72)
    elif dataset == "basketball":
        players = 5 if single_team else 10
        ps = (28.65, 15.24)
    elif dataset == "afootball":
        players = 6 if single_team else 6
        ps = (110, 49)
        # ps = (1, 1)
    elif dataset == "gsr":
        players = 30
        ps = (1, 1)

    return players, ps


def is_inside(polygon_vertices, player_pos):
    def cross(p1, p2, on_line_mask, counters):
        x1, y1 = p1[0], p1[1]  # [bs, time]
        x2, y2 = p2[0], p2[1]
        player_x, player_y = player_pos[..., 0], player_pos[..., 1]  # [bs, time, players]

        for p in range(n_players):
            x_p = player_x[..., p]  # [bs, time]
            y_p = player_y[..., p]

            # horizontal line
            condition1 = y1 - y2 == 0  # [bs, time]
            condition2 = y1 == y_p
            condition3 = (torch.min(x1, x2) <= x_p) & (x_p <= torch.max(x1, x2))

            on_line_indices = (condition1 & condition2 & condition3).nonzero()  # [bs, time]
            if len(on_line_indices):
                on_line_mask[on_line_indices[:, 0], on_line_indices[:, 1], p] = 1

            # vertical line
            condition4 = x1 - x2 == 0
            condition5 = (torch.min(y1, y2) <= y_p) & (y_p <= torch.max(y1, y2))
            condition6 = x_p <= torch.max(x1, x2)
            condition7 = x_p == torch.max(x1, x2)

            counter_indices = (condition4 & condition5 & condition6).nonzero()  # [bs, time]
            if len(counter_indices):
                counters[counter_indices[:, 0], counter_indices[:, 1], p] += 1

            on_line_indices = (condition4 & condition5 & condition6 & condition7).nonzero()  # [bs, time]
            if len(on_line_indices):
                on_line_mask[on_line_indices[:, 0], on_line_indices[:, 1], p] = 1

            # diagonal line
            a = (y1 - y2) / (x1 - x2)
            b = y1 - x1 * a
            x = (y_p - b) / a

            condition8 = x_p <= x
            condition9 = (torch.min(y1, y2) <= y_p) & (y_p <= torch.max(y1, y2))
            condition10 = (x_p == x) | (y_p == y1) | (y_p == y2)

            counter_indices = (condition8 & condition9).nonzero()  # [bs, time]
            if len(counter_indices):
                counters[counter_indices[:, 0], counter_indices[:, 1], p] += 1

            on_line_indices = (condition8 & condition9 & condition10).nonzero()  # [bs, time]
            if len(on_line_indices):
                on_line_mask[counter_indices[:, 0], counter_indices[:, 1], p] += 1

        return on_line_mask, counters

    # MAIN
    bs, seq_len, n_players = player_pos.shape[:3]
    on_line_mask = torch.zeros(bs, seq_len, n_players)
    counters = torch.zeros(bs, seq_len, n_players)

    for x in range(len(polygon_vertices)):
        on_line_mask, counters = cross(polygon_vertices[x], polygon_vertices[x - 1], on_line_mask, counters)

    is_contain_mask = counters % 2 == 0

    missing_mask = is_contain_mask + on_line_mask

    missing_mask_np = np.array((1 - missing_mask))
    return missing_mask_np

def generate_mask(
    data: Dict[str, torch.Tensor],
    sports="soccer",
    mode="camera",
    missing_rate=0.8,
    n_players=22,
    single_team=False,
) -> Tuple[np.ndarray, float]:
    assert sports in ["soccer", "basketball", "afootball"]
    assert mode in ["uniform", "playerwise", "camera", "forecast"]

    # n_players, _ = get_dataset_config(sports)
    player_data = data["target"]  # [bs, time, players * feats]

    # compute the length of each sequence without padding
    if player_data.is_cuda:
        valid_frames = np.array(player_data.cpu()[..., 0] != -100).astype(int).sum(axis=-1)  # [bs]
    else:
        valid_frames = np.array(player_data[..., 0] != -100).astype(int).sum(axis=-1)  # [bs]

    if mode == "uniform":  # assert the first and the last frames are not missing
        mask = np.ones((player_data.shape[0], player_data.shape[1], n_players))  # [bs, time, players]

        if sports == "afootball":
            missing_len = random.randint(40, 48)
            mask[:, random.sample(range(1, valid_frames[0] -1), missing_len)] = 0
            # missing_len = random.randint(40, 49) # prev
            # mask[:, random.sample(range(valid_frames[0]), missing_len)] = 0
            # mask[:, 0] = 1
            # mask[:, -1] = 1
        else:
            missing_len = int(valid_frames[0] * missing_rate)
            start_idx = np.random.randint(1, valid_frames[0] - missing_len - 1)
            end_idx = start_idx + missing_len

            mask[:, start_idx : end_idx] = 0

    elif mode == "forecast":
        mask = np.ones((player_data.shape[0], player_data.shape[1], n_players))  # [bs, time, players]
        missing_frames = (valid_frames * missing_rate).astype(int)  # [bs], number of missing values per player
        start_idxs = valid_frames - missing_frames
        for i in range(mask.shape[0]):
            mask[i, start_idxs[i] : valid_frames[i]] = 0

    elif mode == "playerwise":  # assert the first and the last frames are not missing
        mask = np.ones((player_data.shape[0], player_data.shape[1], n_players))  # [bs, time, players]
        missing_frames = np.zeros((mask.shape[0], n_players)).astype(int)  # [bs, players]
        # numbers of player-wise missing values (will increase during the while-loops)

        residue = (valid_frames * n_players * missing_rate).astype(int)  # [bs]
        # total remaining number of missing values (will decrease during the while-loops)

        max_shares = np.ceil(np.maximum(valid_frames - 10, np.min(valid_frames) * 0.9)).astype(int)
        max_shares = np.tile(max_shares, (n_players, 1)).T  # [bs, players]
        # maximum number of missing values for each player

        assert np.all(residue <= max_shares.sum(axis=-1))

        for i in range(mask.shape[0]):
            while residue[i] > 0:  # iteratively distribute residue to the players with available slots
                slots = missing_frames[i] < max_shares[i]  # [players]
                breakpoints = np.random.choice(residue[i] + 1, slots.astype(int).sum() - 1, replace=True)
                shares = np.diff(np.sort(breakpoints.tolist() + [0, residue[i]]))  # [players]

                missing_frames[i, ~slots] = max_shares[i, ~slots]
                missing_frames[i, slots] += shares
                residue[i] = np.clip(missing_frames[i] - max_shares[i], 0, None).sum()  # clip the overflowing shares

        start_idxs = np.random.randint(1, max_shares - missing_frames + 2)  # [bs, players]
        end_idxs = start_idxs + missing_frames  # [bs, players]

        for i in range(mask.shape[0]):
            for p in range(n_players):
                mask[i, start_idxs[i, p] : end_idxs[i, p], p] = 0

    elif mode == "camera":  # assert the first and the last five frames are not missing
        player_data, ball_data = data["target"].clone().cpu(), data["ball"].clone().cpu()
        player_pos = reshape_tensor(player_data, upscale=True, sports=sports, single_team=single_team)  # [bs, time, players, 2]
        ball_pos = normalize_tensor(ball_data, mode="upscale", sports=sports, single_team=single_team)

        if player_data.is_cuda:
            is_pad = np.array(player_data.cpu()[..., :1] == -100).astype(int)
        else:
            is_pad = np.array(player_data[..., :1] == -100).astype(int)

        # check whether the camera view covers each player's position or not
        # is_inside = 1 for on-screen players and 0 for off-screen players
        camera_vertices = compute_camera_coverage(ball_pos)
        mask = is_inside(camera_vertices, player_pos)  # [bs, time, players]
        mask = (1 - is_pad) * mask + is_pad

        mask[:, :5, :] = 1
        for i in range(mask.shape[0]):
            mask[i, valid_frames[i] - 5 :] = 1

        missing_rate = ((1 - is_pad) * (1 - mask)).sum() / ((1 - is_pad).sum() * n_players)

    return mask, missing_rate  # [bs, time, players]


def compute_deltas(mask: np.ndarray, bidirectional=True) -> Tuple[np.ndarray]:
    cumsum_rmasks_f = (1 - mask).cumsum(axis=1)
    cumsum_prevs_f = np.maximum.accumulate(cumsum_rmasks_f * mask, axis=1)
    deltas_f = cumsum_rmasks_f - cumsum_prevs_f

    if bidirectional:
        cumsum_rmasks_b = np.flip(1 - mask, axis=1).cumsum(axis=1)
        cumsum_prevs_b = np.maximum.accumulate(cumsum_rmasks_b * np.flip(mask, axis=1), axis=1)
        deltas_b = np.flip(cumsum_rmasks_b - cumsum_prevs_b, axis=1)
        return deltas_f, deltas_b  # [bs, time, players]
    else:
        return deltas_f  # [bs, time, players]

def time_interval(mask, time_gap, direction="f"):
    if direction == "b":
        mask = np.flip(deepcopy(mask), axis=[0])  

    deltas = np.zeros(mask.shape) # [bs, time, players]
    for batch in range(deltas.shape[0]):
        masks_ = mask[batch]  # [time, players]
        for t in range(1, mask.shape[1]):
            gap = time_gap[t] - time_gap[t - 1]
            for p, m in enumerate(masks_[t - 1]):
                deltas[batch, t, p] = gap + deltas[batch, t - 1, p] if m == 0 else gap

    return torch.tensor(deltas, dtype=torch.float32)

def shuffle_players(tensor: torch.Tensor, n_players=22, shuffled_idxs=None):
    bs, seq_len = tensor.shape[:2]
    tensor = tensor.reshape(bs, seq_len, n_players, -1)  # [bs, time, players, feats]

    shuffled_tensor = tensor.clone()
    shuffled_idxs = torch.zeros((bs, n_players))

    for batch in range(bs):
        rand_idx = np.random.permutation(n_players)
        # rand_idxs = torch.randperm(n_players)
        shuffled_tensor[batch, :] = tensor[batch, :, rand_idx]
        shuffled_idxs[batch] = torch.tensor(rand_idx)

    return shuffled_tensor.flatten(2, 3), shuffled_idxs  # [bs, time, players * feats], [bs, players]


def sort_players(tensor: torch.Tensor, orig_idxs: torch.LongTensor = None, n_players=22, mode="sort"):
    bs, seq_len = tensor.shape[:2]
    tensor = tensor.reshape(bs, seq_len, n_players, -1)  # [bs, time, players, feats]

    x = tensor[..., 0:1]  # [bs, time, players, 1]
    y = tensor[..., 1:2]
    xy = torch.cat([x, y], dim=-1)  # [bs, time, players, 2]

    if mode == "sort":
        x_plus_y = torch.sum(xy, dim=-1)  # [bs, time, players]

        sorted_tensor = tensor.clone()
        sorted_idxs = torch.zeros(bs, n_players, dtype=int)

        for batch in range(bs):
            batch_sorted_idxs = torch.argsort(x_plus_y[batch].mean(axis=0), dim=0)  # [players]
            sorted_tensor[batch] = tensor[batch, :, batch_sorted_idxs]
            sorted_idxs[batch] = batch_sorted_idxs

        return sorted_tensor.flatten(2, 3), sorted_idxs  # [bs, time, players * feats], [bs, players]

    else:  # Restore sorted tensor
        assert orig_idxs is not None
        restored_tensor = tensor.clone()  # [bs, time, players, x]

        for batch in range(bs):
            batch_orig_idxs = torch.argsort(orig_idxs[batch])
            restored_tensor[batch] = tensor[batch, :, batch_orig_idxs, :]

        return restored_tensor.flatten(2, 3)  # [bs, time, players * feats]

def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_args, params, parser):
    if parser is None:
        return params

    for arg in model_args:
        if arg == "missing_pattern":
            continue
        if arg.startswith("pe_") or arg.startswith("pi_"):
            parser.add_argument("--" + arg, type=int, required=False)
        elif arg.startswith("n_") or arg.endswith("_dim") or arg.endswith("size") or arg.endswith("_int"):
            parser.add_argument("--" + arg, type=int, required=True)
        elif arg == "dropout" or arg.endswith("_float"):
            parser.add_argument("--" + arg, type=float, required=False, default=0)
        elif arg == "param_sharing_strategy":
            parser.add_argument("--" + arg, type=str, required=False, default="between_group")
        else:
            parser.add_argument("--" + arg, action="store_true", default=False)

    args, _ = parser.parse_known_args()
    for arg in model_args:
        params[arg] = getattr(args, arg)

    return params


def get_params_str(model_args, params):
    ret = ""
    for arg in model_args:
        if arg in params:
            ret += " {} {} |".format(arg, params[arg])
    return ret[1:-2]


def calc_coherence_loss(p: torch.Tensor, v: torch.Tensor, a: torch.Tensor, m: torch.Tensor, add_va=True):
    p_next = p.roll(-1, dims=1)  # [bs, time, agents, 2]
    v_next = v.roll(-1, dims=1)  # [bs, time, agents, 2]
    pv_loss = torch.norm(((1 - m) * (p_next - p - v_next * 0.1))[:, :-1], p=1, dim=(1, 3))  # [bs, agents]
    if add_va:
        va_loss = torch.norm(((1 - m) * (v_next - v - a * 0.1))[:, :-1], p=1, dim=(1, 3))  # [bs, agents]
        return torch.mean((pv_loss + 0.1 * va_loss) / ((1 - m).sum(dim=1).squeeze(-1) + 1e-5))
    else:
        return torch.mean(pv_loss / ((1 - m).sum(dim=1).squeeze(-1) + 1e-5))


def calc_class_acc(pred_poss, target_poss, aggfunc="mean"):
    if aggfunc == "mean":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(pred_poss, dim=1) == target_poss).float().sum().item()


def calc_f1_score(target, pred):
    tn, fp, fn, tp = confusion_matrix(target, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def calc_auc_score(target, pred):
    fpr, tpr, _ = roc_curve(target, pred, pos_label=1)
    return auc(fpr, tpr)


def calc_pos_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    n_features=None,
    aggfunc="mean",
    sports="soccer",
    reshape=True,
    upscale=True,
    single_team=False,
) -> torch.Tensor:
    
    """
    pred_traces: [time, players * _] if reshape else [time, players, 2]
    target_traces: [time, players * _] if reshape else [time, players, 2]
    mask: [time, players * _] if reshape else [time, players, 2]
    """
    if reshape:
        pred_pos = reshape_tensor(pred, upscale=upscale, n_features=n_features, sports=sports, single_team=single_team)
        target_pos = reshape_tensor(target, upscale=upscale, n_features=n_features, sports=sports, single_team=single_team)
        mask_pos = reshape_tensor(mask, n_features=n_features, sports=sports, single_team=single_team)
    else:
        pred_pos = pred[..., :2]
        target_pos = target[..., :2]
        mask_pos = mask[..., :2]

    if aggfunc == "mean":
        return (torch.norm((pred_pos - target_pos) * (1 - mask_pos), dim=-1).sum() / ((1 - mask_pos).sum() / 2)).item()
    elif aggfunc == "tensor":
        return torch.norm((pred_pos - target_pos) * (1 - mask_pos), dim=-1)
    else:  # if aggfunc == "sum"
        return torch.norm((pred_pos - target_pos) * (1 - mask_pos), dim=-1).sum().item()

def calc_speed_error(pred, target, mask, sports="soccer", upscale=False, single_team=False):
    """
    pred_traces: [time, players, 2]
    target_traces: [time, players, 2]
    mask: [time, players, 2]
    """
    if upscale:
        pred = normalize_tensor(pred, mode="upscale", sports=sports, single_team=single_team)
        target = normalize_tensor(target, mode="upscale", sports=sports, single_team=single_team)

    pred = mask * target + (1 - mask) * pred

    vx_diff = torch.diff(pred[..., 0::2], dim=0) / 0.1
    vy_diff = torch.diff(pred[..., 1::2], dim=0) / 0.1
    pred_speed = torch.sqrt(vx_diff**2 + vy_diff**2)  # [time - 1, players]

    vx_diff = torch.diff(target[..., 0::2], dim=0) / 0.1
    vy_diff = torch.diff(target[..., 1::2], dim=0) / 0.1
    target_speed = torch.sqrt(vx_diff**2 + vy_diff**2)  # [time - 1, players]

    return torch.abs(pred_speed - target_speed).sum().item()


def calc_step_change_and_path_length_errors(
    pred_pos: torch.Tensor,
    target_pos: torch.Tensor,
    mask_pos: torch.Tensor,
    sports="soccer",
    reshape=False,
    upscale=False,
    single_team=False
) -> Tuple[float, float]:
    """
    pred_traces: [time, players * 2] if reshape else [time, players, 2]
    target_traces: [time, players * 2] if reshape else [time, players, 2]
    mask: [time, players * 2] if reshape else [time, players, 2]
    """
    if reshape:
        pred_pos = reshape_tensor(pred_pos, sports=sports, single_team=single_team)  # [time, players, 2]
        target_pos = reshape_tensor(target_pos, sports=sports, single_team=single_team)
        mask_pos = reshape_tensor(mask_pos, sports=sports, single_team=single_team)

    if upscale:
        pred_pos = normalize_tensor(pred_pos, mode="upscale", sports=sports, single_team=single_team)  # [time, players, 2]
        target_pos = normalize_tensor(target_pos, mode="upscale", sports=sports, single_team=single_team)

    pred_pos = mask_pos * target_pos + (1 - mask_pos) * pred_pos

    step_size = torch.norm(pred_pos[1:] - pred_pos[:-1], dim=-1)  # [time, players]
    pred_change_of_step_size = step_size.std(0)  # [players]
    # pred_speed = (step_size / 0.1).std(0)  # [players]
    pred_dist = step_size.sum(0)

    step_size = torch.norm(target_pos[1:] - target_pos[:-1], dim=-1)  # [time, players]
    target_change_of_step_size = step_size.std(0) # [players]
    # target_speed = (step_size / 0.1).std(0)  # [players]
    target_dist = step_size.sum(0)

    sc_error = torch.abs(pred_change_of_step_size - target_change_of_step_size).sum().item()
    pl_error = (torch.abs(pred_dist - target_dist)).sum().item()
    # pl_error = (torch.abs(pred_dist - target_dist) / target_dist).sum().item()

    return sc_error, pl_error


def calc_pred_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sports="soccer", single_team=False):
    # [time, players * feats] -> [time, players * 2]
    pred = reshape_tensor(pred, sports=sports, upscale=True, single_team=single_team)
    target = reshape_tensor(target, sports=sports, upscale=True, single_team=single_team)
    # pred = reshape_tensor(pred, sports=sports, upscale=False, single_team=single_team)
    # target = reshape_tensor(target, sports=sports, upscale=False, single_team=single_team)
    mask = reshape_tensor(mask, sports=sports, upscale=False, single_team=single_team)

    pe = calc_pos_error(pred, target, mask, aggfunc="sum", sports=sports, reshape=False, upscale=False, single_team=single_team)
    se = calc_speed_error(pred, target, mask, sports, upscale=False, single_team=single_team)
    sce, ple = calc_step_change_and_path_length_errors(pred, target, mask, sports, reshape=False, upscale=False, single_team=single_team)

    return pe, se, sce, ple


def reshape_tensor(
    tensor: torch.Tensor,
    upscale=False,
    n_features=None,
    single_team=False,
    mode="pos",
    sports="soccer",
) -> torch.Tensor:
    # tensor: [..., x] = [..., players * feats] -> xy: [..., players, 2]

    n_players, ps = get_dataset_config(sports, single_team)

    tensor = tensor.clone()
    n_features = tensor.shape[-1] // n_players if n_features is None else n_features

    idx_map = {"pos": (0, 1), "vel": (2, 3), "speed": (4,), "accel": (5,), "cartesian_accel": (4, 5)}
    idx = idx_map.get(mode, None)
    assert idx is not None, f"Invalid mode name : {mode}"

    x = tensor[..., idx[0] :: n_features, None]  # [..., players, 1]
    y = tensor[..., idx[1] :: n_features, None] if len(idx) == 2 else None

    if upscale:
        x *= ps[0]
        y *= ps[1]

    xy = torch.cat([x, y], dim=-1) if y is not None else x  # [..., players, 2]
    return xy


def normalize_tensor(tensor, mode="upscale", sports="soccer", single_team=False):
    n_players, ps = get_dataset_config(sports, single_team)

    tensor = tensor.clone()
    if tensor.shape[-1] < n_players:  # if tensor.shape == [..., players, feats]
        n_features = tensor.shape[-1]
    else:  # if tensor.shape == [..., players * feats]
        n_features = tensor.shape[-1] // n_players

    if mode == "upscale":
        tensor[..., 0::n_features] *= ps[0]
        tensor[..., 1::n_features] *= ps[1]
    else:  # if mode == "downscale":
        tensor[..., 0::n_features] /= ps[0]
        tensor[..., 1::n_features] /= ps[1]

    return tensor


def nll_gauss(pred_mean, pred_std, gt, eps=1e-6):
    normal_distri = torch.distributions.Normal(pred_mean, pred_std + eps)
    LL = normal_distri.log_prob(gt)
    NLL = -LL.sum(-1).mean()
    return NLL


def sample_gauss(pred_mean, pred_std, eps=1e-6):
    normal_distri = torch.distributions.Normal(pred_mean, pred_std + eps)
    return normal_distri.sample()


def load_pretrained_model(model, params, freeze=False, trial_num=301):
    save_path = f"saved/{trial_num:03d}"
    pre_trained_state_dict = torch.load(
        f"{save_path}/model/{params['model']}_state_dict_best.pt", map_location="cuda:0"
    )
    new_model_state_dict = model.state_dict()

    sub_module_keys = []
    for k, v in pre_trained_state_dict.items():
        if k in new_model_state_dict:
            new_model_state_dict[k] = v
            sub_module_keys.append(k)

    if freeze:
        for k, p in model.named_parameters():
            if k in sub_module_keys:
                p.requires_grad = False

    model.load_state_dict(new_model_state_dict)

    print(f"successfully load pretrained submodule parameters.{trial_num}")
    return model


def compute_camera_coverage(ball_pos: torch.Tensor, camera_info=(0, -20, 20, 30), pitch_size=(108, 72)):
    # Camera info
    camera_x = camera_info[0]
    camera_y = camera_info[1]
    camera_z = camera_info[2]
    camera_fov = camera_info[3]

    # Camera settings
    camera_x = pitch_size[0] / 2
    camera_xy = torch.tensor([camera_x, camera_y])
    camera_height = camera_z
    camera_ratio = (16, 9)

    camera_fov_x = math.radians(camera_fov)
    camera_fov_y = math.radians(camera_fov / camera_ratio[0] * camera_ratio[1])

    ball_right = ball_pos[..., 0] > pitch_size[0] / 2

    # Camera-ball angle
    ball_camera_dist = torch.norm(ball_pos - camera_xy, dim=-1)  # [bs, time]
    camera_ball_angle = math.pi / 2 - np.arctan(
        abs(camera_xy[1] - ball_pos[..., 1]) / abs(camera_xy[0] - ball_pos[..., 0])
    )
    camera_ball_angle_y = np.arctan(camera_height / ball_camera_dist)

    front_dist = camera_height / np.tan(camera_ball_angle_y + camera_fov_y / 2)
    rear_dist = camera_height / np.tan(camera_ball_angle_y - camera_fov_y / 2)
    # front_dist = camera_height / math.tan(camera_ball_angle_y + camera_fov_y / 2)
    # rear_dist = camera_height / math.tan(camera_ball_angle_y - camera_fov_y / 2)

    front_ratio = front_dist / ball_camera_dist
    rear_ratio = rear_dist / ball_camera_dist

    camera_fov = math.radians(camera_fov)  # in degree

    # Create a Polygon from the coordinates
    # ball_fov_dist_x = (ball_camera_dist * math.tan(camera_fov_x / 2)) * math.cos(camera_ball_angle)
    # ball_fov_dist_y = (-1) ** (ball_right) * (ball_camera_dist * math.tan(camera_fov_x / 2))
    # * math.sin(camera_ball_angle)
    # ball_fov_dist_y = (ball_camera_dist * math.tan(camera_fov / 2)) * math.sin(camera_ball_angle)

    ball_camera_close_dist = ball_camera_dist * front_ratio
    ball_camera_far_dist = ball_camera_dist * rear_ratio

    sign_y = (-1) ** ball_right
    ball_fov_close_dist_x = (ball_camera_close_dist * np.tan(camera_fov_x / 2)) * np.cos(camera_ball_angle)
    ball_fov_close_dist_y = sign_y * (ball_camera_close_dist * np.tan(camera_fov_x / 2)) * np.sin(camera_ball_angle)
    ball_fov_far_dist_x = (ball_camera_far_dist * np.tan(camera_fov_x / 2)) * np.cos(camera_ball_angle)
    ball_fov_far_dist_y = sign_y * (ball_camera_far_dist * np.tan(camera_fov_x / 2)) * np.sin(camera_ball_angle)

    front_ratio = front_ratio.unsqueeze(-1)
    rear_ratio = rear_ratio.unsqueeze(-1)

    close_fov_center_point = ball_pos * (front_ratio) + camera_xy * (1 - front_ratio)
    far_fov_center_point = ball_pos * rear_ratio + camera_xy * (-rear_ratio + 1)

    poly_loc0 = (
        far_fov_center_point[..., 0] - ball_fov_far_dist_x,
        far_fov_center_point[..., 1] - ball_fov_far_dist_y,
    )
    poly_loc1 = (
        far_fov_center_point[..., 0] + ball_fov_far_dist_x,
        far_fov_center_point[..., 1] + ball_fov_far_dist_y,
    )
    poly_loc2 = (
        close_fov_center_point[..., 0] + ball_fov_close_dist_x,
        close_fov_center_point[..., 1] + ball_fov_close_dist_y,
    )
    poly_loc3 = (
        close_fov_center_point[..., 0] - ball_fov_close_dist_x,
        close_fov_center_point[..., 1] - ball_fov_close_dist_y,
    )
    vertices = (poly_loc0, poly_loc1, poly_loc2, poly_loc3)

    return vertices


# def print_helper(ret, pred_keys, trial=-1, dataset="soccer", save_txt=False):
#     def get_key_index(key, model_keys):
#         for model_key in model_keys:
#             if model_key in key:
#                 return model_keys.index(model_key)

#     n_players, _ = get_dataset_config(dataset)

#     if save_txt:
#         f = open(f"saved/{trial:03d}/results.txt", "w+")
#     keys = [key for key in list(ret.keys()) if key not in ["total_frames", "missing_frames"]]
#     keys = sorted(keys, key=lambda x: get_key_index(x, model_keys=pred_keys))
#     for key in keys:
#         if "df" in key:
#             continue
#         elif "travel" in key:
#             print(f'{key} : {round(ret[key] / (ret["total_frames"] * n_players), 8)}')
#         elif "step_change" in key:
#             print(f'{key} : {round(ret[key] / (ret["total_frames"] * n_players), 8)}')
#         elif "path_length" in key:
#             print(f'{key} : {round(ret[key] / (ret["total_frames"] * n_players), 8)}')
#         elif "dist" in key:
#             print(f'{key} : {round(ret[key].item() / ret["missing_frames"], 8)}')
#         else:
#             print(f'{key} : {round(ret[key] / ret["missing_frames"], 8)}')

#         if save_txt:
#             if "df" in key:
#                 continue
#             elif "travel" in key:
#                 f.write(f'{key} : {round(ret[key] / (ret["total_frames"] * n_players), 8)} \n')
#             elif "step_change" in key:
#                 f.write(f'{key} : {round(ret[key] / (ret["total_frames"] * n_players), 8)} \n')
#             elif "path_length" in key:
#                 f.write(f'{key} : {round(ret[key] / (ret["total_frames"] * n_players), 8)} \n')
#             elif "dist" in key:
#                 f.write(f'{key} : {round(ret[key].item() / ret["missing_frames"], 8)} \n')
#             else:
#                 f.write(f'{key} : {round(ret[key] / ret["missing_frames"], 8)} \n')
#     if save_txt:
#         f.close()
