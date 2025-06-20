import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import shift
from torch.autograd import Variable
from tqdm import tqdm

from dataset import SportsDataset
from datatools.trace_helper import TraceHelper

from models.brits.brits import BRITS
from models.midas.midas import MIDAS
from models.naomi.naomi import NAOMI
from models.nrtsi.nrtsi import NRTSI
from models.utils import *


class NFLDataHelper(TraceHelper):
    def __init__(self, 
        traces: pd.DataFrame,
        params: dict = None,
        pitch_size: tuple = (110, 49), 
        n_sample: int = 1, 
        data_path: str = None
        ):
        if traces is not None:
            self.traces = traces
        self.data_path = data_path
        self.total_players = 6
        self.player_cols = [f"player{p}" for p in range(self.total_players)]
        self.player_xy_cols = [f"player{p}{t}" for p in range(self.total_players) for t in ["_x", "_y"]]

        self.ws = 50
        self.n_sample = n_sample
        self.pitch_size = pitch_size

        self.params = params

    def reconstruct_df(self):
        traces_np = np.load(self.data_path)

        bs, seq_len = traces_np.shape[:2]

        # traces_np[..., :6] *= self.pitch_size[0]
        # traces_np[..., 6:] *= self.pitch_size[1]

        x_data = traces_np[..., :6, None]
        y_data = traces_np[..., 6:, None]
        xy_data = np.concatenate([x_data, y_data], axis=-1)

        traces_np = xy_data.reshape(bs, seq_len, -1)  # rearranging the order of the x and y positions.
        traces_np = traces_np.reshape(-1, self.total_players * 2)  # [timesteps, 12]

        traces_df = pd.DataFrame(traces_np, columns=self.player_xy_cols, dtype=float)

        episodes = np.zeros(len(traces_df))
        episodes[0 :: self.ws] = 1
        episodes = episodes.cumsum()

        traces_df["episode"] = episodes.astype("int")
        traces_df["frame"] = np.arange(len(traces_df)) + 1

        self.traces = traces_df

    @staticmethod
    def player_to_cols(p):
        return [f"{p}_x", f"{p}_y", f"{p}_vx", f"{p}_vy", f"{p}_speed", f"{p}_accel", f"{p}_ax", f"{p}_ay"]

    def calc_single_player_running_features(
        self, p: str, episode_traces: pd.DataFrame, remove_outliers=True, smoothing=True
    ):
        episode_traces = episode_traces[[f"{p}_x", f"{p}_y"]]
        x = episode_traces[f"{p}_x"]
        y = episode_traces[f"{p}_y"]

        fps = 0.1
        if remove_outliers:
            MAX_SPEED = 12
            MAX_ACCEL = 8

        if smoothing:
            W_LEN = 12
            P_ORDER = 2
            x = pd.Series(signal.savgol_filter(x, window_length=W_LEN, polyorder=P_ORDER))
            y = pd.Series(signal.savgol_filter(y, window_length=W_LEN, polyorder=P_ORDER))

        vx = np.diff(x.values, prepend=x.iloc[0]) / fps
        vy = np.diff(y.values, prepend=y.iloc[0]) / fps

        if remove_outliers:
            speeds = np.sqrt(vx**2 + vy**2)
            is_speed_outlier = speeds > MAX_SPEED
            is_accel_outlier = np.abs(np.diff(speeds, append=speeds[-1]) / fps) > MAX_ACCEL
            is_outlier = is_speed_outlier | is_accel_outlier | shift(is_accel_outlier, 1, cval=True)

            vx = pd.Series(np.where(is_outlier, np.nan, vx)).interpolate(limit_direction="both").values
            vy = pd.Series(np.where(is_outlier, np.nan, vy)).interpolate(limit_direction="both").values

        if smoothing:
            vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
            vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)

        speeds = np.sqrt(vx**2 + vy**2)
        accels = np.diff(speeds, append=speeds[-1]) / fps

        ax = np.diff(vx, append=vx[-1]) / fps
        ay = np.diff(vy, append=vy[-1]) / fps

        if smoothing:
            accels = signal.savgol_filter(accels, window_length=W_LEN, polyorder=P_ORDER)
            ax = signal.savgol_filter(ax, window_length=W_LEN, polyorder=P_ORDER)
            ay = signal.savgol_filter(ay, window_length=W_LEN, polyorder=P_ORDER)

        self.traces.loc[episode_traces.index, NFLDataHelper.player_to_cols(p)] = (
            np.stack([x, y, vx, vy, speeds, accels, ax, ay]).round(6).T
        )

    def calc_running_features(self, remove_outliers=False, smoothing=False):
        episode = self.traces["episode"].unique()

        for e in tqdm(episode, desc="Calculating running features..."):
            episode_traces = self.traces[self.traces.episode == e]

            for p in self.player_cols:
                self.calc_single_player_running_features(
                    p, episode_traces, remove_outliers=remove_outliers, smoothing=smoothing
                )

        player_cols = np.array([NFLDataHelper.player_to_cols(p) for p in self.player_cols]).flatten().tolist()
        self.traces = self.traces[["episode", "frame"] + player_cols]  # rearange columns.
        
    def predict(
        self,
        model: MIDAS,
        sports="afootball",
        min_episode_size=50,
        naive_baselines=False,
        gap_models=None,
    ) -> Tuple[dict]:
        model_type = self.params["model"]
        random.seed(1000)
        np.random.seed(1000)

        feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]
        player_cols = [f"player{p}{x}" for p in range(6) for x in feature_types]

        pred_keys = ["pred"]
        if model_type == "midas":
            if self.params["deriv_accum"]:
                pred_keys += ["dap_f"]
                if self.params["missing_pattern"] != "forecast":
                    pred_keys += ["dap_b"]
            if self.params["dynamic_hybrid"]:
                pred_keys += ["hybrid_d"]
        if naive_baselines:
            if self.params["missing_pattern"] == "forecast":
                pred_keys += ["ffill"]
            else:
                pred_keys += ["linear", "cubic_spline", "knn", "ffill"]

        stat_keys = ["total_frames", "missing_frames"]
        stat_keys += [f"{k}_{m}" for k in pred_keys for m in ["pe", "se", "sce", "ple"]]

        stats = {k: 0 for k in stat_keys}

        # initialize resulting DataFrames
        ret = dict()
        ret["target"] = self.traces.copy(deep=True)
        ret["mask"] = pd.DataFrame(-1, index=self.traces.index, columns=["episode"] + player_cols)
        ret["mask"].loc[:, "episode"] = self.traces["episode"]
        for k in pred_keys:
            ret[k] = self.traces.copy(deep=True)

        if model_type == "midas" and self.params["dynamic_hybrid"]:
            lambda_types = ["_w0", "_w1"] if self.params["missing_pattern"] == "forecast" else ["_w0", "_w1", "_w2"]
            lambda_cols = [f"{p}{w}" for p in range(10) for w in lambda_types]
            ret["lambdas"] = pd.DataFrame(-1, index=self.traces.index, columns=lambda_cols)

        x_cols = [c for c in self.traces.columns if c.endswith("_x")]
        y_cols = [c for c in self.traces.columns if c.endswith("_y")]

        if self.params["normalize"]:
            self.traces[x_cols] /= self.pitch_size[0]
            self.traces[y_cols] /= self.pitch_size[1]
            self.pitch_size = (1, 1)

        episodes = [e for e in self.traces["episode"].unique() if e > 0]
        for episode in tqdm(episodes, desc="Episode"):
            ep_traces = self.traces[self.traces["episode"] == episode]
            if len(ep_traces) < min_episode_size:
                continue
            ep_player_traces = torch.FloatTensor(ep_traces[player_cols].values)
            ep_ball_traces = ep_player_traces.clone()
            
            with torch.no_grad():
                    ep_ret, ep_stats = self.predict_episode(
                        model,
                        sports,
                        ep_player_traces,
                        ep_ball_traces,
                        pred_keys=pred_keys,
                        window_size=self.params["window_size"],
                        min_window_size=min_episode_size,
                        naive_baselines=naive_baselines,
                        gap_models=gap_models,
                    )

            # update resulting DataFrames
            pos_cols = [c for c in player_cols if c[-2:] in ["_x", "_y"]]
            if self.params["cartesian_accel"]:
                ip_cols = player_cols
            else:
                ip_cols = [c for c in player_cols if c[-3:] not in ["_ax", "_ay"]]

            for k in pred_keys + ["target", "mask"]:
                if k in ["pred", "target", "mask"]:
                    cols = ip_cols if model_type == "midas" else pos_cols
                    ret[k].loc[ep_traces.index, cols] = np.array(ep_ret[k])
                    # ret[k].loc[ep_traces.index, dp_cols] = np.array(ep_ret[k])
                elif naive_baselines and k in ["linear", "c_linear", "knn", "ffill"]:
                    ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])
                else:
                    ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])

            if model_type == "midas" and self.params["dynamic_hybrid"]:
                ep_players = [c[:-2] for c in player_cols if "_x" in c]
                lambda_cols = [f"{p}{w}" for p in ep_players for w in lambda_types]
                ret["lambdas"].loc[ep_traces.index, lambda_cols] = np.array(ep_ret["lambdas"])

            for key in ep_stats:
                stats[key] += ep_stats[key]

        return ret, stats