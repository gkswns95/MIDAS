import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import random
from typing import Dict

import matplotlib.colors as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
import torch.nn as nn
from matplotlib import animation, axes, text
from matplotlib.patches import Rectangle
from scipy.ndimage import shift
from tqdm import tqdm

from datatools.nba_utils.Constant import Constant
from datatools.nba_utils.Event import Event
from datatools.nba_utils.Team import Team
from datatools.trace_helper import TraceHelper

from models.brits.brits import BRITS
from models.midas.midas import MIDAS
from models.naomi.naomi import NAOMI
from models.nrtsi.nrtsi import NRTSI
from models.utils import *


class NBADataHelper(TraceHelper):
    def __init__(self, 
        traces: pd.DataFrame, 
        params: dict = None,
        pitch_size: tuple = (28.65, 15.24)):
        self.traces = traces

        self.total_players = 10
        self.player_cols = [f"player{p}" for p in range(self.total_players)]
        self.player_xy_cols = [f"player{p}{t}" for p in range(self.total_players) for t in ["_x", "_y"]]

        self.ws = 200
        self.pitch_size = pitch_size

        self.params = params

    def reconstruct_df(self):
        """
        This function helps to reconstruct data into tracking data form.
        The details of the reconstructed data are as follows:
        - Sort players order based on team IDs(Team A: players[:5], Team B: players[5:]).
        - Remove duplicated events and frames based on "quarter", "game_clock" columns.
        (Because some events contain same tracking data.)
        """
        keys = ["quarter", "game_clock", "shot_clock"] + self.player_xy_cols + ["ball_x", "ball_y", "ball_radius"]
        trace_dict = {key: [] for key in keys}

        n_events = len(self.match["events"])
        for e in tqdm(range(n_events), desc="Reconstructing tracking data..."):
            event_info = Event(self.match["events"][e])
            for frame in range(len(event_info.moments)):
                moment = event_info.moments[frame]
                if len(moment.players) == self.total_players:
                    trace_dict["quarter"] += [moment.quarter]
                    trace_dict["game_clock"] += [moment.game_clock]
                    trace_dict["shot_clock"] += [moment.shot_clock]

                    trace_dict["ball_x"] += [moment.ball.x]
                    trace_dict["ball_y"] += [moment.ball.y]
                    trace_dict["ball_radius"] += [moment.ball.radius]

                    sorted_players = sorted(moment.players, key=lambda player: player.team.id)
                    for p in range(self.total_players):
                        p_moment = sorted_players[p]
                        trace_dict[f"player{p}_x"] += [p_moment.x]
                        trace_dict[f"player{p}_y"] += [p_moment.y]

        self.traces = pd.DataFrame(trace_dict)
        self.traces.drop_duplicates(subset=["quarter", "game_clock"], inplace=True)
        self.traces.reset_index(inplace=True, drop=True)

        x_cols = [c for c in self.player_xy_cols if c.endswith("_x")]
        y_cols = [c for c in self.player_xy_cols if c.endswith("_y")]
        self.traces[x_cols + ["ball_x"]] /= 100
        self.traces[y_cols + ["ball_y"]] /= 50
        self.traces[x_cols + ["ball_x"]] *= self.pitch_size[0]
        self.traces[y_cols + ["ball_y"]] *= self.pitch_size[1]

    def split_into_episodes(self, threshold_dist=3.0):
        """
        This function helps to split tracking data into episodes.
          (It should be called after the "reconstruct_df" function)
        The details of the criteria for dividing episodes are as follows:
        - When transitioning from current quarter to the next quarter.
        - When detecting anomaly frames.
          (e.g. A situations in which players move over the threshold distance within a single frame.)
        """
        traces = self.traces.loc[:, self.player_xy_cols]
        episodes = np.zeros(len(self.traces), dtype=int)

        frame_diff = np.diff(traces, axis=0, prepend=traces[:1])
        frame_diff_dist = np.sqrt(frame_diff[:, 0::2] ** 2 + frame_diff[:, 1::2] ** 2)

        anomaly_frames = (frame_diff_dist > threshold_dist).sum(axis=-1) > 0
        episodes[anomaly_frames] = 1

        self.traces["episode"] = episodes.cumsum() + 1
        self.traces["episode"] = self.traces["episode"].astype(int)

    def downsample_to_10fps(self):
        """
        This function helps to downsampling tracking data from 25fps to 10fps.
        1. Upsample from 25fps to 50fps.
        2. Downsample from 50fps to 10fps.
        """
        meta_cols = ["quarter", "game_clock", "shot_clock"]
        ball_cols = ["ball_x", "ball_y", "ball_radius"]
        total_cols = meta_cols + self.player_xy_cols + ball_cols

        traces_10fps_list = []

        # Upsampling 25fps to 50fps
        upsample_idxs = pd.date_range("2020-01-01 00:00:00.02", periods=len(self.traces) * 2, freq="0.02S")
        traces_50fps = pd.DataFrame(index=upsample_idxs, columns=total_cols, dtype="float")
        traces_50fps.index.name = "datetime"

        traces_50fps.loc[traces_50fps.index[1::2]] = self.traces[total_cols].values
        traces_50fps = traces_50fps.interpolate(limit_area="inside")

        # Downsampling 50fps to 10fps
        traces_10fps = traces_50fps.resample("0.1S", closed="right", label="right").mean()
        traces_10fps = traces_10fps.interpolate(limit_direction="both")

        traces_10fps_list.append(traces_10fps.copy(deep=True))

        traces_10fps = pd.concat(traces_10fps_list).reset_index(drop=True)
        traces_10fps["frame"] = np.arange(len(traces_10fps)) + 1

        self.traces = traces_10fps.loc[:, ["frame"] + total_cols]

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
            W_LEN = 15
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

        self.traces.loc[episode_traces.index, NBADataHelper.player_to_cols(p)] = (
            np.stack([x, y, vx, vy, speeds, accels, ax, ay]).round(6).T
        )

    def calc_running_features(self, remove_outliers=False, smoothing=False):
        episode = self.traces["episode"].unique()

        for e in tqdm(episode, desc="Calculating running features..."):
            episode_traces = self.traces[self.traces.episode == e]
            if len(episode_traces) < 100:
                invalid_episode = self.traces["episode"] == e
                self.traces.loc[invalid_episode, "episode"] = 0
                continue

            for p in self.player_cols:
                self.calc_single_player_running_features(
                    p, episode_traces, remove_outliers=remove_outliers, smoothing=smoothing
                )

        meta_cols = ["quarter", "game_clock", "shot_clock"]
        player_cols = np.array([NBADataHelper.player_to_cols(p) for p in self.player_cols]).flatten().tolist()
        ball_cols = ["ball_x", "ball_y", "ball_radius"]
        self.traces = self.traces[["episode", "frame"] + meta_cols + player_cols + ball_cols]  # rearange columns.

    def predict(
        self,
        model: MIDAS,
        sports="basketball",
        min_episode_size=100,
        naive_baselines=False,
        gap_models=None,
        measure_inference_time=False,
    ) -> Tuple[dict]:
        model_type = self.params["model"]
        random.seed(1000)
        np.random.seed(1000)

        feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]

        if self.params["single_team"]: 
            player_cols = [f"player{p}{x}" for p in range(5) for x in feature_types]
        else:
            player_cols = [f"player{p}{x}" for p in range(10) for x in feature_types]

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

        if measure_inference_time:
            stat_keys += ["infer_times", "n_windows"]

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
            lambda_cols = [f"player{p}{w}" for p in range(10) for w in lambda_types]
            ret["lambdas"] = pd.DataFrame(-1, index=self.traces.index, columns=lambda_cols)
            ret["lambdas"].loc[:, "episode"] = self.traces["episode"]


        x_cols = [c for c in self.traces.columns if c.endswith("_x")]
        y_cols = [c for c in self.traces.columns if c.endswith("_y")]

        if self.params["normalize"]:
            self.traces[x_cols] /= self.pitch_size[0]
            self.traces[y_cols] /= self.pitch_size[1]
            self.pitch_size = (1, 1)
        
        missing_segments = []
        segment_pe = []
        segment_lambdas = []
        episodes = [e for e in self.traces["episode"].unique() if e > 0]
        for e in tqdm(episodes, desc="Episode"):
            ep_traces = self.traces[self.traces["episode"] == e]
            if len(ep_traces) < min_episode_size:
                continue
            ep_player_traces = torch.FloatTensor(ep_traces[player_cols].values)
            ep_ball_traces = ep_player_traces.clone() 
            with torch.no_grad():
                    ep_ret, ep_stats, missing_segments, segment_pe, segment_lambdas = self.predict_episode(
                        model,
                        sports,
                        ep_player_traces,
                        ep_ball_traces,
                        pred_keys=pred_keys,
                        window_size=self.params["window_size"],
                        min_window_size=min_episode_size,
                        naive_baselines=naive_baselines,
                        gap_models=gap_models,
                        missing_segments=missing_segments,
                        segment_pe=segment_pe,
                        segment_lambdas=segment_lambdas,
                        measure_inference_time=measure_inference_time,
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
                elif naive_baselines and k in ["linear", "cubic_spline", "knn", "ffill"]:
                    ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])
                else:
                    ret[k].loc[ep_traces.index, pos_cols] = np.array(ep_ret[k])

            if model_type == "midas" and self.params["dynamic_hybrid"]:
                ep_players = [c[:-2] for c in player_cols if "_x" in c]
                lambda_cols = [f"{p}{w}" for p in ep_players for w in lambda_types]
                ret["lambdas"].loc[ep_traces.index, lambda_cols] = np.array(ep_ret["lambdas"])

            for key in ep_stats:
                stats[key] += ep_stats[key]

        return ret, stats, missing_segments, segment_pe, segment_lambdas


class NBADataAnimator:
    def __init__(self, match_ret: Dict[str, pd.DataFrame] = None, show_episodes=False, show_frames=False, masks=None):
        self.trace_dict = match_ret
        self.masks = masks

        self.total_players = 10

        self.show_episodes = show_episodes
        self.show_frames = show_frames

        anim_cols = [
            "traces",
            "target_traces",
            "masks",
            "player_circles",
            "ball_circle",
            "player_rectangles",
            "annotations",
            "patch_annots",
            "clock_info",
        ]
        self.anim_args = pd.DataFrame(index=self.trace_dict.keys(), columns=anim_cols)

    @staticmethod
    def calc_trace_dist(pred, target):
        pred_np = pred.to_numpy()[None, :]
        target_np = target.to_numpy()[None, :]
        return str(round(np.linalg.norm(pred_np - target_np, axis=1)[0], 4)) + "m"

    @staticmethod
    def plot_speeds_and_accels(traces: pd.DataFrame, players: list = None):
        FRAME_DUR = 30
        MAX_SPEED = 40
        MAX_ACCEL = 6

        fig, axes = plt.subplots(4, 1)
        fig.set_facecolor("w")
        fig.set_size_inches(15, 10)
        plt.rcParams.update({"font.size": 15})

        times = (np.arange(len(traces)) + 1) * 0.1
        t0 = int(times[0] - 0.1)

        axes[0].set(xlim=(t0, t0 + FRAME_DUR), ylim=(0, MAX_SPEED))
        axes[1].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-MAX_ACCEL, MAX_ACCEL))
        axes[2].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-MAX_ACCEL, MAX_ACCEL))
        axes[3].set(xlim=(t0, t0 + FRAME_DUR), ylim=(-MAX_ACCEL, MAX_ACCEL))
        axes[0].set_ylabel("speed")
        axes[1].set_ylabel("aceel")
        axes[2].set_ylabel("aceel_x")
        axes[3].set_ylabel("aceel_y")

        for i, ax in enumerate(axes):
            ax.grid()
            if len(axes) - 1 == i:
                ax.set_xlabel("time")
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        speed_plots = dict()
        accel_plots = dict()
        accel_x_plots = dict()
        accel_y_plots = dict()
        colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]

        for i, p in enumerate(players):
            speeds = traces[f"{p}_speed"].values * 3.6
            accels = traces[f"{p}_accel"].values
            accels_x = traces[f"{p}_ax"].values
            accels_y = traces[f"{p}_ay"].values
            (speed_plots[p],) = axes[0].plot(times, speeds, color=colors[i], label=p)
            (accel_plots[p],) = axes[1].plot(times, accels, color=colors[i], label=p)
            (accel_x_plots[p],) = axes[2].plot(times, accels_x, color=colors[i], label=p)
            (accel_y_plots[p],) = axes[3].plot(times, accels_y, color=colors[i], label=p)

        axes[0].legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))

        def animate(i):
            for ax in axes:
                ax.set_xlim(10 * i, 10 * i + FRAME_DUR)

        frames = (len(traces) - 10 * FRAME_DUR) // 100 + 1
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200)
        plt.close(fig)

        return anim

    @staticmethod
    def animate_player_and_ball(
        t,
        traces,
        target_traces,
        masks,
        player_circles,
        ball_circle,
        player_rectangles,
        annotations,
        patch_annotations,
        clock_info,
        trace_key,
    ):
        trace = traces.iloc[t, :]
        target_trace = target_traces.iloc[t, :]
        if masks is not None:
            mask = masks.iloc[t, :]

        for j in range(len(player_circles)):
            player_circles[j].center = trace[f"player{j}_x"], trace[f"player{j}_y"]
            annotations[j].set_position(player_circles[j].center)

            if trace_key == "main":
                if trace.shot_clock is None:
                    trace.shot_clock = 0.0

                clock_text = "Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}".format(
                    int(trace.quarter), int(trace.game_clock) % 3600 // 60, int(trace.game_clock) % 60, trace.shot_clock
                )
                clock_info.set_text(clock_text)

        if trace_key == "main":
            ball_circle.center = trace["ball_x"], trace["ball_y"]
            ball_circle.radius = trace["ball_radius"] / Constant.NORMALIZATION_COEF

        if player_rectangles is not None:
            for i in range(len(player_rectangles)):
                if mask[f"player{i}_x"] == 0:
                    player_rectangles[i].set_xy(xy=[trace[f"player{i}_x"] - 0.5, trace[f"player{i}_y"] - 0.5])
                    player_rectangles[i].set_alpha(1)
                    player_circles[i].set_alpha(0.5)
                    annotations[i].set_alpha(0.5)
                    patch_annotations[i].set_position((trace[f"player{i}_x"] - 1, trace[f"player{i}_y"] + 0.5))
                    patch_annotations[i].set_text(
                        NBADataAnimator.calc_trace_dist(
                            trace[[f"player{i}_x", f"player{i}_y"]], target_trace[[f"player{i}_x", f"player{i}_y"]]
                        )
                    )
                    patch_annotations[i].set_alpha(1)
                else:
                    player_rectangles[i].set_alpha(0)
                    player_circles[i].set_alpha(0)
                    annotations[i].set_alpha(0)
                    patch_annotations[i].set_alpha(0)

        return player_circles, ball_circle

    @staticmethod
    def plot_players_and_ball(ax: axes.Axes, trace_key: str, masks: pd.DataFrame = None):
        clock_info = ax.annotate(
            "",
            xy=[Constant.X_CENTER, Constant.Y_CENTER],
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

        annotations = [
            ax.annotate(
                player_num,
                xy=[0, 0],
                color="w",
                horizontalalignment="center",
                verticalalignment="center",
                fontweight="bold",
            )
            for player_num in range(10)
        ]

        team1_player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color="red") for _ in range(5)]
        team2_player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color="blue") for _ in range(5)]
        player_circles = team1_player_circles + team2_player_circles
        for circle in player_circles:
            ax.add_patch(circle)

        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color="orange")  # hardcoded orange color
        ax.add_patch(ball_circle)

        if masks is not None and trace_key != "main":
            c = "limegreen" if trace_key == "pred" else "purple"
            player_rectangles, patch_annots = [], []
            for _ in range(10):
                player_rectangle = Rectangle(
                    xy=(0, 0), width=1, height=1, facecolor="none", edgecolor=c, alpha=0, zorder=1, linewidth=1.5
                )
                patch_annot = text.Annotation(
                    text=None, xy=(0, 0), color="dimgrey", fontsize=8, zorder=2, fontweight="bold"
                )

                player_rectangles.append(player_rectangle)
                patch_annots.append(patch_annot)

                ax.add_patch(player_rectangle)
                ax.add_artist(patch_annot)
        else:
            player_rectangles = None
            patch_annots = None

        return [masks, player_circles, ball_circle, player_rectangles, annotations, patch_annots, clock_info]

    def run(self):
        # leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN, Constant.X_MAX), ylim=(Constant.Y_MIN, Constant.Y_MAX))

        ax.axis("off")
        fig = plt.gcf()  # get current figure.
        ax.grid(False)  # remove grid.

        for trace_key in self.trace_dict.keys():
            self.anim_args.loc[trace_key] = [
                self.trace_dict[trace_key],
                self.trace_dict["main"],
            ] + NBADataAnimator.plot_players_and_ball(ax=ax, trace_key=trace_key, masks=self.masks)

            if trace_key == "main":
                if self.show_frames:
                    frame_texts = self.trace_dict["main"]["frame"].apply(lambda x: f"Frame : {x}").values
                    frame_annot = ax.text(0, Constant.Y_MAX + 1, frame_texts[0], fontsize=12, ha="left", va="bottom")
                    frame_annot.set_animated(True)
                if self.show_episodes:
                    episode_texts = self.trace_dict["main"]["episode"].apply(lambda x: f"Episode : {x}").values
                    episode_annot = ax.text(
                        24, Constant.Y_MAX + 1, episode_texts[0], fontsize=12, ha="left", va="bottom"
                    )
                    episode_annot.set_animated(True)

        def animate(t):
            for trace_key in self.anim_args.index:
                NBADataAnimator.animate_player_and_ball(t, *self.anim_args.loc[trace_key, :], trace_key)

                if trace_key == "main":
                    if self.show_frames:
                        frame_annot.set_text(str(frame_texts[t]))
                    if self.show_episodes:
                        episode_annot.set_text(str(episode_texts[t]))

        anim = animation.FuncAnimation(fig, animate, frames=len(self.trace_dict["main"]), interval=100.0)

        court = plt.imread("img/basketball_court.png")
        # court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX, Constant.Y_MAX, Constant.Y_MIN])
        plt.show()
        plt.close(fig)

        return anim
