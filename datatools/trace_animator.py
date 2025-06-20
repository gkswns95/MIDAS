import math
import os
import sys
from typing import Dict

import torch

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, axes, collections, lines, text
from matplotlib.patches import Rectangle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import datatools.matplotsoccer as mps
from models.utils import compute_camera_coverage


class TraceAnimator:
    def __init__(
        self,
        match_ret: Dict[str, pd.DataFrame] = None,
        mask: pd.DataFrame = None,
        show_times=True,
        show_episodes=False,
        show_events=False,
        show_frames=False,
        show_polygon=False,
        annot_cols=None,  # column names for additional annotation
        rotate_pitch=False,
        anonymize=False,
        max_frames=np.inf,
        fps=10,
        play_speed=1,
    ):
        self.trace_dict = match_ret
        self.mask = mask

        self.show_times = show_times
        self.show_episodes = show_episodes
        self.show_events = show_events
        self.show_frames = show_frames
        self.show_polygon = show_polygon
        self.annot_cols = annot_cols
        self.rotate_pitch = rotate_pitch
        self.anonymize = anonymize

        self.max_frames = max_frames
        self.fps = fps
        self.play_speed = play_speed

        anim_cols = ["inplay_records", "team1", "team2", "ball"]
        self.anim_args = pd.DataFrame(index=self.trace_dict.keys(), columns=anim_cols)

        # For simualted carema view
        self.anim_args_camera = pd.DataFrame(columns=["camera_args"])

    @staticmethod
    def calc_trace_dist(pred, target):
        pred_np = pred.to_numpy()[None, :]
        target_np = target.to_numpy()[None, :]
        return str(round(np.linalg.norm(pred_np - target_np, axis=1)[0], 4)) + "m"

    @staticmethod
    def plot_players(
        traces: pd.DataFrame,
        target_traces: pd.DataFrame,
        masks: pd.DataFrame,
        ax: axes.Axes,
        sizes=750,
        alpha=1,
        anonymize=False,
        trace_key="main",
    ):
        if len(traces.columns) == 0:
            return None

        color = "tab:red" if traces.columns[0].startswith("A") else "tab:blue"
        x = traces[traces.columns[0::2]].values
        y = traces[traces.columns[1::2]].values

        size = sizes[0, 0] if isinstance(sizes, np.ndarray) else sizes
        scat = ax.scatter(x[0], y[0], s=size, c=color, alpha=alpha, zorder=2)

        players = [c[:-2] for c in traces.columns[0::2]]
        player_dict = dict(zip(players, np.arange(len(players)) + 1))
        plots = dict()
        annots = dict()
        patchs = dict()
        patch_annots = dict()
        for p in players:
            if masks is not None:
                c = "limegreen" if trace_key == "pred" else "purple"
                (plots[p],) = ax.plot([], [], c=c, alpha=alpha, ls="--", lw=2, zorder=0)
            else:
                (plots[p],) = ax.plot([], [], c=color, alpha=alpha, ls="-", zorder=1)
            player_num = player_dict[p] if anonymize else int(p[1:])
            annots[p] = ax.annotate(
                player_num,
                xy=traces.loc[0, [f"{p}_x", f"{p}_y"]],
                ha="center",
                va="center",
                color="w",
                fontsize=18,
                fontweight="bold",
                annotation_clip=False,
                zorder=3,
            )
            annots[p].set_animated(True)

            if masks is not None:
                c = "limegreen" if trace_key == "pred" else "purple"
                patchs[p] = Rectangle(
                    xy=traces.loc[0, [f"{p}_x", f"{p}_y"]],
                    width=3,
                    height=3,
                    facecolor="none",
                    edgecolor=c,
                    alpha=0,
                    zorder=0,
                    linewidth=2,
                )
                ax.add_patch(patchs[p])
                patchs[p].set_animated(True)

                patch_annots[p] = text.Annotation(
                    text=None, xy=(0, 0), color="dimgrey", fontsize=16, zorder=4, fontweight="bold"
                )
                ax.add_artist(patch_annots[p])
                patch_annots[p].set_animated(True)

        return traces, sizes, scat, plots, annots, patchs, patch_annots, masks, target_traces

    @staticmethod
    def animate_players(
        t: int,
        inplay_records: pd.DataFrame,
        traces: pd.DataFrame,
        sizes: np.ndarray,
        scat: collections.PatchCollection,
        plots: Dict[str, lines.Line2D],
        annots: Dict[str, text.Annotation],
        patchs: Dict[str, Rectangle],
        patch_annots: Dict[str, text.Annotation],
        masks: pd.DataFrame = None,
        target_traces: pd.DataFrame = None,
    ):
        x = traces[traces.columns[0::2]].values
        y = traces[traces.columns[1::2]].values

        scat.set_offsets(np.stack([x[t], y[t]]).T)

        if isinstance(sizes, np.ndarray):
            scat.set_sizes(sizes[t])

        for p in plots.keys():
            inplay_start = inplay_records.at[p, "start_idx"]
            inplay_end = inplay_records.at[p, "end_idx"]
            if t >= inplay_start:
                if t <= inplay_end:
                    t_from = max(t - 60, inplay_start)
                    # t_from = inplay_start
                    # if masks is None:
                    plots[p].set_data(traces.loc[t_from:t, f"{p}_x"], traces.loc[t_from:t, f"{p}_y"])
                    annots[p].set_position(traces.loc[t, [f"{p}_x", f"{p}_y"]])

                    if masks is not None:
                        if masks.loc[t, f"{p}_x"] == 0:
                            patchs[p].set_xy([traces.loc[t, f"{p}_x"] - 3 / 2, traces.loc[t, f"{p}_y"] - 3 / 2])
                            patchs[p].set_alpha(1)
                            plots[p].set_alpha(1)

                            patch_annots[p].set_position((traces.loc[t, f"{p}_x"] - 3, traces.loc[t, f"{p}_y"] + 2))
                            patch_annots[p].set_text(
                                TraceAnimator.calc_trace_dist(
                                    traces.loc[t, [f"{p}_x", f"{p}_y"]], target_traces.loc[t, [f"{p}_x", f"{p}_y"]]
                                )
                            )
                            patch_annots[p].set_alpha(1)
                        else:
                            patchs[p].set_alpha(0)
                            patch_annots[p].set_alpha(0)

                elif t == inplay_end + 1:
                    plots[p].set_alpha(0)
                    annots[p].set_alpha(0)
                    if masks is not None:
                        patchs[p].set_alpha(0)
                        patch_annots[p].set_alpha(0)

                elif t == inplay_end + 1:
                    plots[p].set_alpha(0)
                    annots[p].set_alpha(0)
                    if masks is not None:
                        patchs[p].set_alpha(0)
                        patch_annots[p].set_alpha(0)

    @staticmethod
    def plot_ball(xy: pd.DataFrame, ax=axes.Axes, color="w", edgecolor="k", marker="o", show_path=False):
        x = xy.values[:, 0]
        y = xy.values[:, 1]
        scat = ax.scatter(x[0], y[0], s=300, c=color, edgecolors=edgecolor, marker=marker, zorder=4)

        if show_path:
            pathcolor = "k" if color == "w" else color
            (plot,) = ax.plot([], [], pathcolor, zorder=3)
        else:
            plot = None

        return x, y, scat, plot

    @staticmethod
    def animate_ball(
        t: int,
        x: np.ndarray,
        y: np.ndarray,
        scat: collections.PatchCollection,
        plot: lines.Line2D = None,
    ):
        scat.set_offsets(np.array([x[t], y[t]]))

        if plot is not None:
            t_from = max(t, 0)
            # t_from = max(t - 49, 0)
            plot.set_data(x[t_from : t + 1], y[t_from : t + 1])

    @staticmethod
    def plot_polygon(ball_loc: pd.DataFrame, ax=axes.Axes):
        # camera_info = (0, -20, 20, 30) # x, y, z, fov

        # Generate camera view shape polygon
        ball_loc = torch.tensor(np.array(ball_loc))
        point = compute_camera_coverage(ball_loc[0])
        shp = patches.Polygon(point, color="b", zorder=0)

        # point = compute_polygon_loc(ball_loc.iloc[0], camera_info)
        # shp = patches.Polygon(point, color='b', zorder=0)

        # Plot camera view
        ax.add_patch(shp)
        shp.set_animated(True)

        return ball_loc, shp

    @staticmethod
    def animate_polygon(t: int, ball_loc: np.ndarray, patch: Polygon):
        # camera_info = (0, -20, 20, 30) # x, y, z, fov
        # Compute polygon locations according to ball locations
        ball_loc = torch.tensor(np.array(ball_loc))
        point = compute_camera_coverage(ball_loc[t])
        # point = compute_polygon_loc(ball_loc.iloc[t], camera_info)

        # Update new point
        patch.set_xy(point)
        patch.set_alpha(0.2)

    def plot_init(self, ax: axes.Axes, trace_key: str):
        traces = self.trace_dict[trace_key].iloc[(self.play_speed - 1) :: self.play_speed].copy()
        traces = traces.dropna(axis=1, how="all").reset_index(drop=True)
        if self.mask is not None:
            masks = self.mask.iloc[(self.play_speed - 1) :: self.play_speed].copy()
            masks = masks.dropna(axis=1, how="all").reset_index(drop=True)

        xy_cols = [c for c in traces.columns if c.endswith("_x") or c.endswith("_y")]

        if self.rotate_pitch:
            traces[xy_cols[0::2]] = 108 - traces[xy_cols[0::2]]
            traces[xy_cols[1::2]] = 72 - traces[xy_cols[1::2]]

        inplay_records = []
        for c in xy_cols[::2]:
            inplay_idx = traces[traces[c].notna()].index
            inplay_records.append([c[:-2], inplay_idx[0], inplay_idx[-1]])

        inplay_records = pd.DataFrame(inplay_records, columns=["object", "start_idx", "end_idx"]).set_index("object")

        team1_traces = traces[[c for c in xy_cols if c.startswith("A")]].fillna(-100)
        team2_traces = traces[[c for c in xy_cols if c.startswith("B")]].fillna(-100)

        team1_targets, team2_targets, team1_masks, team2_masks = None, None, None, None
        if trace_key != "main":
            target_traces = self.trace_dict["main"].iloc[(self.play_speed - 1) :: self.play_speed].copy()
            target_traces = target_traces.dropna(axis=1, how="all").reset_index(drop=True)

            team1_targets = target_traces[[c for c in xy_cols if c.startswith("A")]].fillna(-100)
            team2_targets = target_traces[[c for c in xy_cols if c.startswith("B")]].fillna(-100)
            team1_masks = masks[[c for c in xy_cols if c.startswith("A")]].fillna(-100)
            team2_masks = masks[[c for c in xy_cols if c.startswith("B")]].fillna(-100)

        team1_sizes = 750
        team2_sizes = 750

        alpha = 1 if trace_key == "main" else 0.5

        team1_args = self.plot_players(
            team1_traces, team1_targets, team1_masks, ax, team1_sizes, alpha, self.anonymize, trace_key
        )
        team2_args = self.plot_players(
            team2_traces, team2_targets, team2_masks, ax, team2_sizes, alpha, self.anonymize, trace_key
        )

        ball_args = None
        if "ball_x" in traces.columns and traces["ball_x"].notna().any():
            ball_xy = traces[["ball_x", "ball_y"]]

            if trace_key == "main":
                ball_args = TraceAnimator.plot_ball(ball_xy, ax, "w", "k", "o")
            # else:
            #     ball_args = TraceAnimator.plot_ball(ball_xy, ax, trace_key, None, "*")

        if self.show_polygon and trace_key == "main":
            ball_xy = traces[["ball_x", "ball_y"]]
            polygon_args = TraceAnimator.plot_polygon(ball_xy, ax)
            self.anim_args_camera["camera_args"] = polygon_args

        self.trace_dict[trace_key] = traces
        self.anim_args.loc[trace_key] = [inplay_records, team1_args, team2_args, ball_args]

    def run(self):
        fig, ax = plt.subplots(figsize=(20.8, 14.4))
        mps.field("white", fig, ax, show=False)

        for key in self.trace_dict.keys():
            self.plot_init(ax, key)

        traces = self.trace_dict["main"]

        if self.show_times:
            # time_texts = traces["time"].apply(lambda x: f"{int(x // 60):02d}:{x % 60:04.1f}").values
            # time_annot = ax.text(0, 73, time_texts[0], fontsize=20, ha="left", va="bottom")
            # time_annot.set_animated(True)
            time_texts = traces["time"].apply(lambda x: f"{int(x // 60):02d}:{x % 60:04.1f}").values
            time_annot = ax.text(0, 73, time_texts[0], fontsize=30, ha="left", va="bottom")
            time_annot.set_animated(True)

        if self.show_episodes:
            # episode_texts = traces["episode"].apply(lambda x: f"Episode {x}")
            # episode_texts = np.where(episode_texts == "Episode 0", "", episode_texts)
            # episode_annot = ax.text(80, 73, episode_texts[0], fontsize=20, ha="left", va="bottom")
            # episode_annot.set_animated(True)
            episode_texts = traces["episode"].apply(lambda x: f"Episode {x}")
            episode_texts = np.where(episode_texts == "Episode 0", "", episode_texts)
            episode_annot = ax.text(80, 73, episode_texts[0], fontsize=30, ha="left", va="bottom")
            episode_annot.set_animated(True)

        if self.show_events:
            if "event_type" in traces.columns:  # (for Metrica data)
                event_texts = traces.apply(lambda x: f"{x['event_type']} by {x['event_player']}", axis=1)
                event_texts = np.where(event_texts == "nan by nan", "", event_texts)
            else:  # if "event_types" in traces.columns: (for GPS-event data)
                event_texts = traces["event_types"].fillna(method="ffill")
                event_texts = np.where(event_texts.isna(), "", event_texts)

            # event_annot = ax.text(15, 73, str(event_texts[0]), fontsize=20, ha="left", va="bottom")
            # event_annot.set_animated(True)
            event_annot = ax.text(15, 73, str(event_texts[0]), fontsize=30, ha="left", va="bottom")
            event_annot.set_animated(True)

        if self.show_frames is not None:
            # frame_texts = traces["frame"].apply(lambda x: f"Frame : {x}").values
            # frame_annot = ax.text(30, 73, frame_texts[0], fontsize=20, ha="left", va="bottom")
            # frame_annot.set_animated(True)
            frame_texts = traces["frame"].apply(lambda x: f"Frame : {x}").values
            frame_annot = ax.text(30, 73, frame_texts[0], fontsize=30, ha="left", va="bottom")
            frame_annot.set_animated(True)

        if self.annot_cols is not None:
            text_dict = {}
            annot_dict = {}
            for i, col in enumerate(self.annot_cols):
                text_dict[col] = f"{col}: " + np.where(traces[col].isna(), "", traces[col].astype(str))
                annot_dict[col] = ax.text(i * 54, -1, str(text_dict[col][0]), fontsize=20, ha="left", va="top")
                annot_dict[col].set_animated(True)

        def animate(t):
            for key in self.trace_dict.keys():
                inplay_records = self.anim_args.at[key, "inplay_records"]
                team1_args = self.anim_args.at[key, "team1"]
                team2_args = self.anim_args.at[key, "team2"]
                ball_args = self.anim_args.at[key, "ball"]

                if team1_args is not None:
                    TraceAnimator.animate_players(t, inplay_records, *team1_args)
                if team2_args is not None:
                    TraceAnimator.animate_players(t, inplay_records, *team2_args)
                if ball_args is not None:
                    TraceAnimator.animate_ball(t, *ball_args)

                if self.show_polygon and key == "main":
                    camera_args = self.anim_args_camera["camera_args"]
                    TraceAnimator.animate_polygon(t, *camera_args)

            if self.show_times:
                time_annot.set_text(str(time_texts[t]))

            if self.show_episodes:
                episode_annot.set_text(str(episode_texts[t]))

            if self.show_events:
                event_annot.set_text(event_texts[t])

            if self.annot_cols is not None:
                for col in self.annot_cols:
                    annot_dict[col].set_text(str(text_dict[col][t]))

            if self.show_frames is not None:
                frame_annot.set_text(str(frame_texts[t]))
            
            
            # pdf_dir = "./animations/snapshots"
            # pdf_path = os.path.join(pdf_dir, f"frame_{t:04d}.pdf")
            # fig.savefig(pdf_path, format='pdf', dpi=300)

        frames = min(self.max_frames, traces.shape[0])
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 / self.fps)
        plt.close(fig)

        return anim