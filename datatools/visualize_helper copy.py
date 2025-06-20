import os
import sys
from collections import Counter
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datatools.trace_helper import TraceHelper

import datatools.matplotsoccer as mps
from models.utils import calc_pos_error, get_dataset_config, reshape_tensor

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())


class VisualizeHelper:
    def __init__(
        self,
        rets: Dict[str, pd.DataFrame],
        plot_mode: str,
        dataset: str,
        helper,
    ):
        self.traces = helper.traces

        self.mode = plot_mode
        self.dataset = dataset

        self.rets = rets
        self.helper = helper

        self.params = helper.params
        self.wlen = helper.params["window_size"]

        if dataset == "soccer":
            if self.params["single_team"]:
                players = helper.team1_players
            else:
                players = helper.team1_players + helper.team2_players
        elif dataset == "basketball":
            if self.params["single_team"]:
                players = ["player" + str(i) for i in range(5)]
            else:
                players = ["player" + str(i) for i in range(10)]

        self.p_cols = [f"{p}{f}" for p in players for f in ["_x", "_y"]]
        if "midas" in rets.keys():
            self.w_cols = [f"{p}{w}" for p in players for w in ["_w0", "_w1", "_w2"]]

    def valid_episodes(self):
        episodes = self.traces["episode"].unique()

        valid_episodes = []
        for e in episodes:
            traces = self.traces[self.traces["episode"] == e]
            if traces.shape[0] >= self.wlen and e != 0:
                valid_episodes.append(e)

        self.val_epi_list = np.array(valid_episodes)
        print(f"Valid episode idxs : {valid_episodes}")

    def get_cfg(self):
        pred_keys = {}
        fs = (16, 3)
        if self.mode == "imputed_traj":
            fs = (8, 5) # width, height
            # pred_keys.update(
            #     {
            #         "mask": "mask",    
            #         "target": "Target",
            #         "midas": "MIDAS",
            #     })
            pred_keys.update(
                {
                    "mask": "mask",
                    "target": "Target",
                    # "linear": "Linear",
                    # "nrtsi": "NRTSI",
                    # "csdi": "CSDI",
                    # "imputeformer": "ImputeFormer",
                    # "midas": "MIDAS",
                    "dbhp": "DBHP",
                })
            # dict_keys(['target', 'mask', 'imputeformer', 'linear', 'midas', 'csdi', 'nrtsi', 'naomi'])

            # pred_keys.update(
            #     {
            #         "mask": "mask",
            #         "target": "Ground Truth",
            #         "midas": "MIDAS-D",
            #         "linear": "Linear Interpolation",
            #         "brits": "BRITS",
            #         "naomi": "NAOMI",
            #         "csdi": "CSDI",
            #         "nrtsi": "NRTSI",
            #         "linear_2": "Graph Imputer",
            #     })
        elif self.mode == "dist_heatmap":
            pred_keys.update(
                {
                    "pred": "",
                    "dap_f": "",
                    "dap_b": "",
                    "hybrid_s2": "",
                    "hybrid_d": "",
                }
            )
        elif self.mode == "weights_heatmap":
            pred_keys["lambdas"] = "lambdas"
            pred_keys["mask"] = "mask"
        
        # pred_keys.update({"target": "Ground Truth", "mask": "mask"})

        if self.mode == "pitch_control":
            pred_keys = {}
            pred_keys["target"] = "target"
            pred_keys["csdi"] = "csdi"
            # pred_keys["dbhp"] = "dbhp"s
            pred_keys["linear"] = "linear"
            pred_keys["ball"] = "ball"

        return pred_keys, fs

    def plot_trajectories(self):
        n_players, ps = get_dataset_config(self.dataset)
        for i, (key, title) in enumerate(self.pred_keys.items()):
            if key == "mask":
                continue

            ### debug ###
            # if key.startswith("tmp"):
            #     key = "linear"
            if key == "target":
                continue
            ### debug ###

            ax = self.fig.add_subplot(4, 2, i)
            # ax.set_xlim(0, ps[0] / 2 + 3)  # Show only the left half of the field
            # ax.set_ylim(0, ps[1])
            ax.set_xlim(0, ps[0])
            ax.set_ylim(0, ps[1])

            if self.dataset == "soccer":
                mps.field("blue", self.fig, ax, show=False)
                s1 = 0.8
                s2 = 3
                lw = 0.5
                c = [
                    "#7D3C98",
                    "#00A591",
                    "#3B3B98",
                    "#E89B98",
                    "#FAB5DA",
                    "#FF5733",
                    "#33FF57",
                    "#5733FF",
                    "#FFFF33",
                    "#33FFFF",
                    "#FF33FF",
                ]
                plt.subplots_adjust(wspace=0.1)
            elif self.dataset == "basketball":
                court = plt.imread("img/basketball_court.png")
                s1 = 1
                s2 = 2
                lw = 1
                # s1 = 10
                # s2 = 20
                # lw = 1.5
                # c = ["#7D3C98", "#00A591", "#3B3B98", "#E89B98", "#FAB5DA"]
                c = ["orangered", "limegreen", "goldenrod", "royalblue", "violet"]
                ax.imshow(court, zorder=1, extent=[0, ps[0], ps[1], 0])

            w_traces = self.pred_dict[f"window_{key}"]
            w_target_traces = self.pred_dict[f"window_target"]
            w_mask = self.pred_dict["window_mask"] 
            for i, k in enumerate(range(n_players // 2)):
                p_xy = torch.cat([w_traces[:, 2 * k, None], w_traces[:, 2 * k + 1, None]], dim=-1)
                target_xy = torch.cat([w_target_traces[:, 2 * k, None], w_target_traces[:, 2 * k + 1, None]], dim=-1)
                mask_xy = torch.cat([w_mask[:, 2 * k, None], w_mask[:, 2 * k + 1, None]], dim=-1) 
                obs_idxs = np.where((w_mask[:, 2 * k] != 0) | (w_mask[:, 2 * k + 1] != 0))
                missing_idxs = np.setdiff1d(np.arange(w_traces.shape[0]), obs_idxs[0])
                
                if key != "target":
                    ax.plot(
                        w_traces[missing_idxs, 2 * k],
                        w_traces[missing_idxs, 2 * k + 1],
                        color=c[i],
                        linewidth=0.7,
                        alpha=1,
                        zorder=3,
                        linestyle="--",
                    )
                else:
                    ax.plot(
                        w_target_traces[missing_idxs, 2 * k],
                        w_target_traces[missing_idxs, 2 * k + 1],
                        color=c[i],
                        linewidth=0.7,
                        alpha=1,
                        zorder=3,
                        linestyle="--",
                    )
                    
                observed_alpha = 0.3 if key == "target" else 0.3
                ax.plot(
                    w_target_traces[:, 2 * k], 
                    w_target_traces[:, 2 * k + 1], 
                    color=c[i], 
                    linewidth=0.5, 
                    alpha=observed_alpha, 
                    zorder=2) # observed

                ax.scatter(
                    w_traces[0, 2 * k], 
                    w_traces[0, 2 * k + 1], 
                    marker="o", 
                    edgecolor="black",
                    c=c[i],
                    linewidths=0.5,
                    s=10, 
                    alpha=1, 
                    zorder=2,
                )  # starting point

                width = 1
                height = 1
                box = Rectangle(
                    xy=(w_traces[0, 2 * k] - width / 2, w_traces[0, 2 * k + 1] - height / 2),
                    width=width,
                    height=height,
                    facecolor="none",
                    edgecolor="black",
                    alpha=1.0,
                    zorder=4,
                    linewidth=0.3,
                )
                ax.add_patch(box)

                pe = (torch.norm((p_xy - target_xy) * (1 - mask_xy), dim=-1).sum() / ((1 - mask_xy).sum() / 2)).item()
       
                ax.text(
                w_traces[0, 2 * k] + 0.5,
                w_traces[0, 2 * k + 1] + 1 / 2 + 0.4,
                f"{pe:.4f}m",
                color="black", 
                fontsize=6, 
                ha="center",
                va="bottom", 
                alpha=0.9,
                zorder=5 
                )   
    
                ax.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False,
                )
                # ax.set_title(title.format(i + 1), fontsize=10, loc="center")

    def plot_hybrid_weights(self):
        plt.rcParams["font.family"] = "sans-serif"
        f, axes = plt.subplots(1,3)
        # f.set_size_inches((28, 7))
        f.set_size_inches((16,6))

        mask = self.pred_dict["window_mask"]
        lambdas = self.pred_dict["window_lambdas"]
    
        m = reshape_tensor(mask, sports=self.dataset, single_team=self.params["single_team"])
        m = m[..., 1].squeeze(-1)
        m = np.array((m == 1))

        for i, title in enumerate(["DP", "DAP-F", "DAP-B"]):
            if title == "DAP-B":
                heatmap = sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=True, mask=m, ax=axes[i], vmin=0, vmax=1)
            else:
                heatmap = sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=False, mask=m, ax=axes[i])

            axes[i].set_xlabel("Agent", fontsize=40)
            if title == "DP":
                axes[i].set_ylabel("Time step", fontsize=40)
            
            axes[i].set_title(title.format(i + 1), fontsize=45, pad=20)
            axes[i].set_xticks([p for p in range(22) if p % 4 == 0])
            axes[i].set_xticklabels([p for p in range(22) if p % 4 == 0], fontsize=30)
            axes[i].set_yticks([s for s in range(200) if s % 40 == 0])
            axes[i].set_yticklabels([s for s in range(200) if s % 40 == 0], fontsize=30)
            # axes[i].set_yticks([0, 50, 100, 150])
            # axes[i].set_yticklabels([0, 50, 100, 150], fontsize=30)

            if title == "DAP-B":  
                colorbar = heatmap.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=25)

        return f

    # def plot_hybrid_weights(self):
    #     ### debug ###
    #     seq1 = torch.load("./seq1")
    #     seq5 = torch.load("./seq5")
    #     ### debug ###

    #     mask = self.pred_dict["window_mask"]
    #     lambdas = self.pred_dict["window_lambdas"]
    
    #     m = reshape_tensor(mask, sports=self.dataset, single_team=self.params["single_team"])
    #     m = m[..., 1].squeeze(-1)
    #     m = np.array((m == 1))

    #     self.fig.set_size_inches(16, 6)

    #     for j in range(2):
    #         ### debug ###
    #         if j == 0:
    #             mask = seq1["window_mask"]
    #             lambdas = seq1["window_lambdas"]
            
    #             m = reshape_tensor(mask, sports=self.dataset, single_team=self.params["single_team"])
    #             m = m[..., 1].squeeze(-1)
    #             m = np.array((m == 1))
    #         else:
    #             mask = seq5["window_mask"]
    #             lambdas = seq5["window_lambdas"]
            
    #             m = reshape_tensor(mask, sports=self.dataset, single_team=self.params["single_team"])
    #             m = m[..., 1].squeeze(-1)
    #             m = np.array((m == 1))

    #         for i, title in enumerate(["DP", "DAP-F", "DAP-B"]):
    #             ax = self.fig.add_subplot(2, 4, j * 4 + i + 1) # for apendix
    #             # ax = self.fig.add_subplot(1, 4, i + 1)

    #             if title == "DAP-B":
    #                 sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=True, mask=m, ax=ax, vmin=0, vmax=1)
    #             else:
    #                 sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=False, mask=m, ax=ax)

    #             ax.set_xlabel("Agent", fontsize=12)
    #             if title == "DP":
    #                 ax.set_ylabel("Time step", fontsize=12)
    #             if j == 0:
    #                 ax.set_title(title.format(i + 1), fontsize=20, loc="center")

    #             ax.set_yticks([0, 50, 100, 150, 200])
    #             ax.set_yticklabels([0, 50, 100, 150, 200], rotation=0)
        
    #     plt.subplots_adjust(hspace=0.5)

    #     return self.fig

    # def plot_hybrid_weights(self):
    #     mask = self.pred_dict["window_mask"]
    #     lambdas = self.pred_dict["window_lambdas"]
    
    #     m = reshape_tensor(mask, sports=self.dataset, single_team=self.params["single_team"])
    #     m = m[..., 1].squeeze(-1)
    #     m = np.array((m == 1))

    #     for i, title in enumerate(["DP", "DAP-F", "DAP-B"]):
    #         ax = self.fig.add_subplot(1, 4, i + 1)

    #         if title == "DAP-B":
    #             sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=True, mask=m, ax=ax, vmin=0, vmax=1)
    #         else:
    #             sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=False, mask=m, ax=ax)

    #         ax.set_xlabel("Agent", fontsize=12)
    #         ax.set_ylabel("Time step", fontsize=12)
    #         ax.set_title(title.format(i + 1), fontsize=20, loc="center")

    #         ax.set_yticks([0, 50, 100, 150, 200])
    #         ax.set_yticklabels([0, 50, 100, 150, 200], rotation=0)

    #     return self.fig

    def plot_dist_heatmap(self):
        target = self.pred_dict["window_target"]
        mask = self.pred_dict["window_mask"]

        i = 0
        for key, title in self.pred_keys.items():
            if key in ["target", "mask"]:
                continue

            ax = self.fig.add_subplot(1, 6, i + 1)

            pred = self.pred_dict[f"window_{key}"]

            pred_dist = calc_pos_error(
                pred, 
                target, 
                mask, 
                upscale=False, 
                aggfunc="tensor", 
                sports=self.dataset,
                single_team=self.helper.params["single_team"])

            m = reshape_tensor(mask, sports=self.dataset, single_team=self.helper.params["single_team"])
            m = m[..., 1].squeeze(-1)
            m = np.array((m == 1))

            sns.heatmap(pred_dist,
                        cmap="viridis", 
                        cbar=True, 
                        mask=m, 
                        ax=ax)

            ax.set_title(f"{title} L2 distance")
            ax.set_xlabel("Agents", fontsize=12)
            ax.set_ylabel("Timesteps", fontsize=12)

            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticks([0, 50, 100, 150, 200])
            ax.set_yticklabels([0, 50, 100, 150, 200], rotation=0)
            # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_title(title.format(i + 1), fontsize=18, loc="center", y=1.05)

            i += 1
        
    def plot_pitch_control(self):
        self.pc_ret = dict()
        ball_traces = self.pred_dict[f"window_ball"] # [time, 2]
        for k in self.pred_keys:
            if k != "ball":
                traces = self.pred_dict[f"window_{k}"] # [time, x_dim]
                self.pc_ret[f"{k}_pc"] = TraceHelper._compute_pitch_control(model_type=k, traces=traces, ball_traces=ball_traces)
                self.pc_ret[k] = traces
            else:
                self.pc_ret[k] = ball_traces

    def plot_run(self, epi_idx):
        e = self.val_epi_list[epi_idx]
        path = f"plots/{self.dataset}/{self.mode}/episode_{e}"
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"Plotting episode : {e}")
        print(f"Plot Mode : {self.mode}")
        print(f"Saved path : {path}")

        epi_traces = self.traces[self.traces["episode"] == e]

        self.pred_keys, fs = self.get_cfg()
        self.pred_dict = {}
        for seq, i in enumerate(range(epi_traces.shape[0] // self.wlen + 1)):
            self.fig = plt.figure(figsize=fs, dpi=300)

            i_from = self.wlen * i
            i_to = self.wlen * (i + 1)

            if epi_traces[i_from:i_to].shape[0] != self.wlen:
                continue

            # Processing window inputs
            for key in self.pred_keys.keys():
                cols = self.w_cols if key == "lambdas" else self.p_cols
                if key.startswith("linear"):
                    epi_df = self.rets["linear"][self.rets["linear"]["episode"] == e][cols]
                elif key.startswith("ball"):
                    epi_df = self.rets["ball"][self.rets["ball"]["episode"] == e][["x", "y"]]
                else:
                    epi_df = self.rets[key][self.rets[key]["episode"] == e][cols]
                # epi_df = self.rets[f"{key}_df"][self.rets[f"{key}_df"]["episode"] == e][cols]
                epi_df = epi_df[i_from:i_to].replace(-1, np.nan)
                self.pred_dict[f"window_{key}"] = torch.tensor(epi_df.dropna(axis=1).values)

            ### debug ###
            # if seq == 1:
            #     torch.save(self.pred_dict, "./seq1")
            # elif seq == 5:
            #     torch.save(self.pred_dict, "./seq5")
            ### debug ###

            # Plotting start
            if self.mode == "imputed_traj":
                self.plot_trajectories()
            elif self.mode == "dist_heatmap":
                self.plot_dist_heatmap()
            elif self.mode == "weights_heatmap":
                fig = self.plot_hybrid_weights()
                plt.tight_layout()
                fig.savefig(f"{path}/seq_{seq}.png", bbox_inches="tight")
                # fig.savefig(f"{path}/seq_{seq}.pdf")
            elif self.mode == "pitch_control":
                self.plot_pitch_control()

            # plt.tight_layout()
            # plt.subplots_adjust(wspace=-0.6, hspace=0.3)
            # self.fig.savefig(f"{path}/seq_{seq}.png", bbox_inches="tight")
            
            # Imputed Trajectories
            self.fig.savefig(f"{path}/seq_{seq}.png", bbox_inches="tight", dpi=300)
            self.fig.savefig(f"{path}/imputeformer_imputed.pdf", bbox_inches="tight", pad_inches=0.009)

            # plt.show()
            plt.close()