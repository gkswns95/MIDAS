import os
import sys
from collections import Counter
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerLine2D

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datatools.trace_helper import TraceHelper

import datatools.matplotsoccer as mps
from models.utils import calc_pos_error, get_dataset_config, reshape_tensor

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

# DashedLineHandler 클래스 추가
class DashedLineHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        legline, = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        legline.set_dashes([2, 1.5])  # 점선 스타일 설정
        return [legline]


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
        # if "midas" in rets.keys():
        if "dbhp" in rets.keys():
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
            pred_keys.update(
                {
                    "mask": "mask",
                    "target": "Target",
                    "dbhp": "DBHP",
                    # "lambdas": "lambdas"
                })
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
        elif self.mode == "integrated_visualization":
            # 통합 시각화에 필요한 키들 추가
            pred_keys.update({
                "mask": "mask",
                # "target": "Target",
                "dbhp": "DBHP",
                # "lambdas": "lambdas"
            })

        if self.mode == "pitch_control":
            pred_keys = {}
            pred_keys["target"] = "target"
            pred_keys["csdi"] = "csdi"
            pred_keys["linear"] = "linear"
            pred_keys["ball"] = "ball"

        return pred_keys, fs

    def plot_trajectories(self, seq):
        n_players, ps = get_dataset_config(self.dataset)
        
        # 결측 구간 길이 계산 및 분류를 위한 함수
        def get_missing_segment_length(mask_xy):
            missing_segments = []
            in_segment = False
            start_idx = None
            
            for i in range(len(mask_xy)):
                is_missing = (mask_xy[i] == 0)
                
                if is_missing and not in_segment:
                    in_segment = True
                    start_idx = i
                elif not is_missing and in_segment:
                    in_segment = False
                    missing_segments.append((start_idx, i - 1))
                    start_idx = None
            
            # 마지막 프레임에서 끝나는 경우 처리
            if in_segment:
                missing_segments.append((start_idx, len(mask_xy) - 1))
                
            # 결측 구간 길이 계산
            segment_lengths = [end_idx - start_idx for start_idx, end_idx in missing_segments]
            # 결측 구간이 없는 경우
            if not segment_lengths:
                return "No Missing"
                
            # 가장 긴 결측 구간의 길이로 분류
            max_length = max(segment_lengths)
            
            # 고정된 구간으로 분류
            if max_length <= 60:
                return "Short"
            elif max_length <= 127:
                return "Medium"
            else:
                return "Long"
        
        for i, (key, title) in enumerate(self.pred_keys.items()):
            if key == "mask":
                continue
        
            if key == "target":
                continue

            ax = self.fig.add_subplot(4, 2, i)
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
                c = ["orangered", "limegreen", "goldenrod", "royalblue", "violet", "red", "green", "blue", "purple", "pink"]
                ax.imshow(court, zorder=1, extent=[0, ps[0], ps[1], 0])

            w_traces = self.pred_dict[f"window_{key}"]
            w_target_traces = self.pred_dict[f"window_target"]
            w_mask = self.pred_dict["window_mask"] 
            
            for i, k in enumerate(range(n_players)):
                if k != 0 and k != 1 and k != 4:
                    print(f"k: {k}")
                    continue # debug
                p_xy = torch.cat([w_traces[:, 2 * k, None], w_traces[:, 2 * k + 1, None]], dim=-1)
                target_xy = torch.cat([w_target_traces[:, 2 * k, None], w_target_traces[:, 2 * k + 1, None]], dim=-1)
                mask_xy = torch.cat([w_mask[:, 2 * k, None], w_mask[:, 2 * k + 1, None]], dim=-1) 
                obs_idxs = np.where((w_mask[:, 2 * k] != 0) | (w_mask[:, 2 * k + 1] != 0))
                missing_idxs = np.setdiff1d(np.arange(w_traces.shape[0]), obs_idxs[0])
                
                if key != "target" and key != "lambdas":
                    ax.plot(
                        w_traces[missing_idxs, 2 * k],
                        w_traces[missing_idxs, 2 * k + 1],
                        color=c[i],
                        linewidth=0.7,
                        alpha=1,
                        zorder=3,
                        linestyle="--",
                    )
                elif key == "target":
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

                if k == 1: # debug
                    ax.scatter(
                        w_traces[15, 2 * k], 
                        w_traces[15, 2 * k + 1], 
                        marker="o", 
                        edgecolor="black",
                        c=c[i],
                        linewidths=0.5,
                        s=10, 
                        alpha=1, 
                        zorder=2,
                    )  # starting point
                else:
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

                if k == 1: # debug
                    box = Rectangle(
                        xy=(w_traces[15, 2 * k] - width / 2, w_traces[15, 2 * k + 1] - height / 2),
                        width=width,
                        height=height,
                        facecolor="none",
                        edgecolor="black",
                        alpha=1.0,
                        zorder=4,
                        linewidth=0.3,
                    )
                else:
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

                # 결측 구간 길이에 따른 분류
                segment_class = get_missing_segment_length(mask_xy[:, 0].numpy())

                if k == 1: # debug
                    ax.text(
                        w_traces[15, 2 * k] + 0.5,
                        w_traces[15, 2 * k + 1] + 1 / 2 + 0.4,
                        segment_class,
                        color="black", 
                        fontsize=6, 
                        ha="center",
                        va="bottom", 
                        alpha=0.9,
                        zorder=5 
                    )
                else:
                    ax.text(
                        w_traces[0, 2 * k] + 0.5,
                        w_traces[0, 2 * k + 1] + 1 / 2 + 0.4,
                        segment_class,
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
                ### debug ###
                # === 선수별 개별 figure 저장 ===
                # if key == "dbhp":
                #     os.makedirs(f"individual_players/epi_{self.epi_idx}", exist_ok=True)
                    
                #     # 궤적 데이터 추출
                #     x = w_traces[:, 2 * k].numpy()
                #     y = w_traces[:, 2 * k + 1].numpy()
                #     target_x = w_target_traces[:, 2 * k].numpy()
                #     target_y = w_target_traces[:, 2 * k + 1].numpy()
                #     mask_x = w_mask[:, 2 * k].numpy()
                #     mask_y = w_mask[:, 2 * k + 1].numpy()
                    
                #     # 결측 구간 찾기
                #     obs_idxs = np.where((mask_x != 0) | (mask_y != 0))[0]
                #     missing_idxs = np.setdiff1d(np.arange(len(x)), obs_idxs)
                    
                #     if len(missing_idxs) == 0:
                #         continue  # 결측 구간이 없으면 건너뛰기
                    
                #     # 결측 구간의 시작점과 끝점 찾기
                #     missing_start_idx = missing_idxs[0]
                #     missing_end_idx = missing_idxs[-1]
                    
                #     # 먼저 두 그림을 위한 하나의 figure 생성 - 여백을 더 확보하기 위해 가로 크기 증가
                #     fig_combined = plt.figure(figsize=(2.6, 1.1), dpi=300, facecolor='white')

                #     # 왼쪽 - 궤적 그래프를 위한 axes (위치와 크기 조정)
                #     ax_traj = fig_combined.add_axes([0.02, 0.05, 0.4, 0.9])  # 왼쪽 부분에 위치

                #     # 궤적 그래프 그리기
                #     ax_traj.set_facecolor('white')
                #     gt_line = ax_traj.plot(target_x, target_y, color=c[i], linewidth=0.7, alpha=0.5,
                #                         label='Ground Truth', markersize=0)[0]
                #     imp_line = ax_traj.plot(x[missing_idxs], y[missing_idxs], 
                #                         color=c[i], linewidth=1.2, linestyle='--', 
                #                         dashes=[2, 1.5], label='Imputed')[0]
                #     ax_traj.scatter(x[0], y[0], color=c[i], s=10, marker='o', edgecolor="black", linewidths=0.5)

                #     # 결측 구간의 시작점과 끝점에 텍스트 추가
                #     if len(missing_idxs) > 0:
                #         # 시작점 텍스트
                #         start_x, start_y = x[missing_idxs[0]], y[missing_idxs[0]]
                #         ax_traj.text(start_x, start_y, "start", fontsize=6, ha='center', va='bottom',
                #                     color='black')
                        
                #         # 끝점 텍스트
                #         end_x, end_y = x[missing_idxs[-1]], y[missing_idxs[-1]]
                #         ax_traj.text(end_x-0.30, end_y, "end", fontsize=6, ha='center', va='bottom',
                #                     color='black')

                #     # 축 숨기기 및 범위 설정
                #     ax_traj.axis('off')
                #     x_range = max(x) - min(x)
                #     y_range = max(y) - min(y)
                #     x_center = (max(x) + min(x)) / 2
                #     y_center = (max(y) + min(y)) / 2
                #     max_range = max(x_range, y_range) * 1.1
                #     ax_traj.set_xlim(x_center - max_range/2, x_center + max_range/2)
                #     ax_traj.set_ylim(y_center - max_range/2, y_center + max_range/2)

                #     # 궤적 그래프에 테두리 추가
                #     border = plt.Rectangle((ax_traj.get_xlim()[0], ax_traj.get_ylim()[0]), 
                #                         ax_traj.get_xlim()[1] - ax_traj.get_xlim()[0],
                #                         ax_traj.get_ylim()[1] - ax_traj.get_ylim()[0],
                #                         fill=False, edgecolor='black', linewidth=1.5)
                #     ax_traj.add_patch(border)

                #     # 범례 추가
                #     legend = ax_traj.legend(handles=[gt_line, imp_line], 
                #                         labels=['Ground Truth', 'Imputed'],
                #                         loc='upper right', fontsize=6, 
                #                         frameon=False, handlelength=1.9,
                #                         handletextpad=0.5, borderaxespad=0.1,
                #                         handler_map={imp_line: DashedLineHandler(), gt_line: HandlerLine2D(numpoints=1)})

                #     # 오른쪽 - 가중치 히트맵을 위한 axes (위치 조정하여 더 많은 여백 확보)
                #     if "window_lambdas" in self.pred_dict:
                #         lambdas = self.pred_dict["window_lambdas"]  # [time, players*3]
                        
                #         if len(missing_idxs) > 0:
                #             # 결측 구간에 대한 가중치 추출
                #             missing_start_idx = missing_idxs[0]
                #             missing_end_idx = missing_idxs[-1]
                #             segment_length = len(missing_idxs)
                            
                #             # 3개 방법에 대한 결측 구간의 가중치 추출
                #             dp_weights = lambdas[missing_idxs, k*3].numpy()
                #             dapf_weights = lambdas[missing_idxs, k*3 + 1].numpy()
                #             dapb_weights = lambdas[missing_idxs, k*3 + 2].numpy()
                            
                #             # 데이터를 2D 행렬로 준비하고 행 사이에 NaN 행 추가하여 간격 생성
                #             nan_row = np.full_like(dp_weights, np.nan)
                #             weights_matrix = np.stack([
                #                 dp_weights,     # IP
                #                 nan_row,        # 간격
                #                 dapf_weights,   # DAP-F
                #                 nan_row,        # 간격    
                #                 dapb_weights    # DAP-B
                #             ])
                            
                #             # 히트맵을 위한 axes 추가 (오른쪽에 위치, 여백 증가)
                #             ax_heatmap = fig_combined.add_axes([0.58, 0.05, 0.38, 0.9])
                            
                #             # 히트맵 그리기
                #             sns.heatmap(weights_matrix, cmap="viridis", 
                #                     cbar=True, 
                #                     xticklabels=False, 
                #                     yticklabels=False,
                #                     vmin=0, vmax=1, ax=ax_heatmap,
                #                     mask=np.isnan(weights_matrix))
                            
                #             # 틱 위치 및 레이블 설정
                #             ax_heatmap.set_yticks([0.5, 2.5, 4.5])
                #             ax_heatmap.set_yticklabels(["IP", "DAP-F", "DAP-B"], fontsize=7)
                            
                #             # x축 레이블 설정
                #             ax_heatmap.set_xticks([0, segment_length-1])
                #             ax_heatmap.set_xticklabels(["start", "end"], fontsize=7)
                            
                #             # 컬러바 설정
                #             cbar = ax_heatmap.collections[0].colorbar
                #             cbar.ax.tick_params(labelsize=7)

                #             # 저장
                #             segment_class = get_missing_segment_length(np.stack([mask_x, mask_y], axis=1)[:, 0])
                #             fig_combined.savefig(f"individual_players/epi_{self.epi_idx}/seq_{seq}_player_{k}_{segment_class.lower()}_combined.png", 
                #                             bbox_inches="tight", pad_inches=0.02, transparent=False)
                #             fig_combined.savefig(f"individual_players/epi_{self.epi_idx}/seq_{seq}_player_{k}_{segment_class.lower()}_combined.pdf", 
                #                             bbox_inches="tight", pad_inches=0.02, transparent=False)
                #             plt.close(fig_combined)
            
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

        for i, title in enumerate(["IP", "DAP-F", "DAP-B"]):
            if title == "IP":
                heatmap = sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=True, mask=m, ax=axes[i], vmin=0, vmax=1)
            else:
                heatmap = sns.heatmap(lambdas[:, i::3], cmap="viridis", cbar=False, mask=m, ax=axes[i])

            axes[i].set_xlabel("Agent", fontsize=40)
            if title == "DAP-B":
                axes[i].set_ylabel("Time step", fontsize=40)
            
            axes[i].set_title(title.format(i + 1), fontsize=45, pad=20)
            axes[i].set_xticks([p for p in range(22) if p % 4 == 0])
            axes[i].set_xticklabels([p for p in range(22) if p % 4 == 0], fontsize=30)
            axes[i].set_yticks([s for s in range(200) if s % 40 == 0])
            axes[i].set_yticklabels([s for s in range(200) if s % 40 == 0], fontsize=30)
            # axes[i].set_yticks([0, 50, 100, 150])
            # axes[i].set_yticklabels([0, 50, 100, 150], fontsize=30)

            if title == "DAP-F":  
                colorbar = heatmap.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=25)

        return f

    def plot_hybrid_weights_per_player(self, save_dir="hybrid_weights_per_player"):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        
        n_players, _ = get_dataset_config(self.dataset)
        os.makedirs(save_dir, exist_ok=True)

        mask = self.pred_dict["window_mask"]
        lambdas = self.pred_dict["window_lambdas"]  # [time, players*3]
        
        method_names = ["DP", "DAP-F", "DAP-B"]

        for k in range(n_players):
            # 결측 구간 찾기
            mask_x = mask[:, 2 * k]
            missing_segments = []
            in_segment = False
            start_idx = None
            
            for idx in range(len(mask_x)):
                is_missing = (mask_x[idx] == 0)
                if is_missing and not in_segment:
                    in_segment = True
                    start_idx = idx
                elif not is_missing and in_segment:
                    in_segment = False
                    missing_segments.append((start_idx, idx - 1))
                    start_idx = None
                
            if in_segment:
                missing_segments.append((start_idx, len(mask_x) - 1))

            # 결측 구간이 없으면 건너뜀
            if not missing_segments:
                continue
            
            # 가장 긴 결측 구간 찾기
            seg_lens = [end - start + 1 for start, end in missing_segments]
            max_idx = np.argmax(seg_lens)
            start, end = missing_segments[max_idx]
            segment_length = end - start + 1
            
            # 가중치 추출 (해당 결측 구간만)
            segment_weights = np.zeros((3, segment_length))  # [방법, 시간]
            for i in range(3):  # DP, DAP-F, DAP-B
                segment_weights[i, :] = lambdas[start:end+1, k*3 + i]
            
            # 정사각형에 가까운 그래프 생성 (5x5 사이즈로 수정)
            fig, ax = plt.subplots(figsize=(5, 4.5))
            
            # 시각화: 각 방법을 행으로, 시간을 열로 표시 (y축 방향으로 더 많은 공간 활용)
            row_height = 0.6  # 행 높이 증가
            row_gap = 0.05    # 행 간 간격 감소
            
            for i, method in enumerate(method_names):
                # 각 행의 위치 계산
                row_pos = i * (row_height + row_gap)
                
                im = ax.imshow(segment_weights[i:i+1, :], 
                              aspect='auto', 
                              cmap='plasma',  # 학술적으로 적합한 컬러맵
                              vmin=0, vmax=1,
                              extent=[0, 1, row_pos+row_height, row_pos])
                
                # 방법 이름 표시
                ax.text(-0.09, row_pos + row_height/2, method, ha='right', va='center', 
                       fontsize=12, fontweight='bold')
                
            # x축 설정
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['missing start', 'missing end'], fontsize=12)
            ax.set_xlabel('Missing Segment Position', fontsize=14)
            
            # y축 설정
            ax.set_yticks([])  # y축 눈금 제거
            y_max = 3 * (row_height + row_gap)
            ax.set_ylim(0, y_max)
            
            # 컬러바 추가 및 설정
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Weight Value', fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
            # 제목 설정
            ax.set_title(f'Player {k} Hybrid Weights (Length: {segment_length})', fontsize=15)
            
            # 레이아웃 조정 및 저장
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, f"player_{k}_hybrid_weights.png"), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(save_dir, f"player_{k}_hybrid_weights.pdf"), bbox_inches='tight')
            plt.close(fig)
            
            # 텍스트 파일에 결측 구간 정보 저장
            with open(os.path.join(save_dir, f"player_{k}_missing_info.txt"), 'w') as f:
                f.write(f"Player {k} Missing Segments:\n")
                for i, (s, e) in enumerate(missing_segments):
                    length = e - s + 1
                    f.write(f"Segment {i+1}: Start={s}, End={e}, Length={length}\n")

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
        self.epi_idx = epi_idx
        e = self.val_epi_list[epi_idx]
        # path = f"plots/{self.dataset}/{self.mode}/episode_{e}"
        path = f"plots/{self.dataset}/{self.mode}/epi_{epi_idx}"
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
                self.plot_trajectories(seq)
            elif self.mode == "dist_heatmap":
                self.plot_dist_heatmap()
            elif self.mode == "weights_heatmap":
                # fig = self.plot_hybrid_weights()
                self.plot_hybrid_weights_per_player()
            elif self.mode == "pitch_control":
                self.plot_pitch_control()
            
            # Imputed Trajectories
            self.fig.savefig(f"{path}/seq_{seq}.png", bbox_inches="tight", dpi=300)
            self.fig.savefig(f"{path}/seq_{seq}.pdf", bbox_inches="tight", pad_inches=0.009)

            # plt.show()
            plt.close()