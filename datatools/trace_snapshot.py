import math
import os
import sys
from typing import Dict, Optional, Tuple, List, Union

import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import datatools.matplotsoccer as mps
from models.utils import compute_camera_coverage


class TraceSnapshot:
    def __init__(
        self,
        match_ret: Dict[str, pd.DataFrame] = None,
        mask: pd.DataFrame = None,
        show_times=True,
        show_episodes=False,
        show_events=False,
        show_frames=False,
        show_errors=False,
        show_polygon=False,
        annot_cols=None,
        rotate_pitch=False,
        anonymize=False,
        play_speed=1,
        legend_position='top',  # 'top', 'bottom', 'bottom_horizontal', 'right', 'bottom_right'
    ):
        self.trace_dict = match_ret
        self.mask = mask

        self.show_times = show_times
        self.show_episodes = show_episodes
        self.show_events = show_events
        self.show_frames = show_frames
        self.show_errors = show_errors
        self.show_polygon = show_polygon

        self.annot_cols = annot_cols
        self.rotate_pitch = rotate_pitch
        self.anonymize = anonymize
        self.play_speed = play_speed
        self.legend_position = legend_position
        
        self.imputed_style = {
            "color": None,  
            "linestyle": ":",
            "linewidth": 2,
            "alpha": 1.0,
            "marker": "o",
            "markersize": 8
        }
    
        self._preprocess_data()
        
    def _preprocess_data(self):
        for key in self.trace_dict.keys():
            traces = self.trace_dict[key].iloc[(self.play_speed - 1)::self.play_speed].copy()
            traces = traces.dropna(axis=1, how="all").reset_index(drop=True)
            
            xy_cols = [c for c in traces.columns if c.endswith("_x") or c.endswith("_y")]
            
            if self.rotate_pitch:
                traces[xy_cols[0::2]] = 108 - traces[xy_cols[0::2]]
                traces[xy_cols[1::2]] = 72 - traces[xy_cols[1::2]]
                
            self.trace_dict[key] = traces
            
        if self.mask is not None:
            self.mask = self.mask.iloc[(self.play_speed - 1)::self.play_speed].copy()
            self.mask = self.mask.dropna(axis=1, how="all").reset_index(drop=True)
    
    @staticmethod
    def calc_trace_dist(pred, target):
        pred_np = pred.to_numpy()[None, :]
        target_np = target.to_numpy()[None, :]
        return str(round(np.linalg.norm(pred_np - target_np, axis=1)[0], 4)) + "m"
    
    def create_snapshot(self, frame_idx: int, save_path: Optional[str] = None, 
                       figsize: Tuple[float, float] = (10.4, 7.2),
                       history_frames: int = 60, ball_history: int = 20) -> Tuple:
        fig, ax = plt.subplots(figsize=figsize)
        mps.field("white", fig, ax, show=False)
    
        main_traces = self.trace_dict["main"]
        t = min(frame_idx, len(main_traces) - 1)
    
        for trace_key in self.trace_dict.keys():
            traces = self.trace_dict[trace_key]
        
            alpha = 1.0 if trace_key == "main" else 0.5
            
            xy_cols = [c for c in traces.columns if c.endswith("_x") or c.endswith("_y")]
            team1_traces = traces[[c for c in xy_cols if c.startswith("A")]]
            team2_traces = traces[[c for c in xy_cols if c.startswith("B")]]
            
            if trace_key == "main":
                trace_info = [(team1_traces, "tab:red"), (team2_traces, "tab:blue")] 
            else:
                trace_info = [(team1_traces, "magenta"), (team2_traces, "darkcyan")]
                
            for team_traces, color in trace_info:
                if len(team_traces.columns) == 0:   
                    continue
                    
                players = [c[:-2] for c in team_traces.columns[0::2]]
                
                for i, p in enumerate(players):
                    if t not in traces.index or pd.isna(traces.loc[t, f"{p}_x"]):
                        continue
                        
                    x_pos = traces.loc[t, f"{p}_x"]
                    y_pos = traces.loc[t, f"{p}_y"]
                    
                    if trace_key == "imputed":
                        marker_color = color
                        marker_alpha = 0.5
                        marker_size = 350
                        marker_edgecolor = color
                        marker_style = 'o'
                        marker_zorder = 3
                        text_zorder = 4
                    else: # e.g., "main"
                        marker_color = color
                        marker_alpha = alpha
                        marker_size = 350
                        marker_edgecolor = color
                        marker_style = 'o'
                        marker_zorder = 5
                        text_zorder = 6
                        
                        # if trace_key == "main" and f"{p}_x" in self.mask.columns:
                        #     if t in self.mask.index and self.mask.loc[t, f"{p}_x"] == 0:
                        #         marker_alpha = 0.5
                    
                    ax.scatter(x_pos, y_pos, s=marker_size, c=marker_color,
                              alpha=marker_alpha, zorder=marker_zorder, 
                              edgecolors=marker_edgecolor, linewidths=1.5,
                              marker=marker_style)
                    
                    text_color = 'white'
                    player_num = int(p[1:]) if not self.anonymize else i + 1
                    ax.annotate(
                        player_num,
                        xy=(x_pos, y_pos),
                        ha="center", va="center", color=text_color,
                        fontsize=12, fontweight="bold", zorder=text_zorder
                    )
                    
                    # display trajectories (up to 'history_frames' frames in the past)
                    if trace_key == "main":
                        t_from = max(t - history_frames, 0)
                        
                        valid_segment = []
                        missing_segment = []
                        
                        for i in range(t_from, t):
                            if i >= len(traces) or i+1 >= len(traces):
                                continue
                                
                            x1, y1 = traces.loc[i, f"{p}_x"], traces.loc[i, f"{p}_y"]
                            x2, y2 = traces.loc[i+1, f"{p}_x"], traces.loc[i+1, f"{p}_y"]
                            
                            if self.mask.loc[i, f"{p}_x"] == 1:
                                if len(valid_segment) == 0:
                                    valid_segment = [[x1, y1], [x2, y2]]
                                else:
                                    valid_segment.append([x2, y2])
                            else:
                                if len(valid_segment) > 0:
                                    valid_x, valid_y = zip(*valid_segment)
                                    ax.plot(valid_x, valid_y, color=color, linestyle='-', 
                                            linewidth=1.5, zorder=2)
                                    valid_segment = []
                                
                                if len(missing_segment) == 0:
                                    missing_segment = [[x1, y1], [x2, y2]]
                                else:
                                    missing_segment.append([x2, y2])
                            
                            if i == t-1:
                                if len(valid_segment) > 0:
                                    valid_x, valid_y = zip(*valid_segment)
                                    ax.plot(valid_x, valid_y, color=color, linestyle='-', 
                                            linewidth=1.5, zorder=2)
                                
                                if len(missing_segment) > 0:
                                    missing_x, missing_y = zip(*missing_segment)
                                    ax.plot(missing_x, missing_y, color='black', linestyle='-',
                                            linewidth=1.5, alpha=0.3, zorder=1)
                    
                    elif trace_key == "imputed":
                        t_from = max(t - history_frames, 0)
                        
                        imputed_color = "magenta" if color == "magenta" else "darkcyan"
                        
                        segments = []
                        current_segment = []
                        
                        for i in range(t_from, t+1):
                            if i >= len(traces) or pd.isna(traces.loc[i, f"{p}_x"]):
                                continue
                                
                            if self.mask.loc[i, f"{p}_x"] == 0:
                                x_pos_i = traces.loc[i, f"{p}_x"]
                                y_pos_i = traces.loc[i, f"{p}_y"]
                                
                                if len(current_segment) == 0:
                                    current_segment = [[x_pos_i, y_pos_i]]
                                else:
                                    current_segment.append([x_pos_i, y_pos_i])
                            else:
                                if len(current_segment) > 0:
                                    segments.append(current_segment)
                                    current_segment = []
                        
                        if len(current_segment) > 0:
                            segments.append(current_segment)
                        
                        for segment in segments:
                            if len(segment) > 1:
                                x_vals, y_vals = zip(*segment)
                                ax.plot(
                                    x_vals, y_vals,
                                    color=imputed_color,
                                    linestyle=self.imputed_style["linestyle"],
                                    linewidth=2.0,
                                    alpha=self.imputed_style["alpha"],
                                    zorder=2.5
                                )
                    
                    if self.show_errors and trace_key != "main":
                        target = self.trace_dict["main"]
                        if t in target.index and f"{p}_x" in target.columns:
                            if not pd.isna(target.loc[t, f"{p}_x"]) and not pd.isna(x_pos):
                                dist = self.calc_trace_dist(
                                    traces.loc[t, [f"{p}_x", f"{p}_y"]], 
                                    target.loc[t, [f"{p}_x", f"{p}_y"]]
                                )
                                
                                rect = Rectangle(
                                    xy=(x_pos - 1.5, y_pos - 1.5),
                                    width=3, height=3,
                                    facecolor="none",
                                    edgecolor="limegreen" if trace_key == "pred" else "purple",
                                    linewidth=2, zorder=1
                                )
                                ax.add_patch(rect)
                                
                                ax.annotate(
                                    dist,
                                    xy=(x_pos - 3, y_pos + 2),
                                    color="dimgrey", fontsize=16, fontweight="bold", zorder=4
                                )
            
            if "ball_x" in traces.columns and "ball_y" in traces.columns:
                alpha = 0.5 if trace_key != "main" else 1.0
                
                if t in traces.index and not pd.isna(traces.loc[t, "ball_x"]) and not pd.isna(traces.loc[t, "ball_y"]):
                    ball_x = traces.loc[t, "ball_x"]
                    ball_y = traces.loc[t, "ball_y"]
                    
                    ax.scatter(ball_x, ball_y, s=150, c='white', edgecolors='black', 
                              linewidth=1.5, alpha=1.0, zorder=6)
                
                # t_from = max(t - history_frames, 0)
                # valid_idx = traces.loc[t_from:t, "ball_x"].notna() & traces.loc[t_from:t, "ball_y"].notna()
                # if valid_idx.any():
                #     ax.plot(
                #         traces.loc[t_from:t, "ball_x"][valid_idx], 
                #         traces.loc[t_from:t, "ball_y"][valid_idx],
                #         color='black', linewidth=1.5, alpha=alpha, zorder=2
                #     )
            
            if self.show_polygon and trace_key == "main" and "ball_x" in traces.columns:
                if t in traces.index and not pd.isna(traces.loc[t, "ball_x"]):
                    ball_loc = torch.tensor(np.array(traces.loc[t:t, ["ball_x", "ball_y"]]))
                    point = compute_camera_coverage(ball_loc[0])
                    ax.add_patch(patches.Polygon(point, color='limegreen', alpha=0.2, zorder=0))
        
        traces = self.trace_dict["main"]
        if self.show_times and "time" in traces.columns and t in traces.index:
            time_val = traces.loc[t, "time"]
            time_text = f"{int(time_val // 60):02d}:{time_val % 60:04.1f}"
            ax.text(0, 73, time_text, fontsize=15, ha="left", va="bottom")
            
        if self.show_episodes and "episode" in traces.columns and t in traces.index:
            episode_val = traces.loc[t, "episode"]
            if episode_val != 0:
                ax.text(30, 73, f"Episode {episode_val}", fontsize=15, ha="left", va="bottom")
                
        if self.show_frames and "frame" in traces.columns and t in traces.index:
            ax.text(10, 73, f"Frame : {traces.loc[t, 'frame']}", fontsize=15, ha="left", va="bottom")
        
        if self.show_events and "event_type" in traces.columns and t in traces.index:
            event_text = ""
            if not pd.isna(traces.loc[t, "event_type"]):
                event_player = traces.loc[t, "event_player"] if "event_player" in traces.columns else ""
                event_text = f"{traces.loc[t, 'event_type']} by {event_player}"
            elif "event_types" in traces.columns and not pd.isna(traces.loc[t, "event_types"]):
                event_text = traces.loc[t, "event_types"]
                
            if event_text:
                ax.text(15, 73, event_text, fontsize=15, ha="left", va="bottom")
        
        team_a_marker = plt.Line2D([0], [0], color='tab:red', linestyle='-', 
                                    linewidth=1.5, label='Observed trajectories (home)')
        team_b_marker = plt.Line2D([0], [0], color='tab:blue', linestyle='-', 
                                    linewidth=1.5, label='Observed trajectories (away)')
        
        team_a_imputed = plt.Line2D([0], [0], color='magenta', linestyle=':', 
                                    linewidth=2.0, label='Imputed trajectories (home)')
        team_b_imputed = plt.Line2D([0], [0], color='darkcyan', linestyle=':', 
                                    linewidth=2.0, label='Imputed trajectories (away)')
        
        missing_marker = plt.Line2D([0], [0], color='black', linestyle='-', 
                                   linewidth=1.5, alpha=0.7, label='Target missing trajectories')
        
        legend_items = [team_a_marker, team_b_marker, team_a_imputed, team_b_imputed, missing_marker]
        
        if self.show_polygon:
            coverage_marker = patches.Patch(facecolor='limegreen', alpha=0.2, 
                                          edgecolor='limegreen', label='Camera coverage')
            legend_items.append(coverage_marker)
            
        legend_fontsize = 14.5
        if self.legend_position == 'top':
            ax.legend(handles=legend_items, loc='upper left', fontsize=legend_fontsize, framealpha=0.9)
        elif self.legend_position == 'right':
            ax.legend(handles=legend_items, loc='upper right', fontsize=legend_fontsize, framealpha=0.9)
        elif self.legend_position == 'bottom_horizontal':
            ax.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                     ncol=len(legend_items), fontsize=legend_fontsize, framealpha=0.9)
        elif self.legend_position == 'bottom_right':
            ax.legend(handles=legend_items, loc='lower right', fontsize=legend_fontsize, framealpha=0.9)
        else:  # 'bottom'
            ax.legend(handles=legend_items, loc='lower left', fontsize=legend_fontsize, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None, None
        else:
            return fig, ax
            
    def create_snapshots_range(self, start_frame: int, end_frame: int, 
                              step: int = 10, output_dir: str = "./snapshots"):
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx in range(start_frame, end_frame + 1, step):
            formatted_idx = f"{frame_idx:05d}"
            save_path = os.path.join(output_dir, f"frame_{formatted_idx}.png")
            
            self.create_snapshot(frame_idx, save_path=save_path)

    def create_cropped_snapshot(self, frame_idx: int, crop_area: Tuple[float, float, float, float],
                              save_path: Optional[str] = None, figsize: Tuple[float, float] = (10.4, 7.2),
                              history_frames: int = 60, ball_history: int = 20) -> Tuple:
        """
        Create a snapshot with cropped view of a specific area of the pitch
        
        Args:
            frame_idx: Frame index to visualize
            crop_area: Tuple of (x_min, y_min, x_max, y_max) defining the crop area
            save_path: Path to save the figure
            figsize: Figure size
            history_frames: Number of frames to show for player trajectory history
            ball_history: Number of frames to show for ball trajectory history
            
        Returns:
            Figure and axes objects
        """
        fig, ax = self.create_snapshot(frame_idx=frame_idx, save_path=None, 
                                     figsize=figsize, history_frames=history_frames,
                                     ball_history=ball_history)
        
        # Apply cropping by setting axis limits
        x_min, y_min, x_max, y_max = crop_area
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Adjust legend position if it would be outside the visible area
        if self.legend_position == 'top' and y_max < 72:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=14.5, framealpha=0.9)
        elif self.legend_position == 'bottom' and y_min > 0:
            ax.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=14.5, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None, None
        else:
            return fig, ax

    def save_cropped_snapshots_sequence(self, start_frame: int, end_frame: int, 
                                      crop_area: Tuple[float, float, float, float],
                                      step: int = 1, output_dir: str = "./cropped_snapshots"):
        """
        Save a sequence of cropped snapshots for a range of frames
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)
            crop_area: Tuple of (x_min, y_min, x_max, y_max) defining the crop area
            step: Frame step size
            output_dir: Directory to save snapshots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx in range(start_frame, end_frame + 1, step):
            formatted_idx = f"{frame_idx:05d}"
            save_path = os.path.join(output_dir, f"cropped_frame_{formatted_idx}.png")
            
            self.create_cropped_snapshot(frame_idx, crop_area, save_path=save_path) 