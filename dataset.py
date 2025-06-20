import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SportsDataset(Dataset):
    def __init__(
        self,
        sports="soccer",  # ["soccer", "basketball", "afootball"]
        data_paths=None,
        window_size=200,
        min_episode_size=100,
        stride=0,
        normalize=False,
        flip_pitch=False,
        single_team=False,
    ):
        random.seed(1000)

        self.sports = sports
        if sports == "soccer":
            self.ps = (108, 72)
            self.team_size = 11  # number of input players per team
        elif sports == "basketball":
            self.ps = (28.65, 15.24)
            self.team_size = 5  # number of input players per team
        else:  # sports == "afootball"
            self.ps = (110, 49)
            self.team_size = 6  # number of input players per team
            window_size = 50
            min_episode_size = 50

        self.feature_types = ["_x", "_y", "_vx", "_vy", "_ax", "_ay"]
        self.n_features = len(self.feature_types)
        self.window_size = window_size
        stride = window_size if stride == 0 else stride

        self.flip_pitch = flip_pitch

        self.pad_value = -100
        halfline_x = 0.5 if normalize else self.ps[0] / 2

        assert data_paths is not None

        player_data_list = []
        ball_data_list = []

        for f in tqdm(data_paths, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            match_traces = pd.read_csv(f, header=0)

            if normalize:
                x_cols = [c for c in match_traces.columns if c.endswith("_x")]
                y_cols = [c for c in match_traces.columns if c.endswith("_y")]
                match_traces[x_cols] /= self.ps[0]
                match_traces[y_cols] /= self.ps[1]

            if sports == "soccer":
                player_cols = [c for c in match_traces.columns if c[0] in ["A", "B"] and c[3:] in self.feature_types]

                for phase in match_traces["phase"].unique():
                    if type(phase) == str:  # For GPS-event traces, ignore phases with n_players < 22
                        phase_tuple = [int(i) for i in phase[1:-1].split(",")]
                        if phase_tuple[0] < 0 or phase_tuple[1] < 0:
                            continue
                    
                    phase_traces = match_traces[match_traces["phase"] == phase]

                    team1_gk, team2_gk = SportsDataset.detect_goalkeepers(phase_traces, halfline_x)
                    team1_code, team2_code = team1_gk[0], team2_gk[0]

                    input_cols = phase_traces[player_cols].dropna(axis=1).columns
                    team1_cols = [c for c in input_cols if c[0] == team1_code]
                    team2_cols = [c for c in input_cols if c[0] == team2_code]
                    
                    if min(len(team1_cols), len(team2_cols)) < self.n_features * self.team_size:
                        continue

                    # Reorder teams so that the left team comes first
                    input_cols = team1_cols + team2_cols

                    episodes = [e for e in phase_traces["episode"].unique() if e > 0]
                    for e in episodes:
                        ep_traces = match_traces[match_traces["episode"] == e]
                        ep_size = len(ep_traces)

                        ep_pdata = ep_traces[input_cols].values
                        ep_bdata = ep_traces[["ball_x", "ball_y"]].values

                        if ep_size > min_episode_size and ep_size < self.window_size:
                            pad_len = self.window_size - ep_size
                            ep_pdata = np.pad(ep_pdata, ((0, pad_len), (0, 0)), constant_values=self.pad_value)
                            ep_bdata = np.pad(ep_bdata, ((0, pad_len), (0, 0)), constant_values=self.pad_value)
                            player_data_list.append(ep_pdata)
                            ball_data_list.append(ep_bdata)

                        elif ep_size >= self.window_size:
                            for i in range(ep_size - self.window_size + 1):
                                if i % stride == 0 or i == ep_size - self.window_size:
                                    player_data_list.append(ep_pdata[i : i + self.window_size])
                                    ball_data_list.append(ep_bdata[i : i + self.window_size])

            else:
                if sports == "basketball":
                    if single_team:
                        player_cols = [f"player{i}{x}" for i in range(self.team_size) for x in self.feature_types]
                    else:
                        player_cols = [f"player{i}{x}" for i in range(self.team_size * 2) for x in self.feature_types]
                else:  # if sports == "afootball"
                    player_cols = [f"player{i}{x}" for i in range(self.team_size) for x in self.feature_types]

                episodes = [e for e in match_traces["episode"].unique() if e > 0]
                for e in episodes:
                    ep_pdata = match_traces[match_traces["episode"] == e][player_cols]
                    ep_size = len(ep_pdata)

                    if ep_size > min_episode_size and ep_size < self.window_size:
                        pad_len = self.window_size - ep_size
                        ep_pdata = np.pad(ep_pdata, ((0, pad_len), (0, 0)), constant_values=self.pad_value)
                        player_data_list.append(ep_pdata)

                    elif ep_size >= self.window_size:
                        for i in range(ep_size - self.window_size + 1):
                            if i % stride == 0 or i == ep_size - self.window_size:
                                player_data_list.append(ep_pdata[i : i + self.window_size])

        player_data = np.stack(player_data_list, axis=0) if player_data_list else np.array([])
        ball_data = np.stack(ball_data_list, axis=0) if ball_data_list else np.array([])  # only for soccer dataset

        if normalize:
            self.ps = (1, 1)

        self.player_data = torch.FloatTensor(player_data)  # [bs, time, players * feats]
        self.ball_data = torch.FloatTensor(ball_data)  # [bs, time, 2], only for soccer dataset

    def __getitem__(self, i):
        if self.sports != "basketball" and self.flip_pitch:
            player_data = np.copy(self.player_data[i])
            is_pad = torch.FloatTensor((player_data[..., :1] == self.pad_value))  # [time, 1]

            flip_x = np.random.choice(2, (1, 1))
            flip_y = np.random.choice(2, (1, 1))

            # (ref, mul) = (ps, -1) if flip == 1 else (0, 1)
            ref_x = flip_x * self.ps[0]
            ref_y = flip_y * self.ps[1]
            sign_x = 1 - flip_x * 2
            sign_y = 1 - flip_y * 2

            # flip (x, y) locations
            player_data[..., 0 :: self.n_features] = player_data[..., 0 :: self.n_features] * sign_x + ref_x
            player_data[..., 1 :: self.n_features] = player_data[..., 1 :: self.n_features] * sign_y + ref_y

            # flip (x, y) velocity and acceleration values
            if self.n_features > 2:
                for i in range(2, self.n_features):
                    sign = sign_x if i % 2 == 0 else sign_y
                    player_data[..., i :: self.n_features] = player_data[..., i :: self.n_features] * sign

            # if flip_x == 1, reorder team1 and team2 features
            team1_input = player_data[..., : self.n_features * self.team_size]
            team2_input = player_data[..., self.n_features * self.team_size :]

            input_permuted = np.concatenate([team2_input, team1_input], -1)
            player_data = torch.FloatTensor(np.where(flip_x, input_permuted, player_data))

            player_data = (1 - is_pad) * player_data + is_pad * self.pad_value
            ball_data = (1 - is_pad) * player_data + is_pad * self.pad_value

            if self.sports == "soccer":
                ball_data = np.copy(self.ball_data[i])
                ball_data[..., [0]] = ball_data[..., [0]] * sign_x + ref_x
                ball_data[..., [1]] = ball_data[..., [1]] * sign_y + ref_y
                return player_data, torch.FloatTensor(ball_data)
            else:
                return player_data

        else:
            return self.player_data[i], self.ball_data[i] if self.sports == "soccer" else self.player_data[i]

    def __len__(self):
        return len(self.player_data)

    @staticmethod
    def detect_goalkeepers(traces: pd.DataFrame, halfline_x=54):
        a_x_cols = [c for c in traces.columns if c.startswith("A") and c.endswith("_x")]
        b_x_cols = [c for c in traces.columns if c.startswith("B") and c.endswith("_x")]

        a_gk = (traces[a_x_cols].mean() - halfline_x).abs().idxmax()[:3]
        b_gk = (traces[b_x_cols].mean() - halfline_x).abs().idxmax()[:3]

        a_gk_mean_x = traces[f"{a_gk}_x"].mean()
        b_gk_mean_y = traces[f"{b_gk}_x"].mean()

        return (a_gk, b_gk) if a_gk_mean_x < b_gk_mean_y else (b_gk, a_gk)
