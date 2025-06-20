import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.naomi.utils import *
from models.utils import reshape_tensor
from set_transformer.model import SetTransformer


def num_trainable_params(model):
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.n_features = params["n_features"]  # number of features per each player
        self.team_size = params["team_size"]  # number of players per team
        # self.n_players = self.team_size if self.dataset == "afootball" else self.team_size * 2
        self.n_players = self.team_size if params["single_team"] else self.team_size * 2

        self.hidden_dim = params["discrim_rnn_dim"]
        self.y_dim = self.n_players * self.n_features

        self.action_dim = self.y_dim
        self.state_dim = self.y_dim
        self.gpu = params["cuda"]
        self.num_layers = params["discrim_num_layers"]

        self.gru = nn.GRU(self.state_dim, self.hidden_dim, self.num_layers)
        self.dense1 = nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim)
        self.dense2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, a, h=None):
        """
        x : [seq, batch, x_dim]
        a : [seq, batch, x_dim]
        """
        p, _ = self.gru(x, h)  # [seq, batch, x_dim]
        p = torch.cat([p, a], 2)  # [seq, batch, x_dim * 2]
        prob = F.sigmoid(self.dense2(F.relu(self.dense1(p))))  # [seq, batch, 1]
        return prob

    def init_hidden(self, batch):
        return Variable(torch.zeros(self.num_layers, batch, self.hidden_dim))


class NAOMIImputer(nn.Module):
    def __init__(self, params):
        super(NAOMIImputer, self).__init__()
        self.params = params

        self.dataset = params["dataset"]
        self.n_features = params["n_features"]  # number of features per each player
        self.team_size = params["team_size"]  # number of players per team
        if params["single_team"]:
            self.n_players = self.team_size
        else:
            self.n_players = self.team_size * 2
        self.stochastic = params["stochastic"]

        self.y_dim = self.n_players * self.n_features

        self.rnn_dim = params["rnn_dim"]
        self.n_layers = params["n_layers"]
        self.highest = params["n_highest"]
        self.batch_size = params["batch_size"]
        self.dims = {}
        self.networks = {}

        if self.params["agent_wise"]:
            self.gru = nn.GRU(self.n_features, self.rnn_dim, self.n_layers)
            self.back_gru = nn.GRU(self.n_features + 1, self.rnn_dim, self.n_layers)
            self.global_context_fc = nn.Sequential(
                nn.Linear(self.n_players * self.rnn_dim, self.rnn_dim), 
                nn.ReLU(),
                nn.Linear(self.rnn_dim, self.rnn_dim))
            self.back_global_context_fc = nn.Sequential(
                nn.Linear(self.n_players * self.rnn_dim, self.rnn_dim), 
                nn.ReLU(),
                nn.Linear(self.rnn_dim, self.rnn_dim))
        else:
            self.gru = nn.GRU(self.y_dim, self.rnn_dim, self.n_layers)
            self.back_gru = nn.GRU(self.y_dim + 1, self.rnn_dim, self.n_layers)

        step = 1
        while step <= self.highest:
            k = str(step)
            self.dims[k] = params["dec" + k + "_dim"]
            dim = self.dims[k]

            curr_level = {}
            curr_level["dec"] = nn.Sequential(nn.Linear(2 * self.rnn_dim, dim), nn.ReLU())

            if self.params["agent_wise"]:
                curr_level["mean"] = nn.Linear(dim, self.n_features)
                if self.stochastic:
                    curr_level["std"] = nn.Sequential(nn.Linear(dim, self.n_features), nn.Softplus())
            else:
                curr_level["mean"] = nn.Linear(dim, self.y_dim)
                if self.stochastic:
                    curr_level["std"] = nn.Sequential(nn.Linear(dim, self.y_dim), nn.Softplus())
            curr_level = nn.ModuleDict(curr_level)

            self.networks[k] = curr_level

            step = step * 2

        self.networks = nn.ModuleDict(self.networks)

        if params["ppe"] or params["fpe"] or params["fpi"]:  # use set transformer for PI/PE embedding
            self.pe_z_dim = params["pe_z_dim"]
            self.pi_z_dim = params["pi_z_dim"]

            in_dim = self.n_features
            if params["ppe"]:
                self.ppe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                in_dim += self.pe_z_dim
            if params["fpe"]:
                self.fpe_st = SetTransformer(self.n_features, self.pe_z_dim, embed_type="e")
                in_dim += self.pe_z_dim
            if params["fpi"]:
                self.fpi_st = SetTransformer(self.n_features, self.pi_z_dim, embed_type="i")
                in_dim += self.pi_z_dim

            # in_dim *= self.n_players
            self.set_transformer_fc = nn.Linear(in_dim, self.n_features)

    def forward(self, data, ground_truth):
        """
        data : [time, bs, 1 + feat_dim]
        ground_truth : [time, bs, feat_dim]
        """

        device = data.device
        seq_len = data.shape[0]
        bs = data.shape[1]

        h = torch.zeros(self.n_layers, bs, self.rnn_dim)
        h_back = torch.zeros(self.n_layers, bs, self.rnn_dim)

        if self.params["cuda"]:
            h, h_back = h.to(device), h_back.to(device)

        loss = 0.0
        pos_dist = 0.0
        count = 0
        imput_count = 0
        h_back_dict = {}

        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t + 1] = h_back
            state_t = data[t]
            _, h_back = self.back_gru(state_t.unsqueeze(0), h_back)

        for t in range(seq_len):
            state_t = ground_truth[t]
            _, h = self.gru(state_t.unsqueeze(0), h)
            count += 1
            for k, dim in self.dims.items():
                step_size = int(k)
                curr_level = self.networks[str(step_size)]
                if t + 2 * step_size <= seq_len:
                    next_t = ground_truth[t + step_size]
                    h_back = h_back_dict[t + 2 * step_size]

                    dec_t = curr_level["dec"](torch.cat([h[-1], h_back[-1]], 1))
                    dec_mean_t = curr_level["mean"](dec_t)  # [bs, y_dim]

                    if self.stochastic:
                        dec_std_t = curr_level["std"](dec_t)  # [bs, y_dim]
                        loss += nll_gauss(dec_mean_t, dec_std_t, next_t)
                        dec_mean_t = reparam_sample_gauss(dec_mean_t, dec_std_t)  # [bs, y_dim]
                    else:
                        loss += self.calc_mae_loss(dec_mean_t, next_t)

                    pred_t = reshape_tensor(
                        dec_mean_t, upscale=True, sports=self.dataset, single_team=self.params["single_team"]
                    )  # [bs, total_players, 2]
                    target_t = reshape_tensor(
                        next_t, upscale=True, sports=self.dataset, single_team=self.params["single_team"]
                    )
                    pos_dist += torch.norm(pred_t - target_t, dim=-1).mean()

                    imput_count += 1

        loss = loss / count / bs
        pos_dist = pos_dist / imput_count

        return loss, pos_dist.item()

    def sample(self, data_list):
        device = data_list[0].device
        self.bs = data_list[0].shape[1]
        seq_len = len(data_list)

        if self.params["ppe"] or self.params["fpe"] or self.params["fpi"]:
            x = torch.cat(data_list, dim=0) # [time, bs, x_dim + 1]
            has_value = x[..., 0].unsqueeze(-1)
            player_data = x[..., 1:]
            player_data = self.set_transformer_emb(player_data.transpose(0, 1)).transpose(0, 1) # [time, bs, x_dim]
            emb_x = torch.cat([has_value, player_data], dim=-1) # [time, bs, x_dim + 1]
            data_list = []
            for j in range(seq_len):
                data_list.append(emb_x[j : j + 1])

        if self.params["agent_wise"]:
            h = torch.zeros(self.params["n_layers"], self.bs * self.n_players, self.rnn_dim)
            h_back = torch.zeros(self.params["n_layers"], self.bs * self.n_players, self.rnn_dim)
        else:
            h = torch.zeros(self.params["n_layers"], self.bs, self.rnn_dim)
            h_back = torch.zeros(self.params["n_layers"], self.bs, self.rnn_dim)

        if self.params["cuda"]:
            h, h_back = h.to(device), h_back.to(device)

        h_back_dict = {}
        for t in range(seq_len - 1, 0, -1):
            state_t = data_list[t]  # [1, bs, 1+x_dim]
            if self.params["agent_wise"]:
                h_back_dict[t + 1] = h_back # [2, bs * players, rnn_dim]
                state_t = self.reshape_state(state_t, direct="b") # [1, bs * players, 1+feats]
                _, h_back = self.back_gru(state_t, h_back) # [2, bs * players, rnn_dim]
            else:
                h_back_dict[t + 1] = h_back  # [2, bs, rnn_dim]
                _, h_back = self.back_gru(state_t, h_back)

        curr_p = 0  # pivot 1
        state_t = data_list[curr_p][:, :, 1:] # [1, bs, x_dim]
        if self.params["agent_wise"]:
            state_t = self.reshape_state(state_t, direct="f") # [1, bs * players, feats]
            _, h = self.gru(state_t, h) #[2, bs * players, rnn_dim]
        else:
            _, h = self.gru(state_t, h) # [2, bs, rnn_dim]
        
        while curr_p < seq_len - 1:
            if data_list[curr_p + 1][0, 0, 0] == 1:  # indicator = 1 (observed_value)
                curr_p += 1
                state_t = data_list[curr_p][:, :, 1:]
                if self.params["agent_wise"]:
                    state_t = self.reshape_state(state_t, direct="f") # [1, bs * players, feats]
                    _, h = self.gru(state_t, h) #[2, bs * players, rnn_dim]
                else:
                    _, h = self.gru(state_t, h) # [2, bs, rnn_dim]
                
            else:  # indicator = 0 (missing_Value)
                next_p = curr_p + 1  # pivot 2
                while next_p < seq_len and data_list[next_p][0, 0, 0] == 0:
                    next_p += 1

                step_size = 1
                while curr_p + 2 * step_size <= next_p and step_size <= self.highest:
                    step_size *= 2
                step_size = step_size // 2

                self.interpolate(data_list, curr_p, h, h_back_dict, step_size, device=device)

        return torch.cat(data_list, dim=0)[:, :, 1:]  # [time, bs, feat_dim]

    def interpolate(self, data_list, curr_p, h, h_back_dict, step_size, p=0, device="cuda:0"):
        h_back = h_back_dict[curr_p + 2 * step_size]
        curr_level = self.networks[str(step_size)]

        dec_t = curr_level["dec"](torch.cat([h[-1], h_back[-1]], 1)) # [bs * players, dec_dim] if agent-wise else [bs, dec_dim]
        dec_mean_t = curr_level["mean"](dec_t) # [bs * players, feat_dim] if agent-wise else [bs, x_dim]
        if self.params["agent_wise"]:
            dec_mean_t = dec_mean_t.reshape(self.bs, -1) # [bs, x_dim]
            if self.stochastic:
                dec_std_t = curr_level["std"](dec_t).reshape(self.bs, -1) # [bs, x_dim]
                state_t = reparam_sample_gauss(dec_mean_t, dec_std_t) # [bs, x_dim]
            else:
                state_t = dec_mean_t
        else:
            if self.stochastic:
                dec_std_t = curr_level["std"](dec_t) # [bs, x_dim]
                state_t = reparam_sample_gauss(dec_mean_t, dec_std_t) # [bs, x_dim]
            else:
                state_t = dec_mean_t

        added_state = state_t.unsqueeze(0)  # [1, bs, x_dim]
        has_value = torch.ones(added_state.shape[0], added_state.shape[1], 1)
        if self.params["cuda"]:
            has_value = has_value.to(device)
        added_state = torch.cat([has_value, added_state], 2)
        if step_size > 1:
            right = curr_p + step_size
            left = curr_p + step_size // 2
            h_back = h_back_dict[right + 1]
            if self.params["agent_wise"]:
                added_state_ = self.reshape_state(added_state, direct="b") # [1, bs * players, 1+feats]
                _, h_back = self.back_gru(added_state_, h_back) # [2, bs * players, rnn_dim]
            else:
                _, h_back = self.back_gru(added_state, h_back) # [2, bs, rnn_dim]
            
            h_back_dict[right] = h_back

            zeros = torch.zeros(added_state.shape[0], added_state.shape[1], self.y_dim + 1) # [1, bs, 1+x_dim]
            if self.params["agent_wise"]:
                zeros = self.reshape_state(zeros, direct="b") # [1, bs * players, 1+feats]
            if self.params["cuda"]:
                zeros = zeros.to(device)
            for i in range(right - 1, left - 1, -1):
                _, h_back = self.back_gru(zeros, h_back)
                h_back_dict[i] = h_back

        data_list[curr_p + step_size] = added_state

    def calc_mae_loss(self, pred, target):
        """
        pred : [bs, feat_dim]
        target : [bs, feat_dim]
        """
        loss = 0.0

        if self.n_features == 2:
            feature_types = ["pos"]
            scale_factor = 1
        elif self.n_features == 4:
            feature_types = ["pos", "vel"]
            scale_factor = 10
        elif self.n_features == 6:
            if self.params["cartesian_accel"]:
                feature_types = ["pos", "vel", "cartesian_accel"]
            else:
                feature_types = ["pos", "vel", "speed", "accel"]
            scale_factor = 10

        for mode in feature_types:
            pred_ = reshape_tensor(
                pred, mode=mode, sports=self.dataset, single_team=self.params["single_team"]
            )  # [bs, total_players, -1]
            target_ = reshape_tensor(
                target, mode=mode, sports=self.dataset, single_team=self.params["single_team"]
            )

            mae_loss = torch.abs(pred_ - target_).mean()

            if mode in ["accel", "speed"]:
                loss += mae_loss * 0
            elif mode in ["pos"]:
                loss += mae_loss * scale_factor
            else:
                loss += mae_loss

        return loss

    def set_transformer_emb(self, x):
        '''
        x : [bs, time, x_dim]
        '''

        bs, seq_len = x.shape[:2]

        x = x.transpose(0, 1) # [bs, time, x_dim] to [time, bs, x_dim]

        team1_x = x[..., : self.n_features * self.team_size].reshape(-1, self.team_size, self.n_features)
        if self.params["single_team"]:
            x = team1_x  # [time * bs, players, feats]
        else:
            team2_x = x[..., self.n_features * self.team_size :].reshape(-1, self.team_size, self.n_features)
            x = torch.cat([team1_x, team2_x], 1)  # [time * bs, players, feats]

        input_list = [x]
        if self.params["ppe"]:
            team1_z = self.ppe_st(team1_x)
            if self.params["single_team"]:
                self.ppe_z = team1_z # [time * bs, players, pe_z]
            else:
                team2_z = self.ppe_st(team2_x) # [time * bs, team_size, pe_z]
                self.ppe_z = torch.cat([team1_z, team2_z], dim=1)  # [time * bs, players, pe_z]
            input_list += [self.ppe_z]
        if self.params["fpe"]:
            self.fpe_z = self.fpe_st(x)  # [time * bs, players, pe_z]
            input_list += [self.fpe_z]
        if self.params["fpi"]:
            self.fpi_z = self.fpi_st(x).unsqueeze(1).expand(-1, self.n_players, -1) # [time * bs, players, pi_z]
            input_list += [self.fpi_z]
        
        x_ = torch.cat(input_list, -1) # [time * bs, players, emb_dim]
        x_ = self.set_transformer_fc(x_) # [time * bs, players, -1]
        x_ = x_.reshape(seq_len, bs, -1).transpose(0, 1) # [bs, time, x_dim]

        return x_ 

    def reshape_state(self, state_t, direct="f"):
        '''
        state_t when direct "f": [1, bs, feat_dim]
        state_t when direct "b": [1, bs, 1+feat_dim]
        '''

        bs = state_t.shape[1]

        if direct == "b":
            has_t = state_t[..., 0].unsqueeze(-1) # [1, bs, 1]
            x = state_t[..., 1:] # [1, bs, feat_dim]

            x = x.reshape(1, bs, self.n_players, -1).flatten(1, 2) # [1, bs * players, feats]
            has_t = has_t.unsqueeze(2).repeat(1, 1, self.n_players, 1).flatten(1, 2) # [1, bs * players, 1]

            state_t = torch.cat([has_t, x], dim=-1) # [1, bs * players, 1 + feats]
        else: # e.g. direc == "f"
            x = state_t # [1, bs, feat_dim]
            state_t = x.reshape(1, bs, self.n_players, -1).flatten(1, 2) # [1, bs * players, feats]

        return state_t