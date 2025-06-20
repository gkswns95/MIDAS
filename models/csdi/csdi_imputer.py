import numpy as np
import torch
import torch.nn as nn

from typing import Dict
from models.csdi.diff_models import diff_CSDI
from set_transformer.model import SetTransformer

class CSDIImputer(nn.Module):
    def __init__(self, params, device):
        super(CSDIImputer, self).__init__()
        self.params = params
        self.device = device

        self.dataset = params["dataset"]
        self.n_features = params["n_features"] # number of features per each player
        self.team_size = params["team_size"] # number of players per team
        if params["single_team"]:
            self.n_players = self.team_size
        else:
            self.n_players = self.team_size * 2
        # self.n_players = self.team_size if self.dataset == "afootball" else self.team_size * 2
        self.target_dim = self.n_players * self.n_features

        self.emb_time_dim = params["timeemb_dim"]
        self.emb_feature_dim = params["featureemb_dim"]
        self.is_unconditional = params["is_unconditional"] # False

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        params["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(params, input_dim)

        # parameters for diffusion models
        self.num_steps = params["n_steps"]

        beta_start = 0.0001
        beta_end = 0.5
        self.beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_steps) ** 2

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

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

            # self.set_transformer_fc = nn.Linear(in_dim, self.n_features)
            self.set_transformer_fc = nn.Sequential(
                nn.Linear(in_dim, self.pi_z_dim), 
                nn.ReLU(),
                nn.Linear(self.pi_z_dim, self.n_features))
    
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
        x_ = self.set_transformer_fc(x_) # [time * bs, players, feats]
        x_ = x_.flatten(1, 2) # [time * bs, players * feats(=x_dim)]
        x_ = x_.reshape(seq_len, bs, -1).transpose(0, 1) # [bs, time, x_dim]

        # x_ = torch.cat(input_list, -1).reshape(seq_len, bs, -1).transpose(0, 1) # [bs, time, players * emb_dim]
        # x_ = self.set_transformer_fc(x_) # [bs, time, x_dim]
    
        return x_
        
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, 
        observed_data, 
        cond_mask, 
        observed_mask, 
        side_info, 
        is_train, 
        set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = (1 - observed_mask)
        residual = (noise - predicted) * target_mask

        # target_mask = observed_mask - cond_mask
        # residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            # cond_obs = (cond_mask * observed_data).unsqueeze(1) # [bs, 1, x_dim, time]
            cond_obs = (cond_mask * observed_data) # [bs, x_dim, time]
            if self.params["ppe"] or self.params["fpe"] or self.params["fpi"]:  # use set transformer for PI/PE embedding
                cond_obs = self.set_transformer_emb(cond_obs.transpose(1, 2)).transpose(1, 2).unsqueeze(1) # [bs, 1, x_dim, time]

            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    # cond_obs = (cond_mask * observed_data) # [bs, x_dim, time]
                    cond_obs = (cond_mask * observed_data).unsqueeze(1) # [bs, 1, x_dim, time]
                    if self.params["ppe"] or self.params["fpe"] or self.params["fpi"]:  # use set transformer for PI/PE embedding
                        cond_obs = self.set_transformer_emb(cond_obs.transpose(1, 2)).transpose(1, 2).unsqueeze(1) # [bs, 1, x_dim, time]
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, ret: Dict[str, torch.Tensor], is_train=1, device="cuda:0") -> Dict[str, torch.Tensor] :
        input = ret["input"]
        target = ret["target"]
        mask = ret["mask"]
        timepoints = ret["timepoints"] # [bs, time]

        bs, seq_len = input.shape[:2]

        input = input.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) # [bs, time, players * feats]
        target = target.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) 
        mask = mask.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)

        input = input.transpose(1,2) # [bs, players * feats, time]
        target = target.transpose(1,2)
        mask = mask.transpose(1,2)
    
        side_info = self.get_side_info(timepoints, cond_mask=mask) # [bs, -1, players * feats, time]

        loss_func = self.calc_loss if is_train else self.calc_loss_valid

        loss = loss_func(input, cond_mask=mask, observed_mask=mask, side_info=side_info, is_train=is_train)

        ret.update({
            "total_loss": loss,
            "target": target.transpose(1,2),
            "input" : input.transpose(1,2),
            "mask" : mask.transpose(1,2)})

        return ret

    def evaluate(self, ret: Dict[str, torch.Tensor], n_samples=1) -> Dict[str, torch.Tensor]:
        input = ret["input"]
        target = ret["target"]
        mask = ret["mask"]
        timepoints = ret["timepoints"] # [bs, time]

        bs, seq_len = input.shape[:2]

        input = input.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) # [bs, time, players * feats]
        target = target.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3) 
        mask = mask.reshape(bs, seq_len, self.n_players, -1)[..., : self.n_features].flatten(2, 3)

        input = input.transpose(1,2) # [bs, players * feats, time]
        target = target.transpose(1,2)
        mask = mask.transpose(1,2)

        with torch.no_grad():
            side_info = self.get_side_info(timepoints, cond_mask=mask)

            samples = self.impute(input, mask, side_info, n_samples) # [bs, n_samples, players * feats, time]
            
            out = samples.squeeze(1)

        pred = mask * target + (1 - mask) * out

        ret.update({
            "pred" : pred.transpose(1,2),
            "target": target.transpose(1,2),
            "input" : input.transpose(1,2),
            "mask" : mask.transpose(1,2)})

        return ret