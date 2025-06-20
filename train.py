import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SportsDataset
from models import load_model
from models.nrtsi.utils import get_gap_lr_bs
from models.utils import get_params_str, load_pretrained_model, num_trainable_params

# Helper functions
def printlog(line):
    print(line)
    with open(save_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def loss_str(loss_dict: dict):
    ret = ""
    for k, v in loss_dict.items():
        if k.endswith("_loss") or k.endswith("_pe"):
            ret += f" {k}: {np.mean(v):.4f} |"
    return ret[1:-2]


def hyperparams_str(epoch, hp):
    ret = "\nEpoch {:d}".format(epoch)
    if hp["pretrain"]:
        ret += " (pretrain)"
    return ret


# For one epoch
def run_epoch(
    model: nn.DataParallel,
    loader: DataLoader,
    optimizer: torch.optim.Adam,
    train=True,
    print_every=50,
    train_args=None,
) -> Dict[str, float]:
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()
    n_batches = len(loader)

    loss_dict = {"missing_rate": [], "total_loss": [], "pred_pe": []}
    if model.module.params["model"] == "midas" and model.module.params["deriv_accum"]:
        loss_dict["dap_f_pe"] = []
        if model.module.params["missing_pattern"] != "forecast":
            loss_dict["dap_b_pe"] = []
        if model.module.params["dynamic_hybrid"]:
            loss_dict["hybrid_d_pe"] = []
    
    if model.module.params["model"] == "csdi":
        loss_dict = {"missing_rate": [], "total_loss": []}

    for batch_idx, data in enumerate(loader):
        if model.module.params["model"] in ["midas", "brits", "naomi", "graph_imputer", "csdi", "imputeformer"]:
            if train:
                ret = model(data, device=default_device)
            else:
                with torch.no_grad():
                    if model.module.params["model"] in ["midas", "brits", "naomi", "graph_imputer", "imputeformer"]:
                        ret = model(data, device=default_device)
                    elif model.module.params["model"] in ["csdi"]:
                        ret = model(data, mode="valid", device=default_device)
                    else:
                        ret = model.module.evaluate(data, device=default_device)

        if model.module.params["model"] == "nrtsi":
            data = data[:2]

            min_gap, max_gap, cur_lr, ta = train_args
            data.append(min_gap)
            data.append(max_gap)

            optimizer.param_groups[0]["lr"] = cur_lr
            if train:
                ret = model(data, model=model, teacher_forcing=ta, device=default_device)
            else:
                with torch.no_grad():
                    ret = model(data, model=model, teacher_forcing=ta, device=default_device)

        for k in loss_dict:
            loss_dict[k] += [ret[k].item()] if k == "total_loss" else [ret[k]]

        if train:
            loss = ret["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), clip)
            optimizer.step()

        if train and batch_idx % print_every == 0:
            print(f"[{batch_idx:>{len(str(n_batches))}d}/{n_batches}]  {loss_str(loss_dict)}")

    for key, value in loss_dict.items():
        loss_dict[key] = np.mean(value)

    return loss_dict


# Main starts here
parser = argparse.ArgumentParser()

parser.add_argument("-t", "--trial", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True, help="soccer, basketball, afootball")
parser.add_argument("--train_metrica", action="store_true", default=False, help="Use Metrica data for training")
parser.add_argument("--valid_metrica", action="store_true", default=False, help="Use Metrica data for validation")
parser.add_argument("--model", type=str, required=True, default="midas")
parser.add_argument("--missing_pattern", type=str, required=True, help="uniform, playerwise, camera, forecast, dynamic")
parser.add_argument("--missing_rate", type=float, required=False, default=0.5, help="proportion of missing frames")
parser.add_argument("--team_size", type=int, required=False, default=11, help="num players per team")
parser.add_argument("--single_team", action="store_true", default=False, help="only using players from one team")
parser.add_argument("--n_features", type=int, required=False, default=2, help="num features")
parser.add_argument("--window_size", type=int, required=False, default=200, help="for SportsDataset")
parser.add_argument("--window_stride", type=int, required=False, default=5, help="for SportsDataset")
parser.add_argument("--normalize", action="store_true", default=False, help="normalize the position scale")
parser.add_argument("--flip_pitch", action="store_true", default=False, help="augment data by flipping the pitch")
parser.add_argument("--player_order", type=str, required=False, default=None, help="shuffle, xy_sort")

parser.add_argument("--n_epochs", type=int, required=False, default=200, help="num epochs")
parser.add_argument("--batch_size", type=int, required=False, default=32, help="batch size")
parser.add_argument("--start_lr", type=float, required=False, default=0.0001, help="starting learning rate")
parser.add_argument("--min_lr", type=float, required=False, default=0.0001, help="minimum learning rate")
parser.add_argument("--clip", type=float, required=False, default=10, help="gradient clipping")
parser.add_argument("--print_every_batch", type=int, required=False, default=50, help="periodically print performance")
parser.add_argument("--save_every_epoch", type=int, required=False, default=10, help="periodically save model")
parser.add_argument("--pretrain_time", type=int, required=False, default=0, help="num epochs to train macro policy")
parser.add_argument("--seed", type=int, required=False, default=128, help="PyTorch random seed")
parser.add_argument("--cuda", action="store_true", default=False, help="use GPU")
parser.add_argument("--cont", action="store_true", default=False, help="continue training the previous bast model")
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best test loss")
parser.add_argument("--load_pretrained", type=int, default=0)
parser.add_argument("--freeze_pretrained", action="store_true", default=0)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    args.cuda = torch.cuda.is_available()
    default_device = "cuda:0"

    # Parameters to save
    params = {
        "dataset": args.dataset,
        "train_metrica": args.train_metrica,
        "valid_metrica": args.valid_metrica,
        "model": args.model,
        "missing_pattern": args.missing_pattern,
        "missing_rate": args.missing_rate,
        "n_features": args.n_features,
        "team_size": args.team_size,
        "single_team": args.single_team,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "normalize": args.normalize,
        "flip_pitch": args.flip_pitch,
        "player_order": args.player_order,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "start_lr": args.start_lr,
        "min_lr": args.min_lr,
        "seed": args.seed,
        "cuda": args.cuda,
        "best_loss": args.best_loss, 
        "load_pretrained": args.load_pretrained,
        "freeze_pretrained": args.freeze_pretrained,
    }

    # Hyperparameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    clip = args.clip
    print_every = args.print_every_batch
    save_every = args.save_every_epoch
    pretrain_time = args.pretrain_time
    # Set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load model
    model = load_model(args.model, params, parser, device=default_device).to(default_device)
    if args.load_pretrained:
        model = load_pretrained_model(model, params, freeze=args.load_pretrained)
    model = nn.DataParallel(model, device_ids=[0])

    # Update params with model parameters
    params = model.module.params
    params["total_params"] = num_trainable_params(model)

    # Create save path and saving parameters
    save_path = "saved/{:03d}".format(args.trial)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/model")
    with open(f"{save_path}/params.json", "w") as f:
        json.dump(params, f, indent=4)

    # Continue a previous experiment, or start a new one
    if args.cont:
        state_dict = torch.load(
            "{}/model/{}_state_dict_best.pt".format(save_path, args.model),
            map_location=default_device,
        )
        print("{}/model/{}_state_dict_best.pt".format(save_path, args.model))
        model.module.load_state_dict(state_dict)
    else:
        title = f"{args.trial} {args.dataset} {args.model} {args.missing_pattern}"

        printlog(title)

        print_keys = ["n_features", "team_size", "single_team", "window_size", "window_stride", "flip_pitch", "player_order", "start_lr"]
        printlog(get_params_str(print_keys, model.module.params))

        printlog(model.module.params_str)
        printlog("n_params {:,}".format(params["total_params"]))

    printlog("############################################################")
    print()
    print("Generating datasets...")

    if args.dataset == "soccer":  # Soccer (Metrica) datasets
        metrica_files = ["match1.csv", "match2.csv", "match3_valid.csv"]
        metrica_paths = [f"data/metrica_traces/{f}" for f in metrica_files]

        assert args.train_metrica
        train_paths = []
        if args.train_metrica:
            train_paths += metrica_paths[:-1]

        assert args.valid_metrica
        valid_paths = []
        
        if args.valid_metrica:
            valid_paths += metrica_paths[-1:]

        window_size = args.window_size

    elif args.dataset == "basketball":  # Basketball (NBA) dataset
        nba_files = os.listdir("data/nba_traces")
        nba_paths = [f"data/nba_traces/{f}" for f in nba_files]
        nba_paths.sort()
        train_paths = nba_paths[:3]
        # train_paths = nba_paths[:70]
        valid_paths = nba_paths[70:80]
        window_size = args.window_size

    else:  # American football (NFL) dataset
        nfl_paths = ["data/nfl_traces/nfl_train.csv", "data/nfl_traces/nfl_test.csv"]
        train_paths = nfl_paths[:-1]
        valid_paths = nfl_paths[-1:]
        window_size = 50

    train_dataset = SportsDataset(
        sports=args.dataset,
        data_paths=train_paths,
        window_size=window_size,
        min_episode_size=window_size,
        stride=args.window_stride,
        normalize=args.normalize,
        flip_pitch=args.flip_pitch,
        single_team=args.single_team,
    )
    valid_dataset = SportsDataset(
        sports=args.dataset,
        data_paths=valid_paths,
        window_size=window_size,
        min_episode_size=window_size,
        stride=window_size,
        normalize=args.normalize,
        flip_pitch=False,
        single_team=args.single_team,
    )

    n_workers = len(model.device_ids) * 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1 if args.dataset != "basketball" else args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )

    # Train loop
    best_valid_loss = args.best_loss
    best_valid_pe = 100

    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    for e in range(n_epochs):
        epoch = e + 1
        hyperparams = {"pretrain": epoch <= pretrain_time}

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        if args.model == "nrtsi":
            train_args, reset_best_loss, save_model = get_gap_lr_bs(args.dataset, e, args.start_lr, use_ta=1)
        else:
            train_args = None

        # Set a custom learning rate schedule
        if args.model != "nrtsi":
            if epochs_since_best == 3 and lr > args.min_lr:
                # Load previous best model
                path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
                if args.model == "nrtsi":
                    path = "{}/model/{}_state_dict_best_gap_{}.pt".format(save_path, args.model, train_args[1])

                if epoch <= pretrain_time:
                    path = "{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model)
                    if args.model == "nrtsi":
                        path = "{}/model/{}_state_dict_best_gap_{}.pt".format(save_path, args.model, train_args[1])

                state_dict = torch.load(path)

                # Decrease learning rate
                lr = max(lr / 2, args.min_lr)
                printlog("########## LR {} ##########".format(lr))
                epochs_since_best = 0
            else:
                epochs_since_best += 1

        printlog(hyperparams_str(epoch, hyperparams))
        start_time = time.time()

        train_losses = run_epoch(model, train_loader, optimizer, print_every=print_every, train_args=train_args)
        valid_losses = run_epoch(model, valid_loader, optimizer, train=False, train_args=train_args)

        if e < 5:
            print(f"Train missing rate: {round(train_losses['missing_rate'], 4)}")
            print(f"Valid missing rate: {round(valid_losses['missing_rate'], 4)}")

        printlog("Train:\t" + loss_str(train_losses))
        printlog("Valid:\t" + loss_str(valid_losses))

        epoch_time = time.time() - start_time
        printlog("Time:\t{:.2f}s".format(epoch_time))

        valid_loss = valid_losses["total_loss"]
        if args.model == "midas" and model.module.params["dynamic_hybrid"]:
            valid_pe = valid_losses["hybrid_d_pe"]
        elif args.model == "midas" and model.module.params["deriv_accum"]:
            if model.module.params["missing_pattern"] == "forecast":
                valid_pe = np.mean(valid_losses["dap_f_pe"])
            else:
                valid_pe = np.mean(valid_losses["dap_f_pe"], valid_losses["dap_b_pe"])
        elif args.model != "csdi":
            valid_pe = valid_losses["pred_pe"]

        # Best model on test set
        if args.model != "nrtsi":
            if best_valid_loss == 0 or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_since_best = 0
                path = "{}/model/{}_state_dict_best.pt".format(save_path, args.model)
                if epoch <= pretrain_time:
                    path = "{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model)
                torch.save(model.module.state_dict(), path)
                printlog("########## Best Model ###########")

            if args.model != "csdi":
                if valid_pe < best_valid_pe:
                    best_valid_pe = valid_pe
                    epochs_since_best = 0
                    path = "{}/model/{}_state_dict_best_pe.pt".format(save_path, args.model)
                    torch.save(model.module.state_dict(), path)
                    printlog("########## Best Dist. ###########")

            # Periodically save model
            if epoch % save_every == 0:
                path = "{}/model/{}_state_dict_{}.pt".format(save_path, args.model, epoch)
                torch.save(model.module.state_dict(), path)
                printlog("########## Saved Model ##########")

            # End of pretrain stage
            if epoch == pretrain_time:
                printlog("######### End Pretrain ##########")
                best_valid_loss = 0
                epochs_since_best = 0
                lr = max(args.start_lr, args.min_lr)

                state_dict = torch.load("{}/model/{}_state_dict_best_pretrain.pt".format(save_path, args.model))
                model.module.load_state_dict(state_dict)

                valid_losses = run_epoch(model, optimizer, train=False)
                printlog("Test:\t" + loss_str(valid_losses))

        else:  # for NRTSI model
            if reset_best_loss:
                best_valid_loss = 0

            if best_valid_loss == 0 or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                path = "{}/model/{}_state_dict_best_gap_{}.pt".format(save_path, args.model, train_args[1])
                torch.save(model.module.state_dict(), path)
                printlog(f"########## Best Model (max_gap : {train_args[1]}) ###########")

    printlog("Best Test Loss: {:.4f}".format(best_valid_loss))
