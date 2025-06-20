import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot_encode(inds, N):
    # inds should be a torch.Tensor, not a Variable
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).cpu().long()
    dims.append(N)
    ret = torch.zeros(dims)
    ret.scatter_(-1, inds, 1)
    return ret


def logsumexp(x, axis=None):
    x_max = torch.max(x, axis, keepdim=True)[0]  # torch.max() returns a tuple
    ret = torch.log(torch.sum(torch.exp(x - x_max), axis, keepdim=True)) + x_max
    return ret


def sample_gumbel(logits, tau=1, eps=1e-20):
    u = torch.zeros(logits.size()).uniform_()
    u = Variable(u)
    if logits.is_cuda:
        u = u.cuda()
    g = -torch.log(-torch.log(u + eps) + eps)
    y = (g + logits) / tau
    return F.softmax(y)


def reparam_sample_gauss(mean, std):
    eps = torch.FloatTensor(std.size()).normal_()
    eps = Variable(eps)
    eps = eps.to(mean.device)
    # if mean.is_cuda:
    #     eps = eps.cuda()
    return eps.mul(std).add_(mean)


def sample_gmm(mean, std, coeff):
    k = coeff.size(-1)
    if k == 1:
        return sample_gauss(mean, std)

    mean = mean.view(mean.size(0), -1, k)
    std = std.view(std.size(0), -1, k)
    index = torch.multinomial(coeff, 1).squeeze()

    # TODO: replace with torch.gather or torch.index_select
    comp_mean = Variable(torch.zeros(mean.size()[:-1]))
    comp_std = Variable(torch.zeros(std.size()[:-1]))
    if mean.is_cuda:
        comp_mean = comp_mean.cuda()
        comp_std = comp_std.cuda()
    for i in range(index.size(0)):
        comp_mean[i, :] = mean.data[i, :, index.data[i]]
        comp_std[i, :] = std.data[i, :, index.data[i]]

    return sample_gauss(comp_mean, comp_std), index


def sample_multinomial(probs):
    inds = torch.multinomial(probs, 1).data.cpu().long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    if probs.is_cuda:
        ret = ret.cuda()
    return ret


def kld_gauss(mean_1, std_1, mean_2, std_2):
    kld_element = (
        2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1
    )
    return 0.5 * torch.sum(kld_element)


def kld_categorical(logits_1, logits_2):
    kld_element = torch.exp(logits_1) * (logits_1 - logits_2)
    return torch.sum(kld_element)


def nll_gauss(mean, std, x):
    pi = Variable(torch.DoubleTensor([np.pi]))

    pi = pi.to(mean.device)
    # if mean.is_cuda:
    #     pi = pi.cuda()
    nll_element = (x - mean).pow(2) / std.pow(2) + 2 * torch.log(std) + torch.log(2 * pi)

    return 0.5 * torch.sum(nll_element)


def nll_gmm(mean, std, coeff, x):
    # mean: (batch, x_dim*k)
    # std: (batch, x_dim*k)
    # coeff: (batch, k)
    # x: (batch, x_dim)

    k = coeff.size(-1)
    if k == 1:
        return nll_gauss(mean, std, x)

    pi = Variable(torch.DoubleTensor([np.pi]))
    if mean.is_cuda:
        pi = pi.cuda()
    mean = mean.view(mean.size(0), -1, k)
    std = std.view(std.size(0), -1, k)

    nll_each = (x.unsqueeze(-1) - mean).pow(2) / std.pow(2) + 2 * torch.log(std) + torch.log(2 * pi)
    nll_component = -0.5 * torch.sum(nll_each, 1)
    terms = torch.log(coeff) + nll_component

    return -torch.sum(logsumexp(terms, axis=1))


def sample_gauss_logvar(mean, logvar):
    eps = torch.DoubleTensor(mean.size()).normal_()
    eps = Variable(eps)
    if mean.is_cuda:
        eps = eps.cuda()
    return eps.mul(torch.exp(logvar / 2)).add_(mean)


def kld_gauss_logvar(mean_1, logvar_1, mean_2, logvar_2):
    kld_element = logvar_2 - logvar_1 + (torch.exp(logvar_1) + (mean_1 - mean_2).pow(2)) / torch.exp(logvar_2) - 1
    return 0.5 * torch.sum(kld_element)


def nll_gauss_logvar(mean, logvar, x):
    pi = Variable(torch.DoubleTensor([np.pi]))
    if mean.is_cuda:
        pi = pi.cuda()
    nll_element = (x - mean).pow(2) / torch.exp(logvar) + logvar + torch.log(2 * pi)

    return 0.5 * torch.sum(nll_element)


def ones(shape, device="cuda:0"):
    return torch.ones(shape).to(device)


def zeros(shape, device="cuda:0"):
    return torch.zeros(shape).to(device)


# train and pretrain discriminator
def update_discrim(
    discrim_net,
    optimizer_discrim,
    discrim_criterion,
    exp_states,
    exp_actions,
    states,
    actions,
    i_iter,
    dis_times,
    use_gpu,
    train=True,
    device="cuda:0",
):
    if use_gpu:
        exp_states, exp_actions, states, actions = (
            exp_states.to(device),
            exp_actions.to(device),
            states.to(device),
            actions.to(device),
        )

    """update discriminator"""
    g_o_ave = 0.0  # g : generator
    e_o_ave = 0.0  # e : expert
    for _ in range(int(dis_times)):
        g_o = discrim_net(
            Variable(states), Variable(actions)
        )  # Policy network(사전학습된 모델)의 예측값을 입력으로 받음.
        e_o = discrim_net(Variable(exp_states), Variable(exp_actions))  # 정답 데이터를 입력으로 받음.

        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()

        if train:
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(
                g_o, Variable(zeros((g_o.shape[0], g_o.shape[1], 1), device))
            ) + discrim_criterion(e_o, Variable(ones((e_o.shape[0], e_o.shape[1], 1), device)))
            discrim_loss.backward()
            optimizer_discrim.step()

    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times


# train policy network
def update_policy(
    policy_net,
    optimizer_policy,
    discrim_net,
    discrim_criterion,
    states_var,
    actions_var,
    i_iter,
    use_gpu,
    device="cuda:0",
):
    """
    Discriminator를 속이도록 policy_net 학습.
    사전학습된 Discriminaotor는 모델의 output은 0에 가깝게 예측하도록 사전학습되었음.
    반대로, 여기에서는 policy network의 output(g_0)가 1에 가깝게 나오도록 학습시킴.
    """
    if use_gpu:
        states_var, actions_var = states_var.to(device), actions_var.to(device)

    optimizer_policy.zero_grad()
    g_o = discrim_net(states_var, actions_var)
    policy_loss = discrim_criterion(g_o, Variable(ones((g_o.shape[0], g_o.shape[1], 1))))
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 10)
    optimizer_policy.step()


# sample trajectories used in GAN training
def collect_samples_interpolate(
    policy_net,
    expert_data,
    use_gpu,
    i_iter,
    task,
    size=64,
    name="sampling_inter",
    draw=False,
    stats=False,
    num_missing=None,
    device="cuda:0",
):
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
    data = expert_data[exp_ind].clone()  # [bs, time, feat_dim]

    # seq_len = data.shape[1]
    if use_gpu:
        data = data.to(device)
    # data = Variable(data.squeeze().transpose(0, 1))
    ground_truth = data.clone()

    data_list = [data, ground_truth]
    out = policy_net(data_list, device=device)

    # if num_missing is None:
    #     num_missing = np.random.randint(seq_len * 4 // 5, seq_len)

    # missing_list = torch.from_numpy(np.random.choice(np.arange(1, seq_len), num_missing, replace=False)).long()
    # sorted_missing_list, _ = torch.sort(missing_list)
    # print("collect sample:", sorted_missing_list)
    # data[missing_list] = 0.0
    # has_value = Variable(torch.ones(seq_len, size, 1))
    # if use_gpu:
    #     has_value = has_value.to(device)
    # has_value[missing_list] = 0.0
    # data = torch.cat([has_value, data], 2)
    # data_list = []
    # for i in range(seq_len):
    #     data_list.append(data[i:i+1])
    # samples = policy_net.sample(data_list) # [time, bs, y_dim]

    samples = out["pred"]
    states = samples[:-1, :, :]
    actions = samples[1:, :, :]
    exp_states = ground_truth[:-1, :, :]
    exp_actions = ground_truth[1:, :, :]

    mod_stats = draw_and_stats(
        samples.data, name + "_" + str(num_missing), i_iter, task, draw=draw, compute_stats=stats
    )
    exp_stats = draw_and_stats(
        ground_truth.data, name + "_expert" + "_" + str(num_missing), i_iter, task, draw=draw, compute_stats=stats
    )

    return exp_states.data, exp_actions.data, ground_truth.data, states, actions, samples.data, mod_stats, exp_stats


# # sample trajectories used in GAN training
# def collect_samples_interpolate(policy_net, expert_data, use_gpu, i_iter, task, size=64, name="sampling_inter", draw=False, stats=False, num_missing=None):
#     exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
#     data = expert_data[exp_ind].clone()
#     seq_len = data.shape[1]
#     #print(data.shape, seq_len)
#     if use_gpu:
#         data = data.cuda()
#     data = Variable(data.squeeze().transpose(0, 1))
#     ground_truth = data.clone()

#     if num_missing is None:
#         num_missing = np.random.randint(seq_len * 4 // 5, seq_len)
#         #num_missing = np.random.randint(seq_len * 18 // 20, seq_len)
#         #num_missing = 40
#     missing_list = torch.from_numpy(np.random.choice(np.arange(1, seq_len), num_missing, replace=False)).long()
#     sorted_missing_list, _ = torch.sort(missing_list)
#     print("collect sample:", sorted_missing_list)
#     data[missing_list] = 0.0
#     has_value = Variable(torch.ones(seq_len, size, 1))
#     if use_gpu:
#         has_value = has_value.cuda()
#     has_value[missing_list] = 0.0
#     data = torch.cat([has_value, data], 2)
#     data_list = []
#     for i in range(seq_len):
#         data_list.append(data[i:i+1])
#     samples = policy_net.sample(data_list) # [time, bs, y_dim]

#     states = samples[:-1, :, :]
#     actions = samples[1:, :, :]
#     exp_states = ground_truth[:-1, :, :]
#     exp_actions = ground_truth[1:, :, :]

#     mod_stats = draw_and_stats(samples.data, name + '_' + str(num_missing), i_iter, task, draw=draw, compute_stats=stats, missing_list=missing_list)
#     exp_stats = draw_and_stats(ground_truth.data, name + '_expert' + '_' + str(num_missing), i_iter, task, draw=draw, compute_stats=stats, missing_list=missing_list)

#     # return exp_states.data, exp_actions.data, ground_truth.data, states, actions, samples.data
#     return exp_states.data, exp_actions.data, ground_truth.data, states, actions, samples.data, mod_stats, exp_stats


# transfer a 1-d vector into a string
def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret


def ave_player_distance(states):
    # states: numpy (seq_lenth, batch, 10)
    count = 0
    ret = np.zeros(states.shape)
    for i in range(5):
        for j in range(i + 1, 5):
            ret[:, :, count] = np.sqrt(
                np.square(states[:, :, 2 * i] - states[:, :, 2 * j])
                + np.square(states[:, :, 2 * i + 1] - states[:, :, 2 * j + 1])
            )
            count += 1
    return ret


# draw and compute statistics
def draw_and_stats(model_states, name, i_iter, task, compute_stats=True, draw=True, missing_list=None):
    stats = {}
    if compute_stats:
        model_actions = model_states[1:, :, :] - model_states[:-1, :, :]

        val_data = model_states.cpu().numpy()
        val_actions = model_actions.cpu().numpy()

        step_size = np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2]))
        step_change = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
        stats["ave_change_step_size"] = np.mean(step_change)
        val_seqlength = np.sum(np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2])), axis=0)
        stats["ave_length"] = np.mean(val_seqlength)  ## when sum along axis 0, axis 1 becomes axis 0
        stats["ave_out_of_bound"] = np.mean((val_data < -0.51) + (val_data > 0.51))
        # stats['ave_player_distance'] = np.mean(ave_player_distance(val_data))
        # stats['diff_max_min'] = np.mean(np.max(val_seqlength, axis=1) - np.min(val_seqlength, axis=1))

    if draw:
        print("Drawing")
        draw_data = model_states.cpu().numpy()[:, 0, :]
        draw_data = unnormalize(draw_data, task)
        colormap = ["b", "r", "g", "m", "y"]
        plot_sequence(
            draw_data, task, colormap=colormap, save_name="imgs/{}_{}".format(name, i_iter), missing_list=missing_list
        )

    return stats


def unnormalize(x, task):
    dim = x.shape[-1]

    if task == "basketball":
        NORMALIZE = [94, 50] * int(dim / 2)
        SHIFT = [25] * dim
        return np.multiply(x, NORMALIZE) + SHIFT
    else:
        NORMALIZE = [128, 128] * int(dim / 2)
        SHIFT = [1] * dim
        return np.multiply(x + SHIFT, NORMALIZE)


def _set_figax(task):
    fig = plt.figure(figsize=(5, 5))

    if task == "basketball":
        img = plt.imread("data/court.png")
        img = resize(img, (500, 940, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img)

        # show just the left half-court
        ax.set_xlim([-50, 550])
        ax.set_ylim([-50, 550])

    else:
        img = plt.imread("data/world.jpg")
        img = resize(img, (256, 256, 3))

        ax = fig.add_subplot(111)
        ax.imshow(img)

        ax.set_xlim([-50, 300])
        ax.set_ylim([-50, 300])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def plot_sequence(seq, task, colormap, save_name="", missing_list=None):
    n_players = int(len(seq[0]) / 2)

    while len(colormap) < n_players:
        colormap += "b"

    fig, ax = _set_figax(task)
    if task == "basketball":
        SCALE = 10
    else:
        SCALE = 1

    for k in range(n_players):
        x = seq[:, (2 * k)]
        y = seq[:, (2 * k + 1)]
        color = colormap[k]
        ax.plot(SCALE * x, SCALE * y, color=color, linewidth=3, alpha=0.7)
        ax.plot(SCALE * x, SCALE * y, "o", color=color, markersize=8, alpha=0.5)

    # starting positions
    x = seq[0, ::2]
    y = seq[0, 1::2]
    ax.plot(SCALE * x, SCALE * y, "o", color="black", markersize=12)

    if missing_list is not None:
        missing_list = missing_list.numpy()
        for i in range(seq.shape[0]):
            if i not in missing_list:
                x = seq[i, ::2]
                y = seq[i, 1::2]
                ax.plot(SCALE * x, SCALE * y, "o", color="black", markersize=8)

    plt.tight_layout(pad=0)

    if len(save_name) > 0:
        plt.savefig(save_name + ".png")
    else:
        plt.show()

    plt.close(fig)
