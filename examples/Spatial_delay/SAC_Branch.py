import gymnasium as gym
import numpy as np
import math
import random
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from pde_control_gym.src import TunedReward1D
from pde_control_gym.src import SpatialDelayReward
import deepxde as dde

import pde_control_gym

from scipy.sparse import coo_matrix, hstack,vstack
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy import integrate
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from examples.Spatial_delay.RONet import BranchFeatureExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor



def noiseFunc(state):
    return state

def add_gaussian_noise(x, std_ratio):
    np.random.seed(3)
    x = np.array(x)
    std = std_ratio
    noise = np.random.normal(0, std, size=x.shape)
    return noise

def cFunc(x):
    return 20 - 20 * x

def fFunc(x, y):
    return 5 * np.cos(2 * np.pi * x) + 5 * np.sin(2 * np.pi * y) 

def buildF(x):
    s_cood, q_cood = np.meshgrid(x, x, indexing="ij")
    return fFunc(s_cood, q_cood)

def tauFunc(gamma, amp, val):
    return amp * np.cos( gamma * np.arccos(val)) + 0.7

def resetTau(val):
    ampT = 0.3 # np.random.uniform(-0.4, 0.)
    gammaT = 4 # np.random.uniform(3, 5)
    tau = tauFunc(gammaT, ampT, val)
    return tau

def resetTau_addnoise(val):
    ampT = 0.3 # np.random.uniform(-0.4, 0.)
    gammaT = 4 # np.random.uniform(3, 5)
    tau = tauFunc(gammaT, ampT, val) + np.random.normal(0, 0.1, size=val.shape)
    return tau

def getInitialCondition(nx):
    return np.ones(nx) * np.random.uniform(2, 8)

def cosine_annealing_schedule(initial_lr, min_lr, total_steps):
    def lr_schedule(progress_remaining):
        current_step = (1 - progress_remaining) * total_steps
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_step / total_steps))
    return lr_schedule

"""注意这是对照试验，使用单个branch net作为特征提取层"""
def main():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    # PDE的步长和运行时间
    T = 5
    dt = 0.002
    control_freq = 0.1


    dx = 0.05
    X = 1
    NStep = int(round(T / dt))

    terminate = False
    truncate = False
    N = int(round(X/dx))
    N1 = N + 1
    x = np.linspace(0, 1, N1) 

    spatial = np.linspace(0, X, N1, dtype=np.float32)
    f1_m = buildF(spatial)
    c1 = cFunc(spatial)

    i = 0
    rew = 0

    # 设置与环境交互的步数
    learning_starts= 10000
    total_steps = 250_000

    spatialDelayParameters = {
            "T": T, 
            "dt": dt, 
            "X": X,
            "dx": dx, 
            "f1_m": f1_m,
            "c1": c1,
            "spatial": spatial,
            "reward_class": SpatialDelayReward(int(round(T/dt)), N1, 2 , -9e5, 200, 10),
            "normalize":True,  
            "sensing_noise_func": lambda state: state,
            "limit_pde_state_size": True,
            "max_state_value": 1e10,
            "max_control_value": 30,
            "reset_init_condition_func": getInitialCondition,
            "reset_delay_func": resetTau, 
            "control_sample_rate": control_freq,  
    }

    

    env = gym.make("PDEControlGym-SpatialDelayPDE", **spatialDelayParameters)
    obs = env.reset()

    # 定义策略参数
    policy_kwargs = dict(
        features_extractor_class=BranchFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )


    model = SAC("MlpPolicy",
                env , 
                verbose=1, 
                policy_kwargs=policy_kwargs,
                learning_starts= learning_starts,
                learning_rate = 8e-5,#learning_rate_schedule,
                batch_size = 512,
                train_freq = 2,
                tau=0.002,
                tensorboard_log="./PDEControlGym/tb/")
    

    """*************************训练代码*********************"""
    retrain = False
    if retrain:
        model.learn(total_timesteps = total_steps, 
                    tb_log_name="onenet",
                    # callback = eval_callback,
                )
        model.save("./PDEControlGym/Model/branchnet_model.zip")
        print("*---------训练结束-----------*")

    """*************************测试部分*********************"""
    test_T = 8  
    uStorage = []
    rewStorage = []
    spatialDelayParameters["reset_init_condition_func"] = lambda nx: np.ones(nx) * 6
    spatialDelayParameters["T"] = test_T
    env = gym.make("PDEControlGym-SpatialDelayPDE", **spatialDelayParameters)
    obs,__ = env.reset()
    model = SAC.load("./PDEControlGym/Model/branchnet_model.zip")
    uStorage.append(obs)


    while not truncate and not terminate:
        # action, _ = model.predict(obs, deterministic=True)
        # obsT = torch.from_numpy(obs).cuda().reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, terminate, truncate, info = env.step(action)
        uStorage.append(obs)
        rewStorage.append(rewards)
    u = np.array(uStorage)
    rewArray = np.array(rewStorage)
    print("Total Reward",  np.sum(rewArray))
    if truncate:
        print("Truncated")

    """*************************绘图部分*********************"""

    # 波形图
    res = 1
    fig = plt.figure(figsize=(5, 4), dpi=300)
    spatial = np.linspace(0, X, int(round(X/dx)) + 1)
    temporal = np.linspace(0, test_T, len(uStorage))
    u = np.array(uStorage)


    subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

    subfig = subfigs
    subfig.subplots_adjust(left=0.07, bottom=0, right=1, top=1.1)
    axes = subfig.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d", "computed_zorder": False})

    for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
        axis._axinfo['axisline']['linewidth'] = 1
        axis._axinfo['axisline']['color'] = "b"
        axis._axinfo['grid']['linewidth'] = 0.2
        axis._axinfo['grid']['linestyle'] = "--"
        axis._axinfo['grid']['color'] = "#d1d1d1"
        axis.set_pane_color((1,1,1))
        
    meshx, mesht = np.meshgrid(spatial, temporal)
    u_plot = u[:, N1 : 2 * N1]
    axes.set_box_aspect([1, 1, 0.6]) 
    axes.plot_surface(meshx, mesht, u_plot, edgecolor="black",lw=0.2, rstride=10, cstride=2, 
                            alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
    axes.plot(np.zeros(len(temporal)), temporal, u_plot[:, 0], color="red", lw=2, antialiased=True)
    axes.view_init(10, 15)
    axes.invert_xaxis()
    axes.set_xlabel(r"x", fontsize=14)
    axes.set_ylabel(r"$t(\text{sec})$", fontsize=14)
    axes.set_zlabel(r"$v(x, t)$", rotation=90, fontsize=14)
    axes.zaxis.set_rotate_label(False)
    axes.set_xticks([0, 0.5, 1])
    #axes.set_zlim([-20, 4])  # 设置y轴范围
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    axes.tick_params(axis='both', labelsize=12, pad=1) 
    plt.savefig('./PDEControlGym/pics/BranchNet/test3.png', dpi=300)


    # 控制曲线图
    fig = plt.figure()
    flg, ax = plt.subplots()
    time2 = temporal
    ax.plot(time2, u[:, N1 ], label= r'$U(t)$',linewidth=2)
    ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
    plt.xlabel(r"$t(\text{sec})$",fontsize="13")
    plt.tick_params(labelsize=13)
    plt.savefig('./PDEControlGym/pics/BranchNet/control3.png', dpi=300, bbox_inches='tight')



    # L2范数
    interL2 = np.sqrt(np.sum(u_plot**2, axis=1) * dx)
    fig = plt.figure()
    flg, ax = plt.subplots()
    time2 = temporal
    ax.plot(time2, interL2, label= r'$L2$',linewidth=2)
    ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
    plt.xlabel(r"$t(\text{sec})$",fontsize="13")
    plt.tick_params(labelsize=13)
    plt.savefig('./PDEControlGym/pics/BranchNet/L2.png', dpi=300, bbox_inches='tight')
    print("Final_x_l2", interL2[-1])
    np.savez("./PDEControlGym/pics/DataForPics/branch_data.npz", time = time2, branch_control = u[:, N1 ], branch_l2 = interL2)
    print("end")


    # """*****************噪声实验*****************"""
    # terminate = False
    # truncate = False
    # test_T = 8
    # uStorage = []
    # rewStorage = []
    # spatialDelayParameters["reset_init_condition_func"] = lambda nx: np.ones(nx) * 6
    # spatialDelayParameters["reset_delay_func"] = resetTau_addnoise
    # spatialDelayParameters["T"] = test_T
    # env = gym.make("PDEControlGym-SpatialDelayPDE", **spatialDelayParameters)
    # obs,__ = env.reset()
    # model = SAC.load("./PDEControlGym/Model/branchnet_model.zip")
    # uStorage.append(obs)


    # while not truncate and not terminate:
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, rewards, terminate, truncate, info = env.step(action)
    #     uStorage.append(obs)
    #     rewStorage.append(rewards)
    # u = np.array(uStorage)
    # rewArray = np.array(rewStorage)
    # print("Total Reward",  np.sum(rewArray))
    # if truncate:
    #     print("Truncated")
    

    # # 波形图
    # res = 1
    # fig = plt.figure()
    # spatial = np.linspace(0, X, int(round(X/dx)) + 1)
    # temporal = np.linspace(0, test_T, len(uStorage))
    # u = np.array(uStorage)


    # subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

    # subfig = subfigs
    # subfig.subplots_adjust(left=0.07, bottom=0, right=1, top=1.1)
    # axes = subfig.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d", "computed_zorder": False})

    # for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
    #     axis._axinfo['axisline']['linewidth'] = 1
    #     axis._axinfo['axisline']['color'] = "b"
    #     axis._axinfo['grid']['linewidth'] = 0.2
    #     axis._axinfo['grid']['linestyle'] = "--"
    #     axis._axinfo['grid']['color'] = "#d1d1d1"
    #     axis.set_pane_color((1,1,1))
        
    # meshx, mesht = np.meshgrid(spatial, temporal)
    # u_plot = u[:, N1 : 2 * N1]
    # axes.plot_surface(meshx, mesht, u_plot, edgecolor="black",lw=0.2, rstride=10, cstride=2, 
    #                         alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
    # axes.plot(np.zeros(len(temporal)), temporal, u_plot[:, 0], color="red", lw=2, antialiased=True)
    # axes.view_init(10, 15)
    # axes.invert_xaxis()
    # axes.set_xlabel("x")
    # axes.set_ylabel("Time")
    # axes.set_zlabel(r"$u(x, t)$", rotation=90)
    # axes.zaxis.set_rotate_label(False)
    # axes.set_xticks([0, 0.5, 1])
    # #axes.set_zlim([-20, 4])  # 设置y轴范围
    # plt.savefig('./PDEControlGym/pics/BranchNet/noise/test3.png', dpi=300)
    # # plt.show()


    # # 控制曲线图
    # fig = plt.figure()
    # flg, ax = plt.subplots()
    # time2 = temporal
    # ax.plot(time2, u[:, N1 ], label= r'$U(t)$',linewidth=2)
    # ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
    # plt.xlabel(r'$Time$',fontsize="13")
    # plt.tick_params(labelsize=13)
    # plt.savefig('./PDEControlGym/pics/BranchNet/noise/control3.png', dpi=300)


    # # L2范数
    # interL2 = np.sqrt(np.sum(u_plot**2, axis=1) * dx)
    # fig = plt.figure()
    # flg, ax = plt.subplots()
    # time2 = temporal
    # ax.plot(time2, interL2, label= r'$L2$',linewidth=2)
    # ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
    # plt.xlabel(r'$Time$',fontsize="13")
    # plt.tick_params(labelsize=13)
    # plt.savefig('./PDEControlGym/pics/BranchNet/noise/L2.png', dpi=300)


if __name__ == '__main__':
    main()
