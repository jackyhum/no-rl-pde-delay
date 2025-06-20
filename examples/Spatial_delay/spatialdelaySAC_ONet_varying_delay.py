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
import math
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
from examples.Spatial_delay.RONet import CustomFeatureExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor



def noiseFunc(state):
    return state

def cFunc(x):
    return 20 - 20 * x

def fFunc(x, y):
    return 5 * np.cos(2 * np.pi * x) + 5 * np.sin(2 * np.pi * y) 

def buildF(x):
    s_cood, q_cood = np.meshgrid(x, x, indexing="ij")
    return fFunc(s_cood, q_cood)

def add_gaussian_noise(x, std_ratio):
    np.random.seed(3)
    x = np.array(x)
    std = std_ratio
    noise = np.random.normal(0, std, size=x.shape)
    return noise

def tauFunc(val, amp = 0.3, gamma = 4):
    return amp * np.cos( gamma * np.arccos(val)) + 0.7

def resetTau(val, ampT = 0.3, gammaT = 4):
    ampT = np.random.uniform(-0.5, 0.5)    
    gammaT = np.random.uniform(3, 5)
    tau = tauFunc(val, ampT, gammaT)
    return tau


def getInitialCondition(nx):
    return np.ones(nx) * 4#* np.random.uniform(3, 5)


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
    total_steps = 400_000

    spatialDelayParameters = {
            "T": T, 
            "dt": dt, 
            "X": X,
            "dx": dx, 
            "f1_m": f1_m,
            "c1": c1,
            "spatial": spatial,
            "reward_class": SpatialDelayReward(int(round(T/dt)), N1, 2 , -1e3, 400, 10),
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
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=441),
    )


    model = SAC("MlpPolicy",
                env , 
                verbose=1, 
                policy_kwargs=policy_kwargs,
                learning_starts= learning_starts,
                learning_rate = 7e-5,#learning_rate_schedule,
                batch_size = 512,
                train_freq = 2,
                tau=0.003,
                tensorboard_log="./PDEControlGym/tb/")

    
    """*************************训练代码*********************"""
    retrain = True
    if retrain:
        model.learn(total_timesteps = total_steps, 
                    tb_log_name="ronet_varying_delay",
                    # callback = eval_callback,
                )
        model.save("./PDEControlGym/Model/spatial_delay_model_varying_delay.zip")
        print("*---------训练结束-----------*")





    """*************************测试部分*********************"""
    test_T = 8
    uStorage = []
    rewStorage = []
    spatialDelayParameters["reset_init_condition_func"] = lambda nx: np.ones(nx) * 4
    spatialDelayParameters["reset_delay_func"] = lambda val: tauFunc(val)
    spatialDelayParameters["T"] = test_T
    env = gym.make("PDEControlGym-SpatialDelayPDE", **spatialDelayParameters)
    obs,__ = env.reset()
    model = SAC.load("./PDEControlGym/Model/spatial_delay_model_varying_delay.zip")
    uStorage.append(obs)

    # # 访问模型的特征提取器
    # features = model.policy.actor.features_extractor
    #     # 获取特征提取层的输出
    # with torch.no_grad():  # 不需要梯度计算
    #     output = features(obs[None, :])
    # # 打印特征提取器的输出
    # print(output)

    while not truncate and not terminate:
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

    axes.tick_params(axis='both', labelsize=12, pad=1) 
    plt.savefig('./PDEControlGym/pics/varying_delay/RONET/test3.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    # plt.show()


    # 控制曲线图
    fig = plt.figure()
    flg, ax = plt.subplots()
    time2 = temporal
    ax.plot(time2, u[:, N1 ], label= r'$U(t)$',linewidth=2)
    ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
    plt.xlabel(r"$t(\text{sec})$",fontsize="13")
    plt.tick_params(labelsize=13)
    plt.savefig('./PDEControlGym/pics/varying_delay/RONET/control3.png', dpi=300, bbox_inches='tight')



    # L2范数
    interL2 = np.sqrt(np.sum(u_plot**2, axis=1) * dx)
    fig = plt.figure()
    flg, ax = plt.subplots()
    time2 = temporal
    ax.plot(time2, interL2, label= r'$L2$',linewidth=2)
    ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
    plt.xlabel(r"$t(\text{sec})$",fontsize="13")
    plt.tick_params(labelsize=13)
    plt.savefig('./PDEControlGym/pics/varying_delay/RONET/L2.png', dpi=300, bbox_inches='tight')

    np.savez("./PDEControlGym/pics/DataForPics/ronet_data_varying_delay.npz", time = time2, ronet_control = u[:, N1 ], ronet_l2 = interL2)
    print("Final_x_l2", interL2[-1])
    print("end")




if __name__ == '__main__':
    main()

