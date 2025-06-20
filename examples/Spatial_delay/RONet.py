import deepxde as dde
import torch
import os
import numpy as np
from torch import nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp
import torch.optim as optim

N1 = 21

def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    obs_first = obs[..., :21] 
    obs_last = obs[..., 21:]  
    obs_last_processed = torch.sign(obs_last) * torch.log1p(torch.abs(obs_last))
    obs_preprocessed = torch.cat([obs_first, obs_last_processed], dim=-1)
    return obs_preprocessed

def convert_to_tensor(input_data):
    # 检查输入是否为 PyTorch Tensor
    if not isinstance(input_data, torch.Tensor):
        # 如果输入是 NumPy 数组，将其转换为 Tensor
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        else:
            # 如果输入既不是 Tensor 也不是 NumPy 数组，尝试直接转换为 Tensor
            input_data = torch.tensor(input_data)
    return input_data

def myReshapeFunc2(all_data, N1 = 21):
    batch = all_data.shape[0]  # 获取输入张量的批量大小
    tau_matrix = all_data[:, :N1].unsqueeze(1).expand(batch, N1, N1)
    x_stat_matrix = all_data[:, N1:2 * N1].unsqueeze(1).expand(batch, N1, N1)
    u_state_tmp = all_data[:, 2 * N1:].reshape(batch, N1, -1) #(bacth, N1, N)
    last_element = all_data[:, 2 * N1 - 1].unsqueeze(1).unsqueeze(2) #(batch, 1, 1)
    ones_column = last_element.expand(batch, N1, 1)
    u_state = torch.cat((u_state_tmp, ones_column), dim=2)
    result = torch.stack([u_state, tau_matrix, x_stat_matrix], dim=1)
    return result

grids = []
grids.append(np.linspace(0, 1, 21, dtype=np.float32))
grids.append(np.linspace(0, 1, 21, dtype=np.float32))
grid2 = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid2 = torch.from_numpy(grid2).cuda()    

class BranchNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=2)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=2)
        self.fc1 = torch.nn.Linear(1152, 256)

        
    def forward(self, x):
        x = self.myReshapeFunc2(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)    
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        return x
    
    def myReshapeFunc2(self, all_data):
        all_data = convert_to_tensor(all_data).cuda()
        batch = all_data.shape[0]  # 获取输入张量的批量大小

        tau_matrix = all_data[:, :N1].unsqueeze(1).expand(batch, N1, N1)
        x_stat_matrix = all_data[:, N1:2 * N1].unsqueeze(1).expand(batch, N1, N1)

        u_state_tmp = all_data[:, 2 * N1:].reshape(batch, N1, -1) #(bacth, N1, N)
        last_element = all_data[:, 2 * N1 - 1].unsqueeze(1).unsqueeze(2) #(batch, 1, 1)
        ones_column = last_element.expand(batch, N1, 1)
        u_state = torch.cat((u_state_tmp, ones_column), dim=2)
        result = torch.stack([u_state, tau_matrix, x_stat_matrix], dim=1)
        return result
    
class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=441):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # 定义你的网络结构
        self.net1 = dde.nn.DeepONetCartesianProd([N1 * N1, BranchNet(N1)], [2, 64 ,128, 256], "relu", "Glorot normal").cuda()
        self.net1.load_state_dict(torch.load("./PDEControlGym/Model/pretrain_deeponet.zip", weights_only=True))
    def forward(self, observations):
        # 前向传播
        x = self.net1((observations, grid2))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(x)
        return x
    

class BranchFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(BranchFeatureExtractor, self).__init__(observation_space, features_dim)
        # 定义你的网络结构
        self.net1 =  BranchNet(N1).cuda()

    def forward(self, observations):
        # 前向传播
        x = self.net1(observations)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(x)
        return x






















LOG_STD_MAX = 2
LOG_STD_MIN = -5

class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = grid2
        self.net1 = dde.nn.DeepONetCartesianProd([N1 * N1, BranchNet(N1)], [2, 64 ,128, 256, 256], "relu", "Glorot normal").cuda()
        self.fc1 = nn.Linear(441, 256)
        self.fc_mean = nn.Linear(256, 1)
        self.fc_logstd = nn.Linear(256, 1)

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor(100, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(0, dtype=torch.float32)
        )
    def forward(self, x):
        x = self.net1((x, self.grid))
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std
    
    def action_log_prob(self, obs, deterministic=False):
        """
        生成动作并计算对数概率。
        :param obs: 输入观测值（状态）
        :param deterministic: 是否使用确定性策略
        :return: 动作和对数概率
        """
        mean, log_std = self(obs)
        if deterministic:
            # 确定性策略：直接返回均值缩放后的动作
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, None
        else:
            # 随机策略：从正态分布中采样动作
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # 重参数化采样
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            # 计算对数概率
            log_prob = normal.log_prob(x_t)
            # 修正对数概率（考虑 tanh 变换）
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 定义第一个 Q 网络
        self.fc1_q1 = nn.Linear(input_dim + 1, hidden_dim)  # +1 是为了动作维度
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, output_dim)

        # 定义第二个 Q 网络
        self.fc1_q2 = nn.Linear(input_dim + 1, hidden_dim)  # +1 是为了动作维度
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, observations, actions):
        # 将状态和动作拼接在一起
        x = torch.cat([observations, actions], dim=1)

        # 计算第一个 Q 值
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        # 计算第二个 Q 值
        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)

        # 返回两个 Q 值
        return q1, q2

class RONet(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.critic = CriticNetwork(input_dim=462, hidden_dim=256, output_dim=1)
        self.critic_target = CriticNetwork(input_dim=462, hidden_dim=256, output_dim=1)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = ActorNetwork()

        # 使用自定义的学习率
        self.actor.optimizer = optim.Adam(self.actor.parameters())
        self.critic.optimizer = optim.Adam(self.critic.parameters())

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        obs_preprocessed = preprocess_obs(obs)
        # 生成动作
        action, log_prob = self.actor.action_log_prob(obs_preprocessed, deterministic)
        
        # 计算 Q 值
        q1, q2 = self.critic(obs_preprocessed, action)
        
        # 返回动作、对数概率和 Q 值
        return action, log_prob, q1, q2

    def _predict(self, observation, deterministic: bool = False):
        """
        预测动作，用于与环境交互。
        :param observation: 输入观测值（状态）
        :param deterministic: 是否使用确定性策略
        :return: 动作
        """
        obs_preprocessed = preprocess_obs(observation)
        with torch.no_grad():
            action, _ = self.actor.action_log_prob(obs_preprocessed, deterministic)
            return action if deterministic else action


    def predict_values(self, obs: torch.Tensor):
        """
        预测状态值函数。
        :param obs: 输入观测值（状态）
        :return: 状态值
        """
        obs_preprocessed = preprocess_obs(obs)
        with torch.no_grad():
            _, _, q1, q2 = self.forward(obs_preprocessed)
        return torch.min(q1, q2)  # 返回两个 Q 值中的较小值

    def action_log_prob(self, obs: torch.Tensor):
        """
        计算动作和对数概率。
        :param obs: 输入观测值（状态）
        :return: 动作、对数概率
        """
        obs_preprocessed = preprocess_obs(obs)
        action, log_prob  = self.actor.action_log_prob(obs_preprocessed)
        return action, log_prob
    import torch





