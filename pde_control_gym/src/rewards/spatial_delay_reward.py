from pde_control_gym.src.rewards.base_reward import BaseReward
from typing import Optional
import numpy as np

class SpatialDelayReward(BaseReward):
    def __init__(self, nt: int, N1:int =21, norm: float = 2, truncate_penalty: float = -1e-4, terminate_reward: float = 1e2, t_horizon_length: int = 5, *extras):
        if nt is None:
            raise Exception("Number of simulation steps must be specified in the NormReward class.")
        self.nt = nt
        self.N1 = N1
        self.norm = norm
        self.t_hoizon_length = t_horizon_length
        self.truncate_penalty = truncate_penalty
        self.terminate_reward = terminate_reward
        self.t_horizon_length = t_horizon_length
        self.previous_action = 0
        self.reward_min = float('inf')  # 初始化最小奖励
        self.reward_max = float('-inf')  # 初始化最大奖励

    def reward(self, uVec: np.ndarray =None, time_index: int = None, terminate: Optional[bool] =None, truncate: Optional[bool] =None, action: Optional[float] =None, control_sample_rate: Optional[float]=0.01):        
        x_state = uVec[:, :self.N1]
        u_state = uVec[:, self.N1:]


        # 如何一个回合按时结束，并且最终范数没有很大
        if terminate and np.linalg.norm(x_state[time_index]) < self.t_horizon_length:
            print("************嘿嘿嘿，获得了额外奖励***********")
            terminate_reward = self.terminate_reward  * (1 / (1 + np.linalg.norm(x_state[time_index - 1000 :time_index : 50])))
            return np.sign(terminate_reward) * np.log1p(abs(terminate_reward)).item()

        x_reward = -1 * (np.linalg.norm(x_state[time_index]) / self.N1)
        # u_reward = -1 * (np.linalg.norm(u_state[time_index]) / (self.N1 * self.N1 - self.N1))
        
        diff_time_idx = int(round(1/control_sample_rate))
        x_diif_reward = -1 * (np.linalg.norm(x_state[time_index-diff_time_idx] - x_state[time_index])) / self.N1
        u_diif_reward = -1 * (np.linalg.norm(u_state[time_index-diff_time_idx] - u_state[time_index])) / (self.N1 * self.N1 - self.N1)
        

        final_rew = np.sign(x_diif_reward) * np.log1p(abs(x_diif_reward)) + \
            0.6 * np.sign(u_diif_reward) * np.log1p(abs(u_diif_reward)) + \
            0.0005* np.sign(x_reward) * np.log1p(abs(x_reward)) 

        


        # reward_step = int(round(self.nt * 0.1))
        # if time_index < reward_step:
        #     tmp = (np.exp(1) - 1) * (time_index / reward_step)
        #     return final_rew.item() * (np.log1p(tmp))
        # else:
        return final_rew.item()        







        
    

