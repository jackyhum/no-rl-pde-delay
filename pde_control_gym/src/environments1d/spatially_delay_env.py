import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
import torch

from scipy.sparse import coo_matrix, hstack,vstack
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy import integrate

from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D
from stable_baselines3.common.buffers import ReplayBuffer

class SpatialDelayPDE(PDEEnv1D):
    def __init__(self, f1_m: NDArray[np.float32], 
                 c1:  NDArray[np.float32],
                 spatial: NDArray[np.float32],
                 sensing_noise_func: Callable[[np.ndarray], np.ndarray],
                 reset_init_condition_func: Callable[[int], np.ndarray],
                 reset_delay_func: Callable[[int], np.ndarray], 
                 limit_pde_state_size: bool = False, 
                 max_state_value: float = 1e10, 
                 max_control_value: float = 20, 
                 control_sample_rate: float=0.1, # 默认的控制频率参数
                 **kwargs):
        super().__init__(**kwargs)
        self.sensing_noise_func = sensing_noise_func
        self.reset_init_condition_func = reset_init_condition_func 
        self.reset_delay_func = reset_delay_func
        self.limit_pde_state_size = limit_pde_state_size
        self.max_state_value = max_state_value
        self.max_control_value = max_control_value
        self.control_sample_rate = control_sample_rate
        self.f1_m = f1_m
        self.c1 = c1
        self.observation_space = spaces.Box(
                    np.full(self.nx * (self.nx + 1), -self.max_state_value, dtype="float32"),
                    np.full(self.nx * (self.nx + 1), self.max_state_value, dtype="float32"),
                )
        self.control_update = lambda control, state, dt: control
        self.sensing_update = lambda state, dx, noise: noise(state)
        ############## my PDE param  ###############
        self.spatial = spatial
        self.iteration_coe = np.zeros(self.nx * self.nx + self.nx)

    def step(self, control: float):
        Nx = self.nx
        dx = self.dx
        dt = self.dt
        sample_count = int(round(self.control_sample_rate/dt)) # 也就是状态更新25次，动作才更新1次，中间空的时间都是使用同一个控制值
        i = 0
        # Actions are applied at a slower rate then the PDE is simulated at
        while i < sample_count and self.time_index < self.nt-1:
            self.time_index += 1
            self.u[self.time_index][:] = self.iteration_coe @ self.u[self.time_index - 1][:]
            self.u[self.time_index][Nx] = self.normalize(self.control_update(
                    control, self.u[self.time_index][-2], dx), self.max_control_value
            )
            i += 1
        terminate = self.terminate()
        truncate = self.truncate()
        return (
            self.sensing_update(
                self.u[self.time_index],
                self.dx,
                self.sensing_noise_func,
            ),
            self.reward_class.reward(self.u[:, Nx : -1], self.time_index, terminate, truncate, control, control_sample_rate = 1),
            terminate,
            truncate, 
            {},
        )

    def terminate(self):
        if self.time_index >= self.nt - 1:
            return True
        else:
            return False

    def truncate(self):
        if (
            self.limit_pde_state_size
            and np.linalg.norm(self.u[self.time_index, : self.nx], 2)  >= self.max_state_value
        ):
            return True
        else:
            return False

    # step返回中触发terminate时，就调用此函数
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        try:
            init_condition = self.reset_init_condition_func(self.nx)# 根据初始化对象的值运行
            tau = self.reset_delay_func(self.spatial)
        except:
            raise Exception(
                "Please pass both an initial condition and a recirculation function in the parameters dictionary. See documentation for more details"
                )
        self.u = np.zeros(
            (self.nt, self.nx * self.nx + self.nx), dtype=np.float32
        )
        self.u[0][:self.nx] = tau
        self.u[0][self.nx: 2 * self.nx] = init_condition
        self.time_index = 0
        self.tau = tau
        self.get_iteration_coe(self.tau)
        return (
            self.sensing_update(
                self.u[self.time_index],
                self.dx,
                self.sensing_noise_func,
            ),
            {},
        )
    
    """PDE相关的其他函数"""

    # 用于 产生状态迭代的系数 A
    def get_iteration_coe(self, tau): 
        coe = self.get_cof_1(self.dt, self.dx, self.nx, 1, self.f1_m, self.c1, tau).toarray()
        coe = np.block([[np.eye(self.nx), np.zeros([self.nx, self.nx **2])], [np.zeros([self.nx **2, self.nx]),  coe]])
        self.iteration_coe = coe


    def get_cof_int_zeta_1(self, N,hx):
        cofmed = np.zeros((N,N))
        cofmed[:, N - 1] = 1 / 2 * np.ones((1,N))
        cof_int_zeta_1 = hx * (np.triu(np.ones((N,N))) - cofmed - np.diag(1 / 2 * np.ones(N)))
        return cof_int_zeta_1
    
    def get_cof_1(self, ht, hx,N1,a,f_m1,c,tau):#计算状态迭代的矩阵A
        N = N1 - 1
        cof_int_zeta_1_Nplus1 = self.get_cof_int_zeta_1(N1,hx)
        cof_int_zeta_1 = cof_int_zeta_1_Nplus1 
        cof_int_zeta_1[0, :] = 0
        C1 = a * ht / hx
        Nsquare = N1 * (N1 - 1)
        At1 = coo_matrix(((1-C1) * np.ones(N), (range(1, N1), range(1, N1))),shape = (N1, N1)).tocsr()
        At1 = At1 + coo_matrix((C1 * np.ones(N), (range(1, N1), range(N1-1))),shape = (N1, N1)).tocsr()
        At1 = At1 + ht * coo_matrix(cof_int_zeta_1 * f_m1).tocsr()

        At2 = coo_matrix((N1,Nsquare)).tocsr()
        At2 = At2 + coo_matrix((c[1 : N+1] * ht, (range(1, N1), range(N, N * (N + 1), N))), shape = (N1, Nsquare)).tocsr()
        tauuu = (np.array(tau) * np.ones((N,N+1))).reshape((-1, 1), order = "f")

        ee = ht / (hx * tauuu)
        diag_vale_c = (1 - ee).squeeze()
        diag_vale_l = ee.squeeze()
        diag_vale_l[N - 1 : Nsquare : N] = 0
        diag_vale_l = np.delete(diag_vale_l, -1)

        At4 = coo_matrix((diag_vale_c, (range(Nsquare), range(Nsquare))), shape = (Nsquare,Nsquare)).tocsr()
        At4 = At4 + coo_matrix((diag_vale_l, (range(Nsquare - 1), range(1, Nsquare))),shape = (Nsquare, Nsquare)).tocsr()

        eee = ee[0 : Nsquare : N].squeeze()

        At3 = coo_matrix((eee, (range(N - 1, Nsquare, N), N1 * [N])), shape = (Nsquare, N1)).tocsr()
        Aa1 = vstack((At1, At3))
        Aa2 = vstack((At2, At4))

        return hstack((Aa1, Aa2))

    def tauFunc(self, gamma, amp, bias, val):
        return amp * np.cos( gamma * np.arccos(val - bias) ) + 3

    def coefficient_matrix(self, length):
        obj_matrix = np.triu(np.ones((length, length))) - np.diag(0.5 * np.ones(length))
        obj_matrix[0, :] = 1 / 2
        obj_matrix[:, -1] = 1 / 2
        return obj_matrix * self.dx * self.dx
    



class LogScaler:
    def __init__(self, indices_to_scale=slice(21, 462)):
        self.indices_to_scale = indices_to_scale

    def log_scale(self, state):
        # 如果输入是 Tensor，先转换为 NumPy
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # 提取需要缩放的部分
        state_to_scale = state[self.indices_to_scale]

        # 对数缩放公式：sign(x) * log(1 + abs(x))
        scaled = np.sign(state_to_scale) * np.log1p(np.abs(state_to_scale))

        # 将缩放后的部分替换回原状态
        state[self.indices_to_scale] = scaled
        return state

class NormalizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, LogScaler, obs_shape, action_shape, device, handle_timeout_termination=False):
        super().__init__(
            size, 
            obs_shape, 
            action_shape, 
            device, 
            handle_timeout_termination=handle_timeout_termination
        )
        self.LogScaler = LogScaler

    def add(self, obs, next_obs, actions, rewards, terminations, infos):
        """在数据存入 ReplayBuffer 前，直接进行归一化"""

        # 对 obs 和 next_obs 进行归一化
        obs = self.LogScaler.log_scale(obs)
        next_obs = self.LogScaler.log_scale(next_obs)

        super().add(obs, next_obs, actions, rewards, terminations, infos)

    def sample(self, batch_size):
        """直接调用父类的 sample 方法，无需再次归一化"""
        return super().sample(batch_size)