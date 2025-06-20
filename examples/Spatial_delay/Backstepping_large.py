import gymnasium as gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from pde_control_gym.src import TunedReward1D
from pde_control_gym.src import SpatialDelayReward
import pde_control_gym


from scipy.sparse import coo_matrix, hstack,vstack
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy import integrate


def noiseFunc(state):
    return state

def getInitialCondition(nx):
    return np.ones(nx) * 6#np.random.uniform(1, 10)

def cFunc(x):
    return 20 - 20 * x

def fFunc(x, y):
    return 5 * np.cos(2 * np.pi * x) + 5 * np.sin(2 * np.pi * y) 

def buildF(x):
    s_cood, q_cood = np.meshgrid(x, x, indexing="ij")
    return fFunc(s_cood, q_cood)

def tauFunc(gamma, amp, val):
    return amp * np.cos( gamma * np.arccos(val)) + 1.5

def resetTau(val):
    ampT = 0.4#np.random.uniform(0, 0.5)
    gammaT = 3#np.random.uniform(2, 4)
    tau = tauFunc(gammaT, ampT, val)
    return tau

def get_cof_equ_K_19(i, N1, f1_m, hx, a):
    num = N1 - i - 2
    coff = f1_m[i, i+1:N1-1].T
    cofi1 = coo_matrix((2 * np.ones(num), (range(num), range(num))), shape = (num, num)).tocsr()
    cofi1 = cofi1 + coo_matrix((-1 * np.ones((num - 1)), (range(num - 1),range(1, num))),shape = (num, num)).tocsr()
    cofi3_1 = np.tril(np.ones((num, num))) - np.diag(1 / 2 * np.ones(num))
    cofi3_1 = hx * cofi3_1
    cofi3_2 = np.zeros((num,num))
    for k in range(num):
        cofi3_2[k, 0:k] = f1_m[i + 1: i + k + 1, i + k + 1].T
    cofi3 = -hx / (a) * cofi3_1 * cofi3_2
    cofi = cofi1 + coo_matrix(cofi3).tocsr() 
    return cofi, coff

def KernelCalc2(a, hx, f1_m, N1): 
    fii = np.diag(f1_m)
    cofmed = np.zeros((N1, N1))
    cofmed[:, -1] = 1/ 2 * np.ones((1, N1))
    cof_int_zeta_1 = hx * (np.triu(np.ones((N1, N1))) - cofmed - np.diag(1 / 2 * np.ones(N1)))
    K_ii = -1 / a * np.dot(cof_int_zeta_1, fii)
    K = np.diag(K_ii)

    for i in range(N1 - 3, -1, -1 ):
        [cofi, coff] = get_cof_equ_K_19(i,N1,f1_m,hx,a)
        Kibud = K[i, i] * hx * hx / (2 * a) - hx / a
        Ki = np.linalg.solve(cofi.toarray(), K[i+1,i+1:N1-1].T) + np.linalg.solve(cofi.toarray(), np.dot(Kibud, coff))
        K[i, i+1:N1-1]=Ki.T  
    Kbud = K[0, :]
    return K, Kbud 

def coefficient_matrix(length, dx):
    obj_matrix = np.triu(np.ones((length, length))) - np.diag(0.5 * np.ones(length))
    obj_matrix[0, :] = 1 / 2
    obj_matrix[:, -1] = 1 / 2
    return obj_matrix * dx * dx

def get_cof_int_zeta_1(N,hx):
    cofmed = np.zeros((N,N))
    cofmed[:, N - 1] = 1 / 2 * np.ones((1,N))
    cof_int_zeta_1 = hx * (np.triu(np.ones((N,N))) - cofmed - np.diag(1 / 2 * np.ones(N)))
    return cof_int_zeta_1

def solve_contrl_U(uu, tau, K, Kbud, xu, coe_m, c1, spatial, dx):
    [eta1_m, zeta1_m]= np.meshgrid(spatial,spatial)
    u1_temp_y = spatial / (1 * tau)
    u1_x, u1_y = np.meshgrid(spatial, u1_temp_y)

    u2_temp_y = spatial / (1 * tau)
    u2_x, u2_y = np.meshgrid(spatial, u2_temp_y)
    
    x_star = np.hstack((zeta1_m.flatten()[:,None], eta1_m.flatten()[:,None]))


    u1 = np.diag(griddata(x_star, uu.ravel(), (u1_x, u1_y), method="cubic", fill_value = 0))
    u2 = griddata(x_star, uu.ravel(), (u2_x, u2_y), method="cubic", fill_value = 0)  

    contrl1 = np.trapz( Kbud * xu, dx = dx)
    contrl2 = np.trapz(c1 * u1 / 1, dx = dx)

    c_m = np.tile(c1, (len(spatial), 1))
    contrl3 = np.sum(c_m * K * u2 * coe_m)
    U = contrl1 - contrl2 + contrl3
    return U


# PDE的步长和运行时间
T = 10
dt = 0.002
control_freq = 0.1



dx = 0.05
X = 1

terminate = False
truncate = False
N = int(round(X/dx))
N1 = N + 1
x = np.linspace(0, 1, N1) 

spatial = np.linspace(0, X, N1, dtype=np.float32)
f1_m = buildF(spatial)
c1 = cFunc(spatial)
coe_m = coefficient_matrix(N1, dx)
kernel, kbud = KernelCalc2(1, dx, f1_m, N1)
i = 0
rew = 0

hyperbolicParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "f1_m": f1_m,
        "c1": c1,
        "spatial": spatial,
        "reward_class": SpatialDelayReward(int(round(T/dt)), N1, 2 , -1e3, 80, 10),
        "normalize":False,  
        "sensing_noise_func": lambda state: state,
        "limit_pde_state_size": True,
        "max_state_value": 1e10,
        "max_control_value": 20,
        "reset_init_condition_func": getInitialCondition,
        "reset_delay_func": resetTau, 
        "control_sample_rate": control_freq,  # 设置控制的频率
}

env = gym.make("PDEControlGym-SpatialDelayPDE", **hyperbolicParameters)
uStorage = []
rewStorage = []
obs,__ = env.reset()
uStorage.append(obs)
my_flag = 0
start_time = time.time()
while not truncate and not terminate:
    tau = obs[: N1]
    xx = obs[N1 : 2 * N1] 
    u0_tmp = obs[2 * N1 : ].reshape((N1, N))
    uu = np.hstack((u0_tmp, xx[-1] * np.ones((N+1,1))))
    # use backstepping controller
    action = solve_contrl_U(uu, tau, kernel, kbud, xx, coe_m, c1, spatial, dx)
    obs, rewards, terminate, truncate, info = env.step(action)
    uStorage.append(obs)
    rewStorage.append(rewards)
    rew += rewards 
    my_flag += 1
end_time = time.time()
print("Time for Backstepping:", end_time - start_time, "seconds")
u = np.array(uStorage)
rewArray = np.array(rewStorage)

print("Total Reward", rew)

# Plot the example
res = 1
fig = plt.figure()
spatial = np.linspace(0, X, int(round(X/dx)) + 1)
temporal = np.linspace(0, T, len(uStorage))
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
axes.plot_surface(meshx, mesht, u_plot, edgecolor="black",lw=0.2, rstride=10, cstride=2, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
axes.plot(np.zeros(len(temporal)), temporal, u_plot[:, 0], color="red", lw=2, antialiased=True)
axes.view_init(10, 15)
axes.invert_xaxis()
axes.set_xlabel("x")
axes.set_ylabel("Time")
axes.set_zlabel(r"$u(x, t)$", rotation=90)
axes.zaxis.set_rotate_label(False)
axes.set_xticks([0, 0.5, 1])
plt.savefig('./PDEControlGym/pics/backstepping_large/test3.png', dpi=300)




print("end")
