import gymnasium as gym
import numpy as np
import math
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

def numerical_derivative(f, x, gamma, amp):
    h = 1e-5
    df = (f(x + h, gamma, amp) - f(x - h, gamma, amp)) / (2 * h)
    return df

def cFunc(x):
    return 20 - 20 * x

def fFunc(x, y):
    return 5 * np.cos(2 * np.pi * x) + 5 * np.sin(2 * np.pi * y) 

def buildF(x):
    s_cood, q_cood = np.meshgrid(x, x, indexing="ij")
    return fFunc(s_cood, q_cood)

def tauFunc(val, gamma, amp):
    return (amp * gamma) * np.exp(-gamma * val)

def gFunc(x, gamma, amp):
    return x - tauFunc(x, gamma, amp)

def resetTau(val):
    ampT = 1#np.random.uniform(0, 0.5)
    gammaT = 0.8#np.random.uniform(2, 4)
    tau = tauFunc(val, gammaT, ampT)
    return tau

def inverse_function(y, gamma, amp):
    result = fsolve(lambda x: gFunc(x, gamma, amp) - y, np.zeros(len(y)))#第二个参数表示预测的起始值
    return result

def iteration_k(iterationNum, gBar, gamma, amp):
    s, q = np.meshgrid(spatial, spatial, indexing="ij")
    k_0 = np.zeros((N1, N1))

    for i in range(N1):
        for j in range(i, N1):

            ff = lambda theta: 5 * np.cos(2 * np.pi * (theta + s[i, j] - q[i, j])) + 5 * np.sin(2 * np.pi * theta) 
            if s[i, j] - q[i, j] + 1 < gBar:
                g_inverse_s_q_1 = inverse_function([s[i, j] - q[i, j] + 1], gamma, amp)
                g_diff_at_g_inverse_s_q_1 = numerical_derivative(gFunc, g_inverse_s_q_1, gamma, amp)
                k_0[i, j]= -1 * integrate.quad(ff, q[i, j], 1)[0] - cFunc(g_inverse_s_q_1) / g_diff_at_g_inverse_s_q_1
            else:
                k_0[i, j]= -1 * integrate.quad(ff, q[i, j], 1)[0]

    simpsonIntCoe = np.ones((N1, N1))
    simpsonIntCoe[0, :] = 0.5
    simpsonIntCoe[-1, :] = 0.5
    simpsonIntCoe[:, 0] = 0.5
    simpsonIntCoe[:, -1] = 0.5
    
    kSum = k_0
    K_iteration = k_0
    for _ in range(iterationNum):
        K_iteration_temp = np.zeros((N1, N1))
        for ii in range(N1):#s
            for jj in range(ii, N1):#q
                mSequence = np.linspace(0, 1 - q[ii, jj], N1)
                dm = abs(mSequence[2] - mSequence[1])
                rSequence = np.linspace(s[ii, jj], q[ii, jj], N1)
                dr = abs(rSequence[2]-rSequence[1])
                
                mMatrix, rMatrix = np.meshgrid(mSequence, rSequence, indexing="ij")
                f_obj = fFunc(mMatrix + rMatrix, mMatrix + q[ii, jj]) 
                
                x_qCoodinate = np.hstack((s.flatten()[:,None], q.flatten()[:,None]))#为插值提供基准坐标
                k_obj = griddata(x_qCoodinate, K_iteration.ravel(), (mMatrix + s[ii, jj], mMatrix + rMatrix), method="linear", fill_value = 0)
                
                if s[ii, jj] -q[ii, jj] + 1 < gBar:
                    g_inverse_s_q_1 = inverse_function([s[ii, jj] - q[ii, jj] + 1],gamma, amp)
                    rrSequence = np.linspace(g_inverse_s_q_1, 1, N1)
                    drr = abs(rrSequence[2] - rrSequence[1])
                    k2_obj = griddata(x_qCoodinate, K_iteration.ravel(), (tauFunc(gamma, amp, rrSequence) + s[ii, jj] - q[ii, jj] + 1, rrSequence), method="linear", fill_value = 0)
                    K_iteration_temp[ii, jj] = np.sum(dm * dr * f_obj * k_obj * simpsonIntCoe) + np.trapz( k2_obj.squeeze() * cFunc(rrSequence).squeeze(), dx = drr)
                else:
                    K_iteration_temp[ii, jj] = np.sum(dm * dr * f_obj * k_obj * simpsonIntCoe)

        K_iteration = K_iteration_temp      
        kSum =  kSum + K_iteration_temp
    return kSum, kSum[0, :]

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

def solve_contrl_U(uu, K,  Kbud, xu, gamma, amp, g_inverse_0):
    [eta1_m, zeta1_m]= np.meshgrid(spatial,spatial)
    x_star = np.hstack((zeta1_m.flatten()[:,None], eta1_m.flatten()[:,None]))
    contrl1 = np.trapz(Kbud * xu, dx = dx)
    sequence2 = np.linspace(0, g_inverse_0, N1)#0\rightarrow g^{-1}(0)
    dx2 = abs(sequence2[1]-sequence2[2])
    u2_temp_y = sequence2 / (tauFunc(sequence2, gamma, amp))
    u2_x, u2_y = np.meshgrid(sequence2, u2_temp_y, indexing="ij")
    u2 = np.diag(griddata(x_star, uu.ravel(), (u2_x, u2_y), method="cubic", fill_value = 0))
    contrl2 = np.trapz(cFunc(sequence2) * u2 , dx = dx2)
    
    #control3
    sequence3 = np.linspace(g_inverse_0, 1, N1)#g^{-1}(0)\rightarrow 1
    dx3 = sequence3[1] - sequence3[0]
    coe3 = np.tril(np.ones((N1, N1))) - np.diag(0.5 * np.ones(N1)) 
    coe3[0, :] = 0.5
    coe3[:, -1] = 0.5
    coem3 = coe3 * dx3 * dx3
    
    u3_x, u3_y_0 = np.meshgrid(sequence3, sequence3, indexing="ij") 
    u3_y = u3_x / tauFunc(u3_x, gamma, amp)#\frac{p}{\tau (q)}
    #对u,k插值
    u3 = griddata(x_star, uu.ravel(), (u3_x, u3_y), method="cubic", fill_value = 0)  
    K_control3 = griddata(x_star, K.ravel(), (u3_x, u3_y_0), method="cubic", fill_value = 0)
    c_m3 = np.tile(cFunc(sequence3), (N1, 1))
    contrl3 = np.sum(c_m3 * K_control3 * u3 * coem3)

    #control4
    sequence4 = sequence2
    dx4 = dx2
    coe4 = np.tril(np.ones((N1, N1))) - np.diag(0.5 * np.ones(N1)) 
    coe4[0, :] = 0.5
    coe4[:, -1] = 0.5
    coem4 = coe4 * dx4 * dx4
    
    u4_x, u4_y_0 = np.meshgrid(sequence4, sequence4, indexing="ij") 
    u4_y = u4_x / tauFunc(u4_x, gamma, amp)#\frac{p}{\tau (q)}
    #对u,k插值
    u4 = griddata(x_star, uu.ravel(), (u4_x, u4_y), method="cubic", fill_value = 0)  
    K_control4 = griddata(x_star, K.ravel(), (u4_x, u4_y_0), method="cubic", fill_value = 0)
    c_m4 = np.tile(cFunc(sequence4), (N1, 1))
    contrl4 = np.sum(c_m4 * K_control4 * u4 * coem4)

    U = contrl1 - contrl2 + contrl3 + contrl4
    return U


# PDE的步长和运行时间
T = 8
dt = 0.002
control_freq = 0.002

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

ampTest = 1
gammaTest = 0.7
g_bar = gFunc(1, gammaTest, ampTest)
kernel, kbud = iteration_k(6, g_bar, gammaTest, ampTest)
g_inverse_value = inverse_function(spatial, gammaTest, ampTest)
g_inverse_0 = g_inverse_value[0]

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
while not truncate and not terminate:
    tau = obs[: N1]
    xx = obs[N1 : 2 * N1] 
    u0_tmp = obs[2 * N1 : ].reshape((N1, N))
    uu = np.hstack((u0_tmp, xx[-1] * np.ones((N+1,1))))
    # use backstepping controller
    # action = solve_contrl_U(uu, tau, kernel, kbud, xx, coe_m, c1, spatial, dx)
    action = solve_contrl_U(uu, kernel, kbud, xx, gammaTest, ampTest, g_inverse_0)
    obs, rewards, terminate, truncate, info = env.step(action)
    uStorage.append(obs)
    rewStorage.append(rewards)
    rew += rewards 
    my_flag += 1
u = np.array(uStorage)
rewArray = np.array(rewStorage)

print("Total Reward", rew)

# Plot the example
# 波形图
res = 1
fig = plt.figure(figsize=(5, 4), dpi=300)
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

temporal = temporal[::50]
meshx, mesht = np.meshgrid(spatial, temporal)
u_plot = u[::50, N1 : 2 * N1]
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
plt.savefig('./PDEControlGym/pics/backstepping_little/test3.png', dpi=300)

# L2范数
interL2 = np.sqrt(np.sum(u_plot**2, axis=1) * dx)
fig = plt.figure()
flg, ax = plt.subplots()
time2 = temporal
ax.plot(time2, interL2, label= r'$U(t)$',linewidth=2)
ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
plt.xlabel(r"$t(\text{sec})$",fontsize="13")
plt.tick_params(labelsize=13)
plt.savefig('./PDEControlGym/pics/backstepping_little/L2.png', dpi=300)

# 控制曲线图
fig = plt.figure()
flg, ax = plt.subplots()
time2 = temporal
ax.plot(time2, u_plot[:, 0], label= r'$U(t)$',linewidth=2)
ax.axhline(y=0, color='black',alpha = 0.5 ,linestyle='--', linewidth=2)
plt.xlabel(r"$t(\text{sec})$",fontsize="13")
plt.tick_params(labelsize=13)
plt.savefig('./PDEControlGym/pics/backstepping_little/control3.png', dpi=300, bbox_inches='tight')

np.savez("./PDEControlGym/pics/DataForPics/backstepping_data.npz", time = time2, b_control = u_plot[:, 0], b_l2 = interL2)

# 计算调节时间
for i in range(len(interL2)):
    if interL2[i] < 0.05:
        print("调节时间为", i * dt)
        break

# 计算稳态误差
print("稳态误差为", interL2[-1])


# 计算能量损耗
energy = np.sum(np.abs(u[:, N1 ])) 
print("能量损耗为", energy)

print("end")

