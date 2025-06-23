# Neural Operator-based Reinforcement Learning for Control of First-Order PDEs with Spatially-Varying State Delay

[📄 arXiv:2501.18201](https://arxiv.org/abs/2501.18201)

## 🔍 Overview

This repository contains the official implementation for the paper:

> **"Neural Operator based Reinforcement Learning for Control of First-order PDEs with Spatially-Varying State Delay"**  
> *Jiaqi Hu, Jie Qi, and Jing Zhang.*

**🎉 News**: This paper has been accepted for presentation at the 5th IFAC Workshop on Control of Systems Governed by Partial Differential Equations (CPDE 2025) on June 18-20, 2025, Beijing, China.




## 🚀 Highlights

- ✅ Addressing **spatially-varying state delays**, which challenge classical Markovian assumptions.
- ✅ Leveraging **DeepONet** to approximate control policies.
- ✅ Integrating with **Soft Actor-Critic (SAC)** to train NO-based feedback controllers.
- ✅ Benchmarking against numerical controllers and various operator architectures.

## 🧠 Key Features

- **Operator-based control**: Learn policies that map delayed profiles and delay functions to actions.
- **Continuous action space**: Designed for PDE systems with real-valued control inputs.
- **Modular RL design**: Easily switch between DeepONet, FNO, and standard NN policies.
- **Data-efficient**: Training with synthetic PDE simulations using low sampling frequency.


## 📁 Project Structure
```
├── Model/                                # Neural network weights
├── example/                              # Executable code
├── pde_control_gym/                      # Gym enviroment for PIDEs with spatially varying delay
├── tb/                                   # data of tensorboard
└── README.md
```
## 📌 Citation
If you find this repository helpful, please consider citing our paper:

```
@article{hum2024neural,
  title={Neural Operator based Reinforcement Learning for Control of First-order PDEs with Spatially-Varying State Delay},
  author={Hum, Jacky and others},
  journal={arXiv preprint arXiv:2501.18201},
  year={2024}
}
```
