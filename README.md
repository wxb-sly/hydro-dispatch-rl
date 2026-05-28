# Hydro Dispatch AI: Autonomous Energy Optimization


## 🎯 Mission Objective
To architect a physics-constrained Reinforcement Learning (RL) environment that simulates a hydropower reservoir, and to train an autonomous agent to optimize water dispatch for maximum commercial revenue. The system handles the complex decision frontier of storing water (building hydraulic head) versus generating power immediately against dynamic Time-of-Day (ToD) electricity tariffs.

## 🏗️ Architecture & Philosophy

1. **Physics First:** If the conservation of mass is violated, the AI learns to optimize a hallucination. The environment hard-enforces real-world physical constraints (reservoir capacities, max turbine discharge, water balance).
2. **Commercial Focus:** The reward function is explicitly mapped to financial performance ($/MWh), bridging engineering models directly to financial liability and revenue.
3. **Modular OOP Structure:** The environment is built natively on the Farama `gymnasium` API contract, ensuring plug-and-play capability with modern deep learning algorithms.

## ⚙️ The Technology Stack
* **Language Engine:** Python 3.11
* **The Arena (Environment):** `gymnasium`
* **Agent Algorithms:** `stable-baselines3` (PPO / SAC)
* **Neural Substrate:** `pytorch`
* **Telemetry & Visualization:** `tensorboard`, `seaborn`

## 🚦 Project Status: Phase 1 Complete (Suspended)
**Phase 1: Environment Engineering** is structurally complete. The physics simulator (`HydroDispatchEnv`) is fully operational, mathematically verified against mass conservation constraints, and integrated with commercial Time-of-Day (ToD) tariff logic.

The project is currently suspended. Forward development will resume with **Phase 2: Agent Training**, utilizing Proximal Policy Optimization (PPO) via `stable-baselines3` to converge on an autonomous dispatch policy.

