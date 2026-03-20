import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import time

# ==========================================
# 1. 神经网络结构 (保持原有结构)
# ==========================================
class ParkourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.ModuleDict({
                "model": nn.Sequential(
                    nn.Linear(231, 128), nn.ELU(),
                    nn.Linear(128, 64), nn.ELU(),
                    nn.Linear(64, 32)
                )
            })
        ])
        self.memory_a = nn.ModuleDict({
            "rnn": nn.GRU(input_size=80, hidden_size=256, num_layers=1)
        })
        self.actor = nn.Sequential(
            nn.Linear(256, 512), nn.ELU(), 
            nn.Linear(512, 256), nn.ELU(), 
            nn.Linear(256, 128), nn.ELU(), 
            nn.Linear(128, 12)             
        )
        self.std = nn.Parameter(torch.ones(12))
        # 显式定义隐藏状态
        self.rnn_hidden = None

    def forward(self, obs, hidden_state=None):
        proprio = obs[:, :48]
        heights = obs[:, 48:]
        latent = self.encoders[0].model(heights)
        rnn_input = torch.cat([proprio, latent], dim=-1).unsqueeze(0)
        out, hidden_state = self.memory_a.rnn(rnn_input, hidden_state)
        actions = self.actor(out.squeeze(0))
        return actions, hidden_state

# ==========================================
# 2. 仿真主类 (完善细节)
# ==========================================
class Go2ParkourSim:
    def __init__(self, xml_path, model_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 物理与控制参数
        self.kp = 40.0
        self.kd = 1.0
        self.action_scale = 0.5
        self.torque_limits = 23.7
        self.decimation = 4  # 神经网络 50Hz, 物理仿真 200Hz (假设 timestep=0.005)
        
        # 状态记录
        self.last_action = torch.zeros(12)
        self.commands = torch.tensor([2.0, 0.0, 0.0]) # 前进速度指令
        
        # Go2 默认姿态 (站立)
        self.default_dof_pos = torch.tensor([
            0.1, 0.7, -1.5, -0.1, 0.7, -1.5, 
            0.1, 1.0, -1.5, -0.1, 1.0, -1.5
        ])
        
        # 缩放参数
        self.scales = {
            "lin_vel": 1.0, "ang_vel": 0.25,
            "commands": np.array([2.0, 2.0, 0.25]),
            "dof_pos": 1.0, "dof_vel": 0.05,
            "height_measurements": 5.0, "height_offset": -0.2
        }
        self.clip_obs = 100.
        
        # 加载模型
        self.policy = self._load_policy(model_path)
        self.reset()

    def _load_policy(self, path):
        policy = ParkourPolicy()
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt['model_state_dict']
        
        # 修正 Key 映射逻辑
        new_dict = {}
        for k, v in state_dict.items():
            name = k.replace("memory_a.rnn.", "memory_a.rnn.") # 如果权重里已经是这个名字，直接保留
            # 注意：如果你的权重里有 "encoders.0.model."，我们要匹配到 self.encoders[0].model
            name = name.replace("encoders.0.model.", "encoders.0.model.")
            if name in policy.state_dict():
                new_dict[name] = v
        
        policy.load_state_dict(new_dict, strict=False)
        policy.eval() # 推理模式
        return policy

    def reset(self):
        """重置机器人到初始状态并清空 RNN 记忆"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.5 # 离地高度
        self.data.qpos[7:19] = self.default_dof_pos # 初始关节
        mujoco.mj_forward(self.model, self.data)
        
        # *** 关键：重置 RNN 隐藏状态 ***
        self.policy.rnn_hidden = torch.zeros(1, 1, 256)
        self.last_action = torch.zeros(12)

    def _get_obs(self):
        """完全按照 legged_gym 的规范获取观测"""
        # 1. 旋转矩阵 (世界到躯干)
        quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        
        # 2. 本体感知
        # 注意：mujoco 的 qvel[:3] 是世界系线速度，需转到局部系
        raw_lin_vel = rot_mat.T @ self.data.qvel[:3]
        raw_ang_vel = rot_mat.T @ self.data.qvel[3:6] # 角速度通常也转到局部
        raw_gravity = rot_mat.T @ np.array([0, 0, -1])
        
        # 关节位置/速度 (减去默认姿态)
        raw_dof_pos = self.data.qpos[7:19] - self.default_dof_pos.numpy()
        raw_dof_vel = self.data.qvel[6:18]
        
        # 3. 外部环境 (高度测量)
        raw_heights = np.zeros(231) # 实际部署需替换为高度采样
        scaled_heights = (raw_heights + self.scales["height_offset"]) * self.scales["height_measurements"]
        
        # 4. 组合
        obs_components = [
            raw_lin_vel * self.scales["lin_vel"],         # 3
            raw_ang_vel * self.scales["ang_vel"],         # 3
            raw_gravity,                                   # 3
            self.commands.numpy() * self.scales["commands"],# 3
            raw_dof_pos * self.scales["dof_pos"],          # 12
            raw_dof_vel * self.scales["dof_vel"],          # 12
            self.last_action.numpy(),                      # 12
            scaled_heights                                 # 231
        ]
        
        obs_array = np.concatenate(obs_components)
        obs_array = np.clip(obs_array, -self.clip_obs, self.clip_obs)
        return torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)

    def step(self):
        """执行一个策略周期 (Policy Step)"""
        obs = self._get_obs()
        
        # 1. 网络推理
        with torch.no_grad():
            # 推理 action 并更新 RNN 状态
            action, self.policy.rnn_hidden = self.policy(obs, self.policy.rnn_hidden)
        
        # 2. Action 处理
        action = torch.clamp(action[0], -1.0, 1.0)
        self.last_action = action # 保存未缩放的 action 供下一个 obs 使用
        
        # 3. 控制周期 (Decimation)
        # 模拟 Motor Strength (DR 鲁棒性)
        motor_strength = 0.4 * torch.rand(12) + 0.8 
        
        for _ in range(self.decimation):
            # PD 控制
            q_now = self.data.qpos[7:19]
            dq_now = self.data.qvel[6:18]
            
            # 目标位置 = Action * Scale + Default
            target_q = (action * motor_strength).numpy() * self.action_scale + self.default_dof_pos.numpy()
            
            # 计算力矩
            tau = self.kp * (target_q - q_now) - self.kd * dq_now
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            print(tau)
            
            self.data.ctrl[:] = tau
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
    def step_no(self,obs):
        """执行一个策略周期 (Policy Step)"""
        obs = obs.cpu()
        
        # 1. 网络推理
        with torch.no_grad():
            # 推理 action 并更新 RNN 状态
            action, self.policy.rnn_hidden = self.policy(obs, self.policy.rnn_hidden)
        
        # 2. Action 处理
        action = torch.clamp(action[0], -1.0, 1.0)
        self.last_action = action # 保存未缩放的 action 供下一个 obs 使用
        
        # 3. 控制周期 (Decimation)
        # 模拟 Motor Strength (DR 鲁棒性)
        motor_strength = 0.4 * torch.rand(12) + 0.8 
        
        for _ in range(self.decimation):
            # PD 控制
            q_now = self.data.qpos[7:19]
            dq_now = self.data.qvel[6:18]
            
            # 目标位置 = Action * Scale + Default
            target_q = (action * motor_strength).numpy() * self.action_scale + self.default_dof_pos.numpy()
            
            # 计算力矩
            tau = self.kp * (target_q - q_now) - self.kd * dq_now
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            print(tau)
            
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
        return action, tau
    def test_tau_in(self,tau):
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

def print_contact_forces(sim):
    # 假设你想监控四条腿的足端，名称通常为 "FL_foot", "FR_foot" 等
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    
    for name in foot_names:
        # 获取 body ID
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id != -1:
            # cfrc_ext[id] 的 6 个分量是：[力矩x, 力矩y, 力矩z, 力x, 力y, 力z]
            force = sim.data.cfrc_ext[body_id][3:6] # 只取力分量
            print(f"{name} 受到外力: {force}")


def test_tau(sim:Go2ParkourSim ,tau, viewer):   
            step_start = time.time()
            
            if torch.is_tensor(tau):
                if tau.device.type != 'cpu':
                    tau = tau.detach().cpu()
                tau = tau.numpy()
            
            sim.test_tau_in(tau)
            
            # 检查是否摔倒 (如果身体 Z 轴过低)
            if sim.data.qpos[2] < 0.15:
                print("机器人摔倒，重置中...")
                sim.reset()        
            
            print_contact_forces(sim)
            # viewer.sync()
            
            # 维持实时率
            time_until_next_step = sim.model.opt.timestep * sim.decimation - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

# ==========================================
# 3. 运行环境
# ==========================================
if __name__ == "__main__":
    XML_PATH = "/root/mym/parkour-main/legged_gym/resources/robots/go2/urdf/go2_mjcf copy.xml"
    MODEL_PATH = "/root/mym/parkour-main/legged_gym/logs/field_go2/Feb04_02-53-08_Go2_10skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_leapHeight2.e-01_motorTorqueClip_fromFeb03_07-44-57/model_41000.pt"
    
    sim = Go2ParkourSim(XML_PATH, MODEL_PATH)
    
    # 使用 passive viewer，不阻塞主循环
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            sim.step()
            
            # 检查是否摔倒 (如果身体 Z 轴过低)
            if sim.data.qpos[2] < 0.15:
                print("机器人摔倒，重置中...")
                sim.reset()
            
            viewer.sync()
            
            # 维持实时率
            time_until_next_step = sim.model.opt.timestep * sim.decimation - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)