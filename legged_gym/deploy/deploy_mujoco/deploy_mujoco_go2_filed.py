import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
from pynput import keyboard
import os 
import glfw
import cv2

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cmd_state = {
    "vx": 0.0,
    "vy": 0.0,
    "yaw": 0.0
}
def on_press(key):
    try:
        step_size = 0.1  # 每次按键增加的速度分量
        yaw_step = 0.2
        
        if key == keyboard.Key.up:
            cmd_state["vx"] += step_size
        elif key == keyboard.Key.down:
            cmd_state["vx"] -= step_size
        elif key == keyboard.Key.left:
            cmd_state["yaw"] += yaw_step
        elif key == keyboard.Key.right:
            cmd_state["yaw"] -= yaw_step
        elif key == keyboard.Key.space:
            cmd_state["vx"] = 0.0
            cmd_state["vy"] = 0.0
            cmd_state["yaw"] = 0.0
            print("\n[STOP] 指令已重置为 0")

        cmd_state["vx"] = np.clip(cmd_state["vx"], -1.8, 2)
        cmd_state["yaw"] = np.clip(cmd_state["yaw"], -1.2, 1.2)
        
        print(f"\r当前指令 -> vx: {cmd_state['vx']:.2f}, yaw: {cmd_state['yaw']:.2f}    ", end="")
    except Exception as e:
        pass

# 启动后台监听线程
listener = keyboard.Listener(on_press=on_press)
listener.start()

def _viewer_add_sphere(viewer, pos, size=0.015, rgba=(1.0, 1.0, 0.0, 0.8)):
    """Compatible marker drawing for mujoco.viewer Handle.
    Uses viewer.user_scn geoms (MuJoCo native mjv scene)."""
    if viewer is None:
        return

    # Some mujoco versions expose add_marker; prefer it if available.
    add_marker = getattr(viewer, "add_marker", None)
    if callable(add_marker):
        add_marker(
            pos=np.asarray(pos, dtype=np.float64),
            size=[size, size, size],
            rgba=list(rgba),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )
        return

    # Fallback: write into user_scn
    scn = getattr(viewer, "user_scn", None)
    if scn is None:
        return
    if scn.ngeom >= scn.maxgeom:
        return

    idx = scn.ngeom
    mujoco.mjv_initGeom(
        scn.geoms[idx],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, size, size], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1

def get_linear_depth(depth_buffer, model):
    extent = model.stat.extent
    znear = model.vis.map.znear * extent
    zfar = model.vis.map.zfar * extent
    depth_linear = znear / (1.0 - depth_buffer * (1.0 - znear / zfar))
    return depth_linear

def get_gravity_orientation(quaternion):
    # 保持你原始的四元数投影逻辑
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def _resolve_base_body_name(model):
    for name in ("base", "base_link"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return name
    raise ValueError("No base body found. Tried: 'base', 'base_link'.")

def get_height_scan(model, data, base_body_name, viewer=None, draw=False, num_points_x=21, num_points_y=11, step_x=0.1, step_y=0.1):
    """
    获取机器人下方的 231 维高程图 (Height Scan)。
    该函数执行射线检测 (Raycast)，返回探测点相对于机体(Base)的相对高度。
    """
    # 1. 获取机体当前的世界坐标和旋转矩阵
    base_pos = data.body(base_body_name).xpos.copy()
    base_body_id = model.body(base_body_name).id
    res_mat = np.zeros(9)
    mujoco.mju_quat2Mat(res_mat, data.qpos[3:7])
    rot_mat = res_mat.reshape(3, 3)
    
    # 2. 准备采样点参数（对齐 `deploy_mujoco_mym.py` / 训练）
    # x: [-0.5, 1.5] 共 21 点；y: [-0.5, 0.5] 共 11 点
    # 展平顺序：先 x 后 y（等价于 meshgrid(indexing="ij") 后 reshape(-1)）
    x_range = np.arange(-0.5, 1.51, step_x)
    y_range = np.arange(-0.5, 0.51, step_y)
    
    height_scan = np.zeros(num_points_x * num_points_y, dtype=np.float32)
    
    # 3. 射线检测配置 (严格符合 MuJoCo Python 绑定要求)
    g_id = np.zeros(1, dtype=np.int32)           # 用于接收命中的 geom id
    p_dir = np.array([0., 0., -1.], dtype=np.float64)  # 垂直向下发射
    ray_start_offset = 0.5                        # 从机器人上方 0.5m 往下打
    
    def _is_descendant_of_base(body_id):
        """Check if body_id belongs to robot subtree rooted at base."""
        while body_id > 0:
            if body_id == base_body_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return body_id == base_body_id

    # 4. 遍历网格进行探测
    counter = 0
    for dx in x_range:
        for dy in y_range:
            # 将网格点从机体局部坐标系投影到世界坐标系
            rel_sample_pos = np.array([dx, dy, 0], dtype=np.float64)
            world_sample_pos = base_pos + rot_mat @ rel_sample_pos
            
            # 射线起点 (世界坐标)
            p_start = world_sample_pos.copy()
            p_start[2] += ray_start_offset
            
            # 执行射线检测：忽略机器人本体，只看环境地形/地面
            ray_start = p_start.copy()
            ground_z = None
            for _ in range(6):
                dist = mujoco.mj_ray(model, data, ray_start, p_dir, None, 1, base_body_id, g_id)
                if dist <= 0:
                    break

                hit_geom = int(g_id[0])
                hit_body = int(model.geom_bodyid[hit_geom]) if hit_geom >= 0 else -1

                if _is_descendant_of_base(hit_body):
                    # 命中机器人：把起点下移到命中点下方一点，继续向下打
                    ray_start = ray_start + p_dir * (dist + 1e-4)
                    continue

                # 命中非机器人（地形/地面/障碍）
                ground_z = ray_start[2] - dist
                break

            if ground_z is not None:
                
                height_offset = 0.2 
                val = (base_pos[2] + height_offset) - ground_z

                height_scan[counter] = np.clip(val, -1.0, 1.0)

                # 可视化：把 231 个采样点(命中的地面点)画出来
                if viewer is not None and draw:
                    _viewer_add_sphere(
                        viewer,
                        pos=np.array([world_sample_pos[0], world_sample_pos[1], ground_z]),
                        size=0.015,
                        rgba=(1.0, 1.0, 0.0, 0.8),
                    )
            else:
                # 训练时不会给“无效点 sentinel”；这里用中性值 0 更贴近训练分布
                height_scan[counter] = 0.0
            
            counter += 1
                
    return height_scan
if __name__ == "__main__":
    # --- 1. 参数配置 ---
    num_actions, num_obs = 12, 279
    lin_vel_scale, ang_vel_scale = 1.0, 0.25
    dof_pos_scale, dof_vel_scale = 1.0, 0.05
    action_scale = 0.5 
    
    default_angles = np.array([0.1, 0.7, -1.5, -0.1, 0.7, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
    
    # 初始参数优化
    target_kps = np.full(12, 40.0) 
    kds = np.full(12, 1)          # 稍微调高阻尼，有助于吸收冲击
    control_decimation = 4 
    simulation_dt = 0.005 

    # --- 2. 环境初始化 ---
    model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/scene_with_stairs.xml"
    policy_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/go2/policy_gru_go2_field.pt"
    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_name = _resolve_base_body_name(m)

    # --- Depth camera setup (offscreen) ---
    depth_camera_name = "depth_camera"
    depth_resolution = (640, 480)  # (width, height)
    depth_camera_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, depth_camera_name)
    depth_window = None
    depth_scene = None
    depth_context = None
    depth_mj_camera = None
    depth_rgb = None
    depth_buffer = None
    depth_viewport = None
    depth_enabled = False

    if depth_camera_id == -1:
        print(f"[Warn] Camera '{depth_camera_name}' not found in XML.")
    else:
        print(f"[Info] Camera '{depth_camera_name}' found, id={depth_camera_id}")
        if glfw.init():
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            depth_window = glfw.create_window(
                depth_resolution[0], depth_resolution[1], "DepthOffscreen", None, None
            )
            if depth_window is not None:
                glfw.make_context_current(depth_window)
                depth_scene = mujoco.MjvScene(m, maxgeom=10000)
                depth_context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, depth_context)
                depth_rgb = np.zeros((depth_resolution[1], depth_resolution[0], 3), dtype=np.uint8)
                depth_buffer = np.zeros((depth_resolution[1], depth_resolution[0], 1), dtype=np.float32)
                depth_viewport = mujoco.MjrRect(0, 0, depth_resolution[0], depth_resolution[1])
                depth_mj_camera = mujoco.MjvCamera()
                depth_mj_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                depth_mj_camera.fixedcamid = depth_camera_id
                depth_enabled = True
            else:
                print("[Warn] Failed to create hidden GLFW window for depth rendering.")
        else:
            print("[Warn] glfw.init() failed, depth rendering disabled.")
    
    d.qpos[7:] = default_angles
    d.qpos[2] = 0.445              # Ddog 身体中心离地约 0.3-0.34m
    d.qvel[:] = 0                 # 初始速度彻底清零
    mujoco.mj_forward(m, d)       # 计算运动学，消除初始应力

    policy = torch.jit.load(policy_path)
    obs = np.zeros(num_obs, dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    # 指令设定
    target_cmd = np.array([0 , 0, 0]) 
    
    counter = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 设置初始相机角度（可选）
        viewer.cam.distance = 3.1  # 相机距离机器人的距离
        viewer.cam.azimuth = 132   # 水平角度
        viewer.cam.elevation = -20 # 俯仰角
        print("Control Ready: Use Arrow Keys to move, 'Space' to stop.")
        DRAW_HEIGHT_POINTS = True
        DRAW_HEIGHT_EVERY_N = control_decimation * 5  # 降低绘制频率，避免太卡
        while viewer.is_running():
            step_start = time.time()
            viewer.cam.lookat[:] = d.body(base_body_name).xpos
            current_kp_scale = min(1.0, d.time / 1.0)
            current_kps = target_kps * current_kp_scale
            
            # --- 3. 物理仿真 step ---
            tau = pd_control(target_dof_pos, d.qpos[7:], current_kps, 0, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            
            # --- 4. 策略推断 ---
            if counter % control_decimation == 0:
                res_mat = np.zeros(9)
                mujoco.mju_quat2Mat(res_mat, d.qpos[3:7])
                rot_mat = res_mat.reshape(3, 3)
                rot_mat_T = rot_mat.T # 机体坐标系基向量

                qj, dqj = d.qpos[7:], d.qvel[6:]
                quat, omega = d.qpos[3:7], d.qvel[3:6]
                lin_vel = d.qvel[:3]
                # 获取传感器线速度
                #base_lin_vel = d.sensor('base_lin_vel').data.copy()
                
                # 【补丁 3：初始观测过滤】
                # 前 0.5 秒即使身体有晃动，我们也告诉网络速度为 0，防止它产生过大的纠偏动作
                if d.time < 0.5:
                    input_lin_vel_world = np.zeros(3)
                    current_cmd = np.zeros(3)
                    target_dof_pos = default_angles
                    d.ctrl[:] = pd_control(target_dof_pos, d.qpos[7:], 80, 0, d.qvel[6:], 3) # 用更硬的KP站稳
                    action[:] = 0
                else:
                    input_lin_vel_world = d.qvel[:3]
                    current_cmd = np.array([cmd_state["vx"], cmd_state["vy"], cmd_state["yaw"]])
                # 构造 Observation
                obs[0:3] = (rot_mat_T @ input_lin_vel_world)* lin_vel_scale 
                obs[3:6] = omega * ang_vel_scale
                obs[6:9] = get_gravity_orientation(quat)
                obs[9:12] = current_cmd * np.array([lin_vel_scale, lin_vel_scale, ang_vel_scale])
                #obs[9:12]=0
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action
                draw_heights = DRAW_HEIGHT_POINTS and (counter % DRAW_HEIGHT_EVERY_N == 0)
                if draw_heights and getattr(viewer, "user_scn", None) is not None:
                    # Clear previous user geoms to avoid running out of slots.
                    viewer.user_scn.ngeom = 0
                obs[48:279] = get_height_scan(m, d, base_body_name, viewer=viewer, draw=draw_heights)
                #obs[48:279]= 0.53
                # 推理
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    new_action = policy(obs_tensor).detach().numpy().squeeze()

                action = new_action.copy()
                
                # 【修正 2：增加停止判定补丁】
                # 如果指令全为 0 且时间较长，可以尝试让 action 缓慢归零（防止原地踏步）
                if np.abs(current_cmd).sum() < 0.01 and d.time > 1.0:
                    # 逐渐收敛到默认姿态，而不是任由网络抖动
                    target_dof_pos = action * action_scale + default_angles
                elif d.time > 0.2:
                    target_dof_pos = action * action_scale + default_angles
                ###打印调试信息
                if counter % 100 == 0:
                    #  print(f"Time: {d.time:.2f} | KP_Scale: {current_kp_scale:.2f} | Z-Vel: {lin_vel[2]:.2f}")
                    #  print(obs[0:3])
                    #  print(d.qpos[2])
                     print(obs[48:279])
            counter += 1

            # --- Depth rendering ---
            if depth_enabled:
                mujoco.mjv_updateScene(
                    m, d, mujoco.MjvOption(), None, depth_mj_camera, mujoco.mjtCatBit.mjCAT_ALL, depth_scene
                )
                mujoco.mjr_render(depth_viewport, depth_scene, depth_context)
                mujoco.mjr_readPixels(depth_rgb, depth_buffer, depth_viewport, depth_context)

                depth_raw = np.flipud(depth_buffer).squeeze()
                depth_meters = get_linear_depth(depth_raw, m)
                max_dist = 5.0
                depth_norm = np.clip(depth_meters, 0, max_dist) / max_dist
                depth_gray = np.uint8((1.0 - depth_norm) * 255)
                cv2.imshow("Ddog Depth View (Gray)", depth_gray)
                if cv2.waitKey(1) == 27:
                    break

            viewer.sync()
            
            # 频率控制
            time_to_sleep = simulation_dt - (time.time() - step_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    # --- Cleanup depth resources ---
    if depth_enabled:
        cv2.destroyAllWindows()
        if depth_window is not None:
            glfw.destroy_window(depth_window)
        glfw.terminate()