import mujoco
import numpy as np
import glfw
import cv2
import os
# --- 深度值转换工具函数 ---
def get_linear_depth(depth_buffer, model):
    extent = model.stat.extent
    znear = model.vis.map.znear * extent
    zfar = model.vis.map.zfar * extent
    depth_linear = znear / (1.0 - depth_buffer * (1.0 - znear / zfar))
    return depth_linear

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- 初始化 ---
resolution = (640, 480)
if not glfw.init():
    exit()

glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(resolution[0], resolution[1], "Offscreen Rendering", None, None)
glfw.make_context_current(window)

model = mujoco.MjModel.from_xml_path(f'{LEGGED_GYM_ROOT_DIR}/resources/robots/Ddog/Ddog_zrg.xml') 
data = mujoco.MjData(model)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# 1. 配置深度相机 (Fixed)
camera_name = "depth_camera"
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
depth_cam = mujoco.MjvCamera()
depth_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
depth_cam.fixedcamid = cam_id

# 2. 配置全局相机 (Free Camera)
global_cam = mujoco.MjvCamera()
# 你可以手动调整全局视角的位置
global_cam.distance = 2.5
global_cam.azimuth = 135
global_cam.elevation = -30
global_cam.lookat = np.array([0, 0, 0.2])

viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

print("正在运行：[Ddog Depth] 和 [Global View]...")

while True:
    mujoco.mj_step(model, data)

    # --- 渲染 1: 全局视角 (RGB) ---
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, global_cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    
    rgb_global = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb_global, None, viewport, context)
    bgr_global = cv2.cvtColor(np.flipud(rgb_global), cv2.COLOR_RGB2BGR)

    # --- 渲染 2: 深度相机视角 (Depth) ---
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, depth_cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    
    rgb_depth = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
    depth_buf = np.zeros((resolution[1], resolution[0], 1), dtype=np.float32)
    mujoco.mjr_readPixels(rgb_depth, depth_buf, viewport, context)
    
    # 处理深度
    depth_raw = np.flipud(depth_buf).squeeze()
    depth_meters = get_linear_depth(depth_raw, model)
    
    # 转换为灰色可视化
    max_dist = 5.0
    depth_norm = np.clip(depth_meters, 0, max_dist) / max_dist
    depth_gray = np.uint8((1.0 - depth_norm) * 255)

    # --- 显示 ---
    cv2.imshow('Global View (RGB)', bgr_global)
    cv2.imshow('Ddog Depth View (Gray)', depth_gray)

    if cv2.waitKey(1) == 27: # Esc 退出
        break

cv2.destroyAllWindows()
glfw.terminate()