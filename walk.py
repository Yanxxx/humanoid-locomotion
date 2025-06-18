import mujoco
import mujoco.viewer
import numpy as np

import time

# 加载模型
model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")  # 或者 .mjcf/.urdf 文件
data = mujoco.MjData(model)

print("qpos:", data.qpos)
print("qvel:", data.qvel)
print("ctrl:", data.ctrl)

# 找出所有可控关节
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
               for i in range(model.njnt)]
n_ctrl = model.nu  # 控制输入维度

print("所有关节：")
for i, name in enumerate(joint_names):
    print(f"  {i}: {name}")
    
print("控制通道数：", n_ctrl)

# 控制 loop
t = 0.0

# print(model.actuator_names)
print("Actuators:")
for i in range(model.nu):  # model.nu = number of control inputs
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  ctrl[{i}]: actuator name = '{name}'")

for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    joint_id = model.actuator_trnid[i][0]  # actuator_trnid 是二维数组
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    print(f"Actuator '{actuator_name}' controls joint '{joint_name}'")

with mujoco.viewer.launch_passive(model, data) as viewer:
# with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = 0.5 * np.sin(2 * np.pi * 0.5 * t)

        mujoco.mj_step(model, data)
        viewer.sync()
        t += model.opt.timestep
        time.sleep(model.opt.timestep)
        
        