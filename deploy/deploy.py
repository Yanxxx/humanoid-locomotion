import torch
import mujoco
import mujoco.viewer
import numpy as np
import time

# --- 1. Finalized Parameters (Aligned to the 37-DOF Policy) ---

# The policy was trained on a 37-DOF version of the G1 robot.
# This joint order is based on the G1_CFG from the unitree.py file.
JOINT_ORDER = [
    # This list must match the joint order in your new 37-DOF XML file.
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
    "torso_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    "left_five_joint", "left_three_joint", "left_six_joint", "left_four_joint",
    "left_zero_joint", "left_one_joint", "left_two_joint",
    "right_five_joint", "right_three_joint", "right_six_joint", "right_four_joint",
    "right_zero_joint", "right_one_joint", "right_two_joint"
]
NUM_ACTIONS = len(JOINT_ORDER)  # Must be 37

# The network's expected input size is 310.
NUM_OBS = 310
HEIGHT_SCAN_DIMS = 160

ACTION_SCALE = 0.5

# --- 2. MuJoCo Observation Function ---
def get_observation_from_mujoco(model, data, command, last_action, default_joint_pos):
    dof_pos = data.qpos[7:]
    dof_vel = data.qvel[6:]

    try:
        torso_id = model.body('torso_link').id
        root_rot_mat = data.xmat[torso_id].reshape(3, 3)
    except KeyError:
        pelvis_id = model.body('pelvis').id
        root_rot_mat = data.xmat[pelvis_id].reshape(3, 3)

    body_lin_vel = root_rot_mat.T @ data.qvel[:3]
    body_ang_vel = root_rot_mat.T @ data.qvel[3:6]
    projected_gravity = root_rot_mat.T @ np.array([0., 0., -9.81])
    joint_pos_rel = dof_pos - default_joint_pos
    joint_vel_rel = dof_vel
    height_scan = np.zeros(HEIGHT_SCAN_DIMS)

    known_obs_part = np.concatenate([
        body_lin_vel, body_ang_vel, projected_gravity,
        np.array(command), joint_pos_rel, joint_vel_rel,
        last_action, height_scan
    ])
    
    padding_len = NUM_OBS - len(known_obs_part)
    if padding_len < 0:
         raise ValueError(f"Observation vector length is {len(known_obs_part)}, which is greater than the expected {NUM_OBS}.")

    padding = np.zeros(padding_len)
    obs_vec = np.concatenate([known_obs_part, padding])
        
    return obs_vec

# --- 3. Main Deployment Script ---
def main_deployment():
    # Load the MuJoCo model that has 37 joints. 
    try:
        # You must use a model file that matches the policy. 
        mj_model = mujoco.MjModel.from_xml_path("../g1-37dof/scene.xml") 
    except FileNotFoundError:
        print("Error: g1.xml not found. Please ensure you have the correct 37-joint model file.")
        return
        
    mj_data = mujoco.MjData(mj_model)

    try:
        policy = torch.jit.load('checkpoints/policy_1800.pt', map_location='cpu')  
    except FileNotFoundError:
        print("Error: checkpoints/policy.pt not found.")  
        return
    policy.eval()  
    print("TorchScript policy model loaded successfully.")

    # Get default joint positions from the keyframe. 
    try:
        default_joint_pos = mj_model.key('stand').qpos[7:]  
    except KeyError:
        print("Error: Keyframe 'stand' not found in the XML model. Please add a keyframe.")
        return

    # This check is crucial. 
    if len(default_joint_pos) != NUM_ACTIONS:
        raise ValueError(f"CRITICAL ERROR: Mismatch between number of joints in XML keyframe ({len(default_joint_pos)}) and policy's expected actions ({NUM_ACTIONS}).")  

    last_action = np.zeros(NUM_ACTIONS)  
    command = [0.5, 0.0, 0.0]  
    
    # --- MODIFICATION START ---
    # Set the number of simulation steps per control step 
    sims_per_control_step = 4
    # Calculate the duration of one control step 
    # print("=============>", mj_model.opt.timestep)
    control_step_duration = mj_model.opt.timestep * sims_per_control_step
    # --- MODIFICATION END ---
    
    
    # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    #     print("\n[DEBUG] Running simulation without control to test model stability.")
    #     print("The robot should load and fall like a ragdoll.")
        
    #     # Set the robot to its initial keyframe pose once
    #     mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0) # 0 is the index of the first keyframe ('stand')

    #     while viewer.is_running():
    #         # Only step the physics, do not send any control commands
    #         mujoco.mj_step(mj_model, mj_data)
    #         viewer.sync()
            
            
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print("\nSimulation started. Robot commanded to walk forward.")
        while viewer.is_running():
            step_start_time = time.time()

            # --- CONTROL LOGIC (Runs once per control step) ---
            obs_np = get_observation_from_mujoco(mj_model, mj_data, command, last_action, default_joint_pos)  
            
            obs_tensor = torch.from_numpy(obs_np).float().unsqueeze(0)  
            with torch.no_grad():
                action_tensor = policy(obs_tensor)  
            
            last_action = action_tensor.cpu().numpy().flatten()  
            
            print("===========>", obs_np)
            print("###########>", last_action)
            
            # Now the shapes will match: (37,) + (37,) 
            target_joint_pos = default_joint_pos + last_action * ACTION_SCALE  
            
            mj_data.ctrl[:NUM_ACTIONS] = target_joint_pos  
            
            # --- SIMULATION LOGIC (Runs multiple times per control step) ---
            # --- MODIFICATION START ---
            for _ in range(sims_per_control_step):
                mujoco.mj_step(mj_model, mj_data)  
            # --- MODIFICATION END ---
            
            viewer.sync()
            
            # --- MODIFICATION START ---
            # Adjust sleep time to match the new control step duration 
            time.sleep(max(0, control_step_duration - (time.time() - step_start_time)))
            # --- MODIFICATION END ---

if __name__ == '__main__':
    main_deployment()
