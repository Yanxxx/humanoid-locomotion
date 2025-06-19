import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
# MODIFY THIS PATH
XML_PATH = '../g1/scene.xml' 

# Load the model and data
try:
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
except FileNotFoundError:
    print(f"Error: XML file not found at '{XML_PATH}'")
    exit()

# --- Gait Parameters ---
WALK_FREQ = 1.5              # Steps per second
STEP_HEIGHT = 0.08           # How high to lift the foot (in radians for the knee)
HIP_PITCH_AMP = 0.4          # Forward/backward leg swing amplitude (radians)
HIP_ROLL_AMP = 0.1           # Side-to-side hip swing for balance (radians)
ANKLE_PITCH_COMP = 0.2       # Ankle compensation to keep foot parallel (radians)
ARM_PITCH_AMP = 0.3          # Arm swing amplitude (radians)

# --- Actuator Mapping (CORRECTED SECTION) ---
# Create a dictionary to map actuator names to their integer IDs.
# We iterate from 0 to model.nu-1 and get the name for each actuator ID.
actuator_map = {model.actuator(i).name: i for i in range(model.nu)}

# Define the actuators we will be controlling for the gait
leg_actuators = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_knee_joint', 'left_ankle_pitch_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_pitch_joint'
]
arm_actuators = [
    'left_shoulder_pitch_joint', 'left_elbow_joint',
    'right_shoulder_pitch_joint', 'right_elbow_joint'
]

# Get a list of all other actuators to hold them steady
all_controlled_actuators = leg_actuators + arm_actuators
other_actuator_indices = [idx for name, idx in actuator_map.items() if name not in all_controlled_actuators]

# --- Simulation ---
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # Initialize a full control array
    ctrl = np.zeros(model.nu)

    while viewer.is_running():
        sim_time = time.time() - start_time
        
        # --- Gait Generation ---
        # Phase determines the point in the walking cycle (0 to 2*pi)
        phase = 2 * np.pi * WALK_FREQ * sim_time
        
        # Left Leg and Right Arm Phase (they swing forward together)
        phase_L_leg_R_arm = np.sin(phase)
        
        # Right Leg and Left Arm Phase (offset by 180 degrees)
        phase_R_leg_L_arm = np.sin(phase + np.pi)

        # Calculate joint angles based on phase
        left_hip_pitch = HIP_PITCH_AMP * phase_L_leg_R_arm
        right_hip_pitch = HIP_PITCH_AMP * phase_R_leg_L_arm
        left_knee = STEP_HEIGHT * (1 - np.cos(phase)) / 2
        right_knee = STEP_HEIGHT * (1 - np.cos(phase + np.pi)) / 2
        hip_roll = HIP_ROLL_AMP * np.cos(phase)
        left_arm_pitch = ARM_PITCH_AMP * phase_R_leg_L_arm
        right_arm_pitch = ARM_PITCH_AMP * phase_L_leg_R_arm

        # --- Set Control Signals ---
        # Set all other actuators to 0 (neutral position)
        ctrl[other_actuator_indices] = 0.0

        # Left Leg
        ctrl[actuator_map['left_hip_pitch_joint']] = left_hip_pitch
        ctrl[actuator_map['left_knee_joint']] = left_knee
        ctrl[actuator_map['left_hip_roll_joint']] = -hip_roll  # Roll outwards
        ctrl[actuator_map['left_ankle_pitch_joint']] = -left_knee # Simple ankle compensation

        # Right Leg
        ctrl[actuator_map['right_hip_pitch_joint']] = right_hip_pitch
        ctrl[actuator_map['right_knee_joint']] = right_knee
        ctrl[actuator_map['right_hip_roll_joint']] = -hip_roll
        ctrl[actuator_map['right_ankle_pitch_joint']] = -right_knee

        # Arms
        ctrl[actuator_map['left_shoulder_pitch_joint']] = left_arm_pitch
        ctrl[actuator_map['right_shoulder_pitch_joint']] = right_arm_pitch
        ctrl[actuator_map['left_elbow_joint']] = 0.2 # Slightly bent elbow
        ctrl[actuator_map['right_elbow_joint']] = 0.2

        # Apply control
        data.ctrl[:] = ctrl
        
        # Step simulation and synchronize viewer
        mujoco.mj_step(model, data)
        viewer.sync()
