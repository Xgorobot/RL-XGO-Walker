# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class xgoCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 48
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.13] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'fl_hip_joint': 0.1,   # [rad]
            'bl_hip_joint': 0.1,   # [rad]
            'fr_hip_joint': -0.1 ,  # [rad]
            'br_hip_joint': -0.1,   # [rad]

            'fl_thigh_joint': 0.6,     # [rad]
            'bl_thigh_joint': 0.6,   # [rad]
            'fr_thigh_joint': 0.6,     # [rad]
            'br_thigh_joint': 0.6,   # [rad]

            'fl_calf_joint': -1.5,   # [rad]
            'bl_calf_joint': -1.5,    # [rad]
            'fr_calf_joint': -1.5,  # [rad]
            'br_calf_joint': -1.5,    # [rad]
        }
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = False
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 2.45}  # [N*m/rad]
        damping = {'joint': 0.042}     # [N*m*s/rad]#0.005
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4*5
    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [-0.0, 0.3] # min max [m/s]
            lin_vel_y = [-0.1, 0.1]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/xgo/urdf/xgo.urdf'
        name = "xgo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "hip", "body", "calf", "hand", "arm", "thigh", "right", "left"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        # fix_base_link = True
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.05
    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 10.0
            ang_vel = 0.5
            dof_pos = 1.0
            dof_vel = 0.1
            height_measurements = 5.0
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.1
        tracking_sigma = 0.05#0.05
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.2
            dof_pos_limits = -1.0
            feet_air_time = 3
            action_rate = -0.12
            dof_acc = -2.5e-7
            tracking_lin_vel = 2.5#2.5
            tracking_ang_vel = 1
            mimic = 2
    class sim(LeggedRobotCfg.sim):
        dt=0.001
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 4
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 6###up
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.00   # [m] up
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class xgoCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'xgo'
