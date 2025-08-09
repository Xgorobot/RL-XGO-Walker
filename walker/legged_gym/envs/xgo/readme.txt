5.12 20:47
feet_airtime reward 1->2
rew_airTime = torch.sum((self.feet_air_time - 0.1) * first_contact, dim=1)  0.05->0.1

5.12 21:23
action_rate = -0.16 0.08->0.16(too large) 0.12(too large)
#rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)  0.1->0.3
action_rate = -0.1

5.13 09:49
tracking_sigma = 0.03  #0.05->0.03
feet_air_time = 3 2->3
rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)  0.3->0.1
tracking_lin_vel = 2.5 2->2.5

5.13 10:31
action_rate = -0.1 0.1->0.12
tracking_lin_vel = 2.5 2->2.5
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.15] # x,y,z [m]
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
