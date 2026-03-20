""" Config to train the whole parkour oracle policy """
import numpy as np
from os import path as osp
from collections import OrderedDict

from legged_gym.envs.ddog.ddog_config import Ddog2RoughCfg, Ddog2RoughPPO

class DdogFieldCfg( Ddog2RoughCfg ):
    class init_state( Ddog2RoughCfg.init_state ):
        pos = [0.0, 0.0, 0.3]
        zero_actions = False

    class sensor( Ddog2RoughCfg.sensor):
        class proprioception( Ddog2RoughCfg.sensor.proprioception ):
            # latency_range = [0.0, 0.0]write
            latency_range = [0.005, 0.045] # [s]

    class terrain( Ddog2RoughCfg.terrain ):
        num_rows = 10
        num_cols = 40
        selected = "BarrierTrack"
        slope_treshold = 20.

        max_init_terrain_level = 2
        curriculum = True
        
        pad_unavailable_info = True
        BarrierTrack_kwargs = dict(
            options= [
                "jump",
                "leap",
                "hurdle",
                "down",
                "tilted_ramp",
                "stairsup",
                "stairsdown",
                "discrete_rect",
                "slope",
                "wave",
            ], # each race track will permute all the options
            jump= dict(
                height= [0.05, 0.5],
                depth= [0.1, 0.3],
                # fake_offset= 0.1,
            ),
            leap= dict(
                length= [0.05, 0.8],
                depth= [0.5, 0.8],
                height= 0.2, # expected leap height over the gap
                # fake_offset= 0.1,
            ),
            hurdle= dict(
                height= [0.05, 0.5],
                depth= [0.2, 0.5],
                # fake_offset= 0.1,
                curved_top_rate= 0.1,
            ),
            down= dict(
                height= [0.1, 0.6],
                depth= [0.3, 0.5],
            ),
            tilted_ramp= dict(
                tilt_angle= [0.2, 0.5],
                switch_spacing= 0.,
                spacing_curriculum= False,
                overlap_size= 0.2,
                depth= [-0.1, 0.1],
                length= [0.6, 1.2],
            ),
            slope= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-3.14, 0, 1.57, -1.57],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopeup= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopedown= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            stairsup= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                residual_distance= 0.05,
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            stairsdown= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            discrete_rect= dict(
                max_height= [0.05, 0.2],
                max_size= 0.6,
                min_size= 0.2,
                num_rects= 10,
            ),
            wave= dict(
                amplitude= [0.1, 0.15], # in meter
                frequency= [0.6, 1.0], # in 1/meter
            ),
            track_width= 3.2,
            track_block_length= 2.4,
            wall_thickness= (0.01, 0.6),
            wall_height= [-0.5, 2.0],
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 0.8,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
            randomize_obstacle_order= True,
            n_obstacles_per_track= 1,
        )

    class commands( Ddog2RoughCfg.commands ):
        # a mixture of command sampling and goal_based command update allows only high speed range
        # in x-axis but no limits on y-axis and yaw-axis
        lin_cmd_cutoff = 0.2
        class ranges( Ddog2RoughCfg.commands.ranges ):
            # lin_vel_x = [0.6, 1.8]
            lin_vel_x = [-0.6, 2.0]
        
        is_goal_based = True
        class goal_based:
            # the ratios are related to the goal position in robot frame
            x_ratio = None # sample from lin_vel_x range
            y_ratio = 1.2
            yaw_ratio = 1.
            follow_cmd_cutoff = True
            x_stop_by_yaw_threshold = 1. # stop when yaw is over this threshold [rad]

    class asset( Ddog2RoughCfg.asset ):
        terminate_after_contacts_on = []
        penalize_contacts_on = ["thigh", "calf", "base"]

    class termination( Ddog2RoughCfg.termination ):
        roll_kwargs = dict(
            threshold= 1.4, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad]
        )
        timeout_at_border = True
        timeout_at_finished = False

    class rewards( Ddog2RoughCfg.rewards ):
        class scales:
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            energy_substeps = -2e-7
            torques = -1e-7
            stand_still = -1.
            dof_error_named = -1.
            dof_error = -0.005
            collision = -0.05
            lazy_stop = -3.
            # penalty for hardware safety
            exceed_dof_pos_limits = -0.1
            exceed_torque_limits_l1norm = -0.1
            # penetration penalty
            penetrate_depth = -0.05

    class noise( Ddog2RoughCfg.noise ):
        add_noise = False

    class curriculum:
        penetrate_depth_threshold_harder = 100
        penetrate_depth_threshold_easier = 200
        no_moveup_when_fall = True
    
logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")

class Ddog2FieldCfgPPO( Ddog2RoughPPO ):
    class algorithm( Ddog2RoughPPO.algorithm ):
        entropy_coef = 0.0

    class runner( Ddog2RoughPPO.runner ):
        experiment_name = "field_ddog"

        resume = True
        load_run = osp.join(logs_root, "field_ddog",
            "/root/mym/parkour-main/legged_gym/logs/field_ddog/Feb25_02-19-27_Ddog_10skills_pEnergy2.e-07_pTorques-1.e-07_pLazyStop-3.e+00_pPenD5.e-02_penEasier200_penHarder100_leapHeight2.e-01_motorTorqueClip_fromFeb11_08-00-01",
        )

        run_name = "".join(["Ddog_",
            ("{:d}skills".format(len(DdogFieldCfg.terrain.BarrierTrack_kwargs["options"]))),
            ("_pEnergy" + np.format_float_scientific(-DdogFieldCfg.rewards.scales.energy_substeps, precision=2)),
            # ("_pDofErr" + np.format_float_scientific(-Ddog2RoughCfg.rewards.scales.dof_error, precision=2) if getattr(Ddog2RoughCfg.rewards.scales, "dof_error", 0.) != 0. else ""),
            # ("_pHipDofErr" + np.format_float_scientific(-Ddog2RoughCfg.rewards.scales.dof_error_named, precision=2) if getattr(Ddog2RoughCfg.rewards.scales, "dof_error_named", 0.) != 0. else ""),
            # ("_pStand" + np.format_float_scientific(Ddog2RoughCfg.rewards.scales.stand_still, precision=2)),
            # ("_pTerm" + np.format_float_scientific(Ddog2RoughCfg.rewards.scales.termination, precision=2) if hasattr(Ddog2RoughCfg.rewards.scales, "termination") else ""),
            ("_pTorques" + np.format_float_scientific(DdogFieldCfg.rewards.scales.torques, precision=2) if hasattr(DdogFieldCfg.rewards.scales, "torques") else ""),
            # ("_pColl" + np.format_float_scientific(Ddog2RoughCfg.rewards.scales.collision, precision=2) if hasattr(Ddog2RoughCfg.rewards.scales, "collision") else ""),
            ("_pLazyStop" + np.format_float_scientific(DdogFieldCfg.rewards.scales.lazy_stop, precision=2) if hasattr(DdogFieldCfg.rewards.scales, "lazy_stop") else ""),
            # ("_trackSigma" + np.format_float_scientific(Ddog2RoughCfg.rewards.tracking_sigma, precision=2) if Ddog2RoughCfg.rewards.tracking_sigma != 0.25 else ""),
            # ("_pPenV" + np.format_float_scientific(-Ddog2RoughCfg.rewards.scales.penetrate_volume, precision=2)),
            ("_pPenD" + np.format_float_scientific(-DdogFieldCfg.rewards.scales.penetrate_depth, precision=2)),
            # ("_pTorqueL1" + np.format_float_scientific(-Ddog2RoughCfg.rewards.scales.exceed_torque_limits_l1norm, precision=2)),
            ("_penEasier{:d}".format(DdogFieldCfg.curriculum.penetrate_depth_threshold_easier)),
            ("_penHarder{:d}".format(DdogFieldCfg.curriculum.penetrate_depth_threshold_harder)),
            # ("_leapMin" + np.format_float_scientific(Ddog2RoughCfg.terrain.BarrierTrack_kwargs["leap"]["length"][0], precision=2)),
            ("_leapHeight" + np.format_float_scientific(DdogFieldCfg.terrain.BarrierTrack_kwargs["leap"]["height"], precision=2)),
            ("_motorTorqueClip" if DdogFieldCfg.control.motor_clip_torque else ""),
            # ("_noMoveupWhenFall" if Ddog2RoughCfg.curriculum.no_moveup_when_fall else ""),
            ("_noResume" if not resume else "_from" + "_".join(load_run.split("/")[-1].split("_")[:2])),
        ])

        max_iterations = 38000
        save_interval = 500
        log_interval = 100
        