from pydrake.all import RigidTransform, RollPitchYaw
import numpy as np

col_poses_yellow = []

col_1_pose = RigidTransform()
col_1_pose.set_translation([3*0.0825, 0.125, 0.7])
col_1_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_yellow.append(col_1_pose)

col_2_pose = RigidTransform()
col_2_pose.set_translation([2*0.0825, 0.125, 0.7])
col_2_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_yellow.append(col_2_pose)

col_3_pose = RigidTransform()
col_3_pose.set_translation([0.0825, 0.125, 0.7])
col_3_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_yellow.append(col_3_pose)

col_4_pose = RigidTransform()
col_4_pose.set_translation([0, 0.125, 0.7])
col_4_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_yellow.append(col_4_pose)

col_5_pose = RigidTransform()
col_5_pose.set_translation([-0.0825, 0.125, 0.7])
col_5_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_yellow.append(col_5_pose)

col_6_pose = RigidTransform()
col_6_pose.set_translation([-2*0.0825, 0.125, 0.7])
col_6_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_yellow.append(col_6_pose)

col_7_pose = RigidTransform()
col_7_pose.set_translation([-3*0.0825, 0.125, 0.7])
col_7_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_yellow.append(col_7_pose)

col_poses_red = []

col_1_pose = RigidTransform()
col_1_pose.set_translation([3*0.0825, 0.15, 0.7])
col_1_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_red.append(col_1_pose)

col_2_pose = RigidTransform()
col_2_pose.set_translation([2*0.0825, 0.15, 0.7])
col_2_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_red.append(col_2_pose)

col_3_pose = RigidTransform()
col_3_pose.set_translation([0.0825, 0.15, 0.7])
col_3_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_red.append(col_3_pose)

col_4_pose = RigidTransform()
col_4_pose.set_translation([0, 0.15, 0.7])
col_4_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_red.append(col_4_pose)

col_5_pose = RigidTransform()
col_5_pose.set_translation([-0.0825, 0.15, 0.7])
col_5_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_red.append(col_5_pose)

col_6_pose = RigidTransform()
col_6_pose.set_translation([-2*0.0825, 0.15, 0.7])
col_6_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_red.append(col_6_pose)

col_7_pose = RigidTransform()
col_7_pose.set_translation([-3*0.0825, 0.15, 0.7])
col_7_pose.set_rotation(RollPitchYaw([0, 0, np.pi]))
col_poses_red.append(col_7_pose)