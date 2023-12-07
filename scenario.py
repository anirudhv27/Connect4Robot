import os
from pydrake.all import RigidTransform, RollPitchYaw, Quaternion, AngleAxis
import numpy as np

current_directory = os.getcwd()

robot1_pose = RigidTransform()
robot1_pose.set_translation([0, 0.5, 0])
rot = AngleAxis()
rot.set_angle(180 * np.pi / 2)
rot.set_axis([0, 0, 1])
robot1_pose.set_rotation(rot)

robot2_pose = RigidTransform()
robot2_pose.set_translation([0, 0.5, 0])

scenario_data = f"""
directives:
- add_directives:
    file: file:///{current_directory}/connect4-assets/robots.yaml
- add_model:
    name: connect4
    file: file:///{current_directory}/connect4-assets/connect4-convex.sdf
- add_weld:
    parent: world
    child: connect4
    X_PC:
        translation: [0, -0.25, 0]
        
- add_model:
    name: table_top
    file: file:///{current_directory}/connect4-assets/table_top.sdf
- add_weld:
    parent: world
    child: table_top::table_top_center
"""


NUM_CHIPS = 5
for i in range(1, NUM_CHIPS+1):
  x_coord = 0.0825 * ((i - 1) % 5) + 0.35
  y_coord = 0.0825 * ((i - 1) // 5) + 0.35
    
  scenario_data += f"""
- add_model:
    name: red_chip_{i}
    file: file:///{current_directory}/connect4-assets/red_chip.sdf
    default_free_body_pose:
        red_chip:
            translation: [{x_coord}, {y_coord}, 0.05]
            rotation: !Rpy
                deg: [0, 0, 0]
                
- add_model:
    name: yellow_chip_{i}
    file: file:///{current_directory}/connect4-assets/yellow_chip.sdf
    default_free_body_pose:
        yellow_chip:
            translation: [{-x_coord}, {-y_coord}, 0.05]
            rotation: !Rpy
                deg: [0, 0, 0]
                
  """
    
scenario_data += """
model_drivers:
    iiwa1: !IiwaDriver
      hand_model_name: wsg1
    wsg1: !SchunkWsgDriver {}
    iiwa2: !IiwaDriver
      hand_model_name: wsg2
    wsg2: !SchunkWsgDriver {}
"""