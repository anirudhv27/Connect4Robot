import os

current_directory = os.getcwd()

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


NUM_CHIPS = 7
for i in range(1, NUM_CHIPS+1):
  height = (i - 1) * 0.1
  offset = 0.0825 * (i - 4)
  
  scenario_data += f"""
- add_model:
    name: red_chip_{i}
    file: file:///{current_directory}/connect4-assets/red_chip.sdf
    default_free_body_pose:
        red_chip:
            translation: [{offset}, 0.022, 0.6]
            rotation: !Rpy
                deg: [90, 0, 0]
                
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