from urllib.request import urlretrieve
import numpy as np
from IPython.display import HTML
from pydrake.all import (
    AbstractValue,
    AngleAxis,
    ConstantVectorSource,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MeshcatVisualizer,
    RollPitchYaw,
    MathematicalProgram,
    RandomGenerator,
    RigidTransform,
    UniformlyRandomRotationMatrix,
    Simulator,
    SnoptSolver,
    StartMeshcat,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    ge,
    le,
)
import re
import random

from pydrake.systems.framework import GenerateHtml
from manipulation.exercises.robot.test_hardware_station_io import (
    TestHardwareStationIO,
)
from manipulation.scenarios import AddIiwaDifferentialIK, AddRgbdSensors
from manipulation.station import MakeHardwareStation, load_scenario

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os

import matplotlib.pyplot as plt

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

scenario_data += """
cameras:
    camera0:
      name: camera0
      depth: True
      X_PB:
        base_frame: camera0::base
    
    camera1:
      name: camera1
      depth: True
      X_PB:
        base_frame: camera1::base
"""

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
class PoseSource(LeafSystem):
    def __init__(self, pose):
        LeafSystem.__init__(self)
        self._pose = pose
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.CalcOutput
        )

    def CalcOutput(self, context, output):
        output.set_value(self._pose)
        
    def set_pose(self, new_pose):
        self._pose = new_pose

class Connect4Game:
    
    def __init__(self, meshcat): # Initialize Diagram with HardwareStation, 
        self.meshcat = meshcat
        self.set_game_state()        
        self.build_diagram()
        self.init_frames()        
        self.init_robot()        
        self.init_simulator()
        self.scan_piece_positions()
    
    def set_game_state(self):
        self.next_red_coin = 0
        self.next_yellow_coin = 0
        self.curr_player = 0
        
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
    
    def prefinalize_callback(self, parser):
        self.init_vacuum_constraints(parser)        
    
    def init_vacuum_constraints(self, parser):
        plant = parser.plant()
        self.yellow_chip_constraints = [] # wsg 1
        self.red_chip_constraints = [] # wsg2
        
        wsg1_model_instance = plant.GetModelInstanceByName("wsg1")
        wsg1_tip = plant.GetBodyByName("body", wsg1_model_instance)
        
        wsg2_model_instance = plant.GetModelInstanceByName("wsg2")
        wsg2_tip = plant.GetBodyByName("body", wsg2_model_instance)
        
        pose = RigidTransform()
        pose.set_translation([0, 0.15, 0])
        pose.set_rotation(RollPitchYaw([np.pi/2, 0, 0]))
        
        # Yellow Chips
        for i in range(1, NUM_CHIPS + 1):
            model_instance = plant.GetModelInstanceByName(f"yellow_chip_{i}")
            body = plant.GetBodyByName("yellow_chip", model_instance)
            
            # wsg1_constraint_id = plant.AddDistanceConstraint(body, [0, 0, 0], wsg1_tip, [0, 0.15, 0], 0.01)
            wsg1_constraint_id = plant.AddWeldConstraint(body, RigidTransform(), wsg1_tip, pose)
            
            self.yellow_chip_constraints.append(wsg1_constraint_id)
        
        # Red Chips
        for i in range(1, NUM_CHIPS + 1):
            model_instance = plant.GetModelInstanceByName(f"red_chip_{i}")
            body = plant.GetBodyByName("red_chip", model_instance)
            
            # wsg2_constraint_id = plant.AddDistanceConstraint(body, [0, 0, 0], wsg2_tip, [0.1, 0, 0], 0.01)
            wsg2_constraint_id = plant.AddWeldConstraint(body, RigidTransform(), wsg2_tip, pose)
            
            self.red_chip_constraints.append(wsg2_constraint_id)
    
    def build_diagram(self):
        # Build diagram
        self.builder = DiagramBuilder()
        self.scenario = load_scenario(data=scenario_data)

        self.station = self.builder.AddSystem(MakeHardwareStation(self.scenario, meshcat=meshcat, parser_prefinalize_callback=self.prefinalize_callback))
        self.plant = self.station.GetSubsystemByName("plant")
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        
        self.controller_plant_1 = self.station.GetSubsystemByName(
            "iiwa1.controller"
        ).get_multibody_plant_for_control()

        self.controller_plant_2 = self.station.GetSubsystemByName(
            "iiwa2.controller"
        ).get_multibody_plant_for_control()

        pose1 = RigidTransform()
        pose1.set_translation([0.4, 0, 0.55])
        pose1.set_rotation(RollPitchYaw([-np.pi/2, 0, np.pi/2]))
        self.pose1_source = self.builder.AddSystem(PoseSource(pose1))
        controller1 = AddIiwaDifferentialIK(
            self.builder,
            self.controller_plant_1, 
            frame=self.controller_plant_1.GetFrameByName("body")
        )

        self.builder.Connect(
            self.pose1_source.get_output_port(),
            controller1.get_input_port(0),
        )

        self.builder.Connect(
            self.station.GetOutputPort("iiwa1.state_estimated"),
            controller1.GetInputPort("robot_state"),
        )

        self.builder.Connect(
            controller1.get_output_port(),
            self.station.GetInputPort("iiwa1.position"),
        )

        pose2 = RigidTransform()
        pose2.set_translation([0.4, 0, 0.55])
        pose2.set_rotation(RollPitchYaw([-np.pi/2, 0, np.pi/2]))
        self.pose2_source = self.builder.AddSystem(PoseSource(pose2))
        controller2 = AddIiwaDifferentialIK(
            self.builder,
            self.controller_plant_2, 
            frame=self.controller_plant_2.GetFrameByName("body")
        )

        self.builder.Connect(
            self.pose2_source.get_output_port(),
            controller2.get_input_port(0),
        )

        self.builder.Connect(
            self.station.GetOutputPort("iiwa2.state_estimated"),
            controller2.GetInputPort("robot_state"),
        )

        self.builder.Connect(
            controller2.get_output_port(),
            self.station.GetInputPort("iiwa2.position"),
        )

        wsg1_position = self.builder.AddSystem(ConstantVectorSource([0]))
        self.builder.Connect(
            wsg1_position.get_output_port(),
            self.station.GetInputPort("wsg1.position"),
        )

        wsg2_position = self.builder.AddSystem(ConstantVectorSource([0]))
        self.builder.Connect(
            wsg2_position.get_output_port(),
            self.station.GetInputPort("wsg2.position"),
        )

        self.visualizer = MeshcatVisualizer.AddToBuilder(
            self.builder, self.station.GetOutputPort("query_object"), meshcat
        )

        self.diagram = self.builder.Build() 
    
    def init_frames(self):
        self.gripper_frame_1 = self.controller_plant_1.GetFrameByName("body")
        self.gripper_frame_2 = self.controller_plant_2.GetFrameByName("body")
        self.world_frame = self.plant.world_frame()
    
    def init_robot(self):
        # Set Context
        self.context = self.diagram.CreateDefaultContext()
        
        self.plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, self.context
        )

        all_constraints = self.red_chip_constraints + self.yellow_chip_constraints
        for constraint_id in all_constraints:
            if constraint_id is not None:
                self.plant.SetConstraintActiveStatus(self.plant_context, constraint_id, False)

        # provide initial states
        q0 = np.array(
            [
                1.40666193e-05,
                1.56461165e-01,
                -3.82761069e-05,
                -1.32296976e00,
                -6.29097287e-06,
                1.61181157e00,
                -2.66900985e-05,
            ]
        )

        # set the joint positions of both kuka arms
        self.iiwa1 = self.plant.GetModelInstanceByName("iiwa1")
        self.plant.SetPositions(self.plant_context, self.iiwa1, q0)
        self.plant.SetVelocities(self.plant_context, self.iiwa1, np.zeros(7))
        self.wsg1 = self.plant.GetModelInstanceByName("wsg1")
        self.plant.SetPositions(self.plant_context, self.wsg1, [-0.001, 0.001])
        self.plant.SetVelocities(self.plant_context, self.wsg1, [0, 0])

        self.iiwa2 = self.plant.GetModelInstanceByName("iiwa2")
        self.plant.SetPositions(self.plant_context, self.iiwa2, q0)
        self.plant.SetVelocities(self.plant_context, self.iiwa2, np.zeros(7))
        self.wsg2 = self.plant.GetModelInstanceByName("wsg2")
        self.plant.SetPositions(self.plant_context, self.wsg2, [-0.001, 0.001])
        self.plant.SetVelocities(self.plant_context, self.wsg2, [0, 0])   
        
        self.velocity_1 = self.plant.GetVelocities(self.plant_context, self.iiwa1)
        self.velocity_2 = self.plant.GetVelocities(self.plant_context, self.iiwa2)

        self.diagram.ForcedPublish(self.context)
        
        # Get other info about the camera
        red_cam = self.station.GetSubsystemByName("rgbd_sensor_camera0")
        red_cam_context = red_cam.GetMyMutableContextFromRoot(self.context)
        self.X_WC_red = red_cam.body_pose_in_world_output_port().Eval(red_cam_context)
        self.red_cam_info = red_cam.depth_camera_info()
        
        # Get other info about the camera
        yellow_cam = self.station.GetSubsystemByName("rgbd_sensor_camera1")
        yellow_cam_context = yellow_cam.GetMyMutableContextFromRoot(self.context)
        self.X_WC_yellow = yellow_cam.body_pose_in_world_output_port().Eval(yellow_cam_context)
        self.yellow_cam_info = yellow_cam.depth_camera_info()
    
    def init_simulator(self):
        self.simulator_time = 0.1
        self.simulator = Simulator(self.diagram, self.context)
        self.simulator.set_target_realtime_rate(1.0)
        self.simulator.AdvanceTo(self.simulator_time);
        
    def get_gripper_velocity(self, robot_num):
        self.velocity_1 = self.plant.GetVelocities(self.plant_context, self.iiwa1)
        self.velocity_2 = self.plant.GetVelocities(self.plant_context, self.iiwa2)
        if robot_num == 0:
            return self.velocity_1
        if robot_num == 1:
            return self.velocity_2
    
    def advance_till_stop(self, robot_num):
        self.advance_time(0.3)
        # Advance until it stops moving
        threshold = 1e-2
        velocity = self.get_gripper_velocity(robot_num)
        while any(abs(v) >= threshold for v in velocity):
            self.advance_time()
            velocity = self.get_gripper_velocity(robot_num)
            
    def generate_color_map(self, n):
        return [tuple(random.sample(range(256), 3)) for _ in range(n)]
    
    def visualize_components(self, labeled_mask, num_components):
        color_map = self.generate_color_map(num_components + 2)  # +2 for background and white color
        color_map[0] = (0, 0, 0)  # Background color (black)
        color_map[1] = (255, 255, 255)  # White color

        # Create an RGB image
        height, width = labeled_mask.shape
        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                color_image[i, j] = color_map[labeled_mask[i, j]]

        return color_image
        
    def scan_piece_positions(self):
        # print("scanning images")
        # Read color and depth images from each side
        index = 1
        station_context = self.station.CreateDefaultContext()
        color_image_red = self.station.GetOutputPort(
            f"camera0.rgb_image"
        ).Eval(station_context)
        depth_image_red = self.station.GetOutputPort(
            f"camera0.depth_image"
        ).Eval(station_context)

        color_image_yellow = self.station.GetOutputPort(
            f"camera1.rgb_image"
        ).Eval(station_context)
        depth_image_yellow = self.station.GetOutputPort(
            f"camera1.depth_image"
        ).Eval(station_context)
        
        body_poses = self.station.GetOutputPort(
            f"body_poses"
        ).Eval(station_context)

        # print("segmenting images")
        # Segment grey from color images to get segmentation mask
        yellow_image = color_image_yellow.data
        yellow_mask = (yellow_image[:, :, 0] == 148) & (yellow_image[:, :, 1] == 148) & (yellow_image[:, :, 2] == 148)
        yellow_mask_img = ~yellow_mask
        yellow_mask_img = np.uint8(yellow_mask_img)
        
        red_image = color_image_red.data
        red_mask = (red_image[:, :, 0] == 148) & (red_image[:, :, 1] == 148) & (red_image[:, :, 2] == 148)
        red_mask_img = ~red_mask
        red_mask_img = np.uint8(red_mask_img)
        
        # print("finding chips in frame with floodfill")
        # Find the number of chips in the frame, along with the average coordinate of each
        labeled_mask_iterative_red, num_components_iterative_red = self.label_components_iterative(red_mask_img)
        average_coordinates_red = self.average_pixel_coordinates(labeled_mask_iterative_red, num_components_iterative_red)
        color_seg_red = self.visualize_components(labeled_mask_iterative_red, num_components_iterative_red)
        plt.imshow(color_seg_red)
        plt.show()
        

        labeled_mask_iterative_yellow, num_components_iterative_yellow = self.label_components_iterative(yellow_mask_img)
        average_coordinates_yellow = self.average_pixel_coordinates(labeled_mask_iterative_yellow, num_components_iterative_yellow)
        color_seg_yellow = self.visualize_components(labeled_mask_iterative_yellow, num_components_iterative_yellow)
        plt.imshow(color_seg_yellow)
        plt.show()
        
        # Convert pixel coords to 3d coords in camera frame
        red_depths = np.array([np.array([u, v, depth_image_red.data[u, v].item()]) for u, v in average_coordinates_red])
        yellow_depths = np.array([np.array([u, v, depth_image_yellow.data[u, v].item()]) for u, v in average_coordinates_yellow])
        
        # Unproject camera coords for each (u, v, depth)
        print("red cam pose", self.X_WC_red)
        print("red cam info", self.red_cam_info)
        
        print("yellow cam pose", self.X_WC_yellow)
        print("yellow cam info", self.yellow_cam_info)
        
        red_points_cam = self.project_depth_to_pC(red_depths, self.red_cam_info)
        yellow_points_cam = self.project_depth_to_pC(yellow_depths, self.yellow_cam_info)
        
        red_points_world = self.X_WC_red @ red_points_cam.T
        yellow_points_world = self.X_WC_yellow @ yellow_points_cam.T
        
        self.red_chip_positions = red_points_world.T
        self.yellow_chip_positions = yellow_points_world.T
        
    def project_depth_to_pC(self, depth_pixel, cam_info):
        v = depth_pixel[:, 0]
        u = depth_pixel[:, 1]
        Z = depth_pixel[:, 2]
        cx = cam_info.center_x()
        cy = cam_info.center_y()
        fx = cam_info.focal_x()
        fy = cam_info.focal_y()
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        pC = np.c_[X, Y, Z]
        return pC

    # Step 2: Iterative flood fill algorithm to label the components
    def flood_fill_iterative(self, mask, x, y, label):
        if x < 0 or x >= mask.shape[0] or y < 0 or y >= mask.shape[1] or mask[x, y] != 1:
            return  # Early exit if start point is invalid

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # print(x, y, mask[x, y])
            if x < 0 or x >= mask.shape[0] or y < 0 or y >= mask.shape[1]:
                continue
            if mask[x, y] == 1:
                mask[x, y] = label
                stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
    
    def label_components_iterative(self, binary_mask):
        # print("labeling components")
        labeled_mask = np.copy(binary_mask)
        label_count = 2  # Start labeling from 2 (since 1 is for white pixels)
        for i in range(labeled_mask.shape[0]):
            for j in range(labeled_mask.shape[1]):
                if labeled_mask[i, j] == 1:
                    self.flood_fill_iterative(labeled_mask, i, j, label_count)
                    label_count += 1
        return labeled_mask, label_count - 2

    # Step 3: Calculate the average pixel coordinates for each component
    def average_pixel_coordinates(self, labeled_mask, num_components):
        coord_sums = {label: np.array([0, 0]) for label in range(2, num_components + 2)}
        counts = {label: 0 for label in range(2, num_components + 2)}
        for i in range(labeled_mask.shape[0]):
            for j in range(labeled_mask.shape[1]):
                label = labeled_mask[i, j]
                if label > 1:  # Skip background and white color
                    coord_sums[label] += np.array([i, j])
                    counts[label] += 1
        avg_coords = {label: (coord_sums[label] / counts[label]).astype(int) for label in coord_sums}
        avg_coords_list = [tuple(avg_coords[label]) for label in sorted(avg_coords)]
        return avg_coords_list

    def plot_manipulation_station_camera_images(self):
        index = 1
        station_context = self.station.CreateDefaultContext()
        color_image_0 = self.station.GetOutputPort(
            f"camera0.rgb_image"
        ).Eval(station_context)
        depth_image_0 = self.station.GetOutputPort(
            f"camera0.depth_image"
        ).Eval(station_context)

        color_image_1 = self.station.GetOutputPort(
            f"camera1.rgb_image"
        ).Eval(station_context)
        depth_image_1 = self.station.GetOutputPort(
            f"camera1.depth_image"
        ).Eval(station_context)

        plt.subplot(2, 2, index)
        plt.imshow(color_image_0.data)
        index += 1
        plt.title("Color image")
        plt.subplot(2, 2, index)
        plt.imshow(np.squeeze(depth_image_0.data))
        index += 1
        plt.title("Depth image")
        plt.subplot(2, 2, index)
        plt.imshow(color_image_1.data)
        index += 1
        plt.title("Color image")
        plt.subplot(2, 2, index)
        plt.imshow(np.squeeze(depth_image_1.data))
        index += 1
        plt.title("Depth image")

        plt.show()
            
    def move_gripper(self, robot_num, pose: RigidTransform):
        '''
        Helper to move gripper to specified pose in the world frame
        '''
        if robot_num == 0:
            diff_ik_source = self.pose1_source
            X_WG = robot1_pose.inverse() @ pose
        elif robot_num == 1:
            diff_ik_source = self.pose2_source
            X_WG = robot2_pose.inverse() @ pose
                
        diff_ik_source.set_pose(X_WG)
        
        self.advance_till_stop(robot_num)
        
    def reset_robot_hand(self, robot_num):
        pose = RigidTransform()
        pose.set_translation([0.4, 0, 0.55])
        pose.set_rotation(RollPitchYaw([-np.pi/2, 0, np.pi/2]))
        if robot_num == 0:
            diff_ik_source = self.pose1_source
        elif robot_num == 1:
            diff_ik_source = self.pose2_source
        
        diff_ik_source.set_pose(pose)
        self.advance_till_stop(robot_num)

    
    def grab_next_chip(self, robot_num):
        # Move gripper to next chip, suction cup, then return to home position
        if robot_num == 0:
            coeff = -1
            chip_positions = self.yellow_chip_positions
            next_coin = self.next_yellow_coin
        elif robot_num == 1:
            coeff = 1
            chip_positions = self.red_chip_positions
            next_coin = self.next_red_coin
        
        if next_coin >= 21:
            print("No more coins available")
            return False
            
        pose = RigidTransform()
        print(chip_positions[next_coin, :])
        x_coord, y_coord, _ = chip_positions[next_coin, :]
        
        pose.set_translation([coeff * x_coord, coeff * y_coord, 0.12])
        pose.set_rotation(RollPitchYaw([-np.pi/2, 0, np.pi/2]))
        
        self.move_gripper(robot_num, pose=pose)
        
        # Grab chip
        if robot_num == 0:
            constraint_id = self.yellow_chip_constraints[next_coin]
        elif robot_num == 1:
            constraint_id = self.red_chip_constraints[next_coin]
            
        self.plant.SetConstraintActiveStatus(self.plant_context, constraint_id, True)
            
        return constraint_id
    
    def drop_piece(self, col_num, robot_num):
        board_col_num = col_num - 1
        if robot_num == 1: # Red pieces
            board_col_num = 6 - board_col_num
        
        if self.board[0][board_col_num] != ' ':
            print("Column is full, not valid!")
            return False
        
        grabbed_id = self.grab_next_chip(robot_num)
        if not grabbed_id:
            print("Out of chips!")
            return False
        
        col_poses = col_poses_yellow if robot_num == 0 else col_poses_red
        self.move_gripper(robot_num, col_poses[col_num - 1])
        self.plant.SetConstraintActiveStatus(self.plant_context, grabbed_id, False)
        
        self.reset_robot_hand(robot_num)
        
        # Update game state
        for row in reversed(self.board):
            if row[board_col_num] == ' ':
                row[board_col_num] = self.curr_player
                break
        
        if robot_num == 0:
            self.next_yellow_coin += 1
        elif robot_num == 1:
            self.next_red_coin += 1
                            
    def check_winner(self): # Returns the player who wins if at all
        # Check horizontal, vertical, and diagonal for winning condition
        for row in range(6):
            for col in range(7):
                if self.board[row][col] != ' ':
                    if self.check_direction(row, col, 1, 0) or \
                       self.check_direction(row, col, 0, 1) or \
                       self.check_direction(row, col, 1, 1) or \
                       self.check_direction(row, col, 1, -1):
                        return self.board[row][col]
        return None

    def check_direction(self, row, col, delta_row, delta_col):
        consecutive = 0
        player = self.board[row][col]
        for _ in range(4):
            if 0 <= row < 6 and 0 <= col < 7 and self.board[row][col] == player:
                consecutive += 1
                row += delta_row
                col += delta_col
            else:
                break
        return consecutive == 4
    
    def advance_time(self, increment_time=0.3):
        self.simulator_time += increment_time
        self.simulator.AdvanceTo(self.simulator_time)    
        
meshcat = StartMeshcat()
rng = np.random.default_rng(145)  # this is for python
generator = RandomGenerator(rng.integers(0, 1000))  # this is for c++
    
game = Connect4Game(meshcat)

while True:
    col_num = int(input("Please give me a column number (1-7): "))
    game.drop_piece(col_num, robot_num=game.curr_player) 
    game.plot_manipulation_station_camera_images()
    
    winner = game.check_winner()
    if winner:
        print(f"Player {winner} wins!")
        break
        
    if all(game.board[0][col] != ' ' for col in range(7)):
        game.display_board()
        print("The game is a draw.")
        
    game.curr_player = 1 - game.curr_player