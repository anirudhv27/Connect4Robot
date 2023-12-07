import numpy as np
from IPython.display import HTML
from pydrake.all import (
    AbstractValue,
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
    ge,
    le,
)
import re

from pydrake.systems.framework import GenerateHtml
from manipulation.exercises.robot.test_hardware_station_io import (
    TestHardwareStationIO,
)
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import MakeHardwareStation, load_scenario

from scenario import scenario_data, robot1_pose, robot2_pose, NUM_CHIPS
from col_poses import col_poses

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
    
    def set_game_state(self):
        self.next_red_coin = 0
        self.next_yellow_coin = 0
        self.curr_player = 0
    
    def init_vacuum_constraints(self, parser):
        plant = parser.plant()
        self.red_chip_constraints = [[None], [None]] # wsg 1 and wsg 2
        self.yellow_chip_constraints = [[None], [None]]
        
        wsg1_model_instance = plant.GetModelInstanceByName("wsg1")
        wsg1_tip = plant.GetBodyByName("left_finger", wsg1_model_instance)
        
        wsg2_model_instance = plant.GetModelInstanceByName("wsg2")
        wsg2_tip = plant.GetBodyByName("left_finger", wsg2_model_instance)
        
        # Red Chips
        for i in range(1, NUM_CHIPS + 1):
            model_instance = plant.GetModelInstanceByName(f"red_chip_{i}")
            body = plant.GetBodyByName("red_chip", model_instance)
            
            wsg1_constraint_id = plant.AddDistanceConstraint(body, [0, 0, 0], wsg1_tip, [0, 0, 0], 0.1)
            wsg2_constraint_id = plant.AddDistanceConstraint(body, [0, 0, 0], wsg2_tip, [0, 0, 0], 0.1)
            
            
            self.red_chip_constraints[0].append(wsg1_constraint_id)
            self.red_chip_constraints[1].append(wsg2_constraint_id)
        
        # Yellow Chips
        for i in range(1, NUM_CHIPS + 1):
            model_instance = plant.GetModelInstanceByName(f"yellow_chip_{i}")
            body = plant.GetBodyByName("yellow_chip", model_instance)
            
            wsg1_constraint_id = plant.AddDistanceConstraint(body, [0, 0, 0], wsg1_tip, [0, 0, 0], 0.01)
            wsg2_constraint_id = plant.AddDistanceConstraint(body, [0, 0, 0], wsg2_tip, [0, 0, 0], 0.01)
            
            self.yellow_chip_constraints[0].append(wsg1_constraint_id)
            self.yellow_chip_constraints[1].append(wsg2_constraint_id)            
                
    def enable_vaccum_constraints(self):
        pass
    
    def build_diagram(self):
        # Build diagram
        self.builder = DiagramBuilder()
        self.scenario = load_scenario(data=scenario_data)

        self.station = self.builder.AddSystem(MakeHardwareStation(self.scenario, meshcat=meshcat, parser_prefinalize_callback=self.init_vacuum_constraints))
        self.plant = self.station.GetSubsystemByName("plant")
        
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
            next_coin = self.next_yellow_coin
        elif robot_num == 1:
            next_coin = self.next_red_coin
        
        if next_coin >= 21:
            print("No more coins available")
            return False
            
        pose = RigidTransform()
        x_coord = 0.0825 * ((next_coin) % 5) + 0.35
        y_coord = 0.0825 * ((next_coin) // 5) + 0.35
        pose.set_translation([x_coord, y_coord, 0.2])
        pose.set_rotation(RollPitchYaw([-np.pi/2, 0, np.pi/2]))
        
        self.move_gripper(robot_num, pose=pose)
        
        # TODO: grab chip
        if robot_num == 0:
            self.next_yellow_coin += 1
        elif robot_num == 1:
            self.next_red_coin += 1
            
        return True
    
    def do_turn(self, col_num, robot_num):
        grabbed = self.grab_next_chip(robot_num)
        
        if not grabbed:
            print("Out of chips!")
            return False
        
        self.move_gripper(robot_num, col_poses[col_num - 1])
        # Todo: Release chip
        
        self.reset_robot_hand(robot_num)
        self.curr_player = 1 - self.curr_player
    
    def advance_time(self, increment_time=0.1):
        self.simulator_time += increment_time
        self.simulator.AdvanceTo(self.simulator_time)

if __name__ == '__main__':
    meshcat = StartMeshcat()
    rng = np.random.default_rng(145)  # this is for python
    generator = RandomGenerator(rng.integers(0, 1000))  # this is for c++
    
    game = Connect4Game(meshcat)
    
    while True:
        col_num = int(input("Please give me a column number (1-7): "))
        game.do_turn(col_num, robot_num=game.curr_player) 
        game.advance_time()