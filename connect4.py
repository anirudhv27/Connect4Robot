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

from pydrake.systems.framework import GenerateHtml
from manipulation.exercises.robot.test_hardware_station_io import (
    TestHardwareStationIO,
)
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import MakeHardwareStation, load_scenario

from scenario import scenario_data

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
        self.build_diagram()
        self.init_frames()
        self.init_robot()
        self.init_simulator()
    
    def build_diagram(self):
        # Build diagram
        self.builder = DiagramBuilder()
        self.scenario = load_scenario(data=scenario_data)

        self.station = self.builder.AddSystem(MakeHardwareStation(self.scenario, meshcat=meshcat))
        self.plant = self.station.GetSubsystemByName("plant")

        self.controller_plant_1 = self.station.GetSubsystemByName(
            "iiwa1.controller"
        ).get_multibody_plant_for_control()

        self.controller_plant_2 = self.station.GetSubsystemByName(
            "iiwa2.controller"
        ).get_multibody_plant_for_control()

        pose1 = RigidTransform()
        pose1.set_translation([-0.25, -0.25, 0.5])
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

        self.pose2_source = self.builder.AddSystem(PoseSource(pose1))
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

        wsg1_position = self.builder.AddSystem(ConstantVectorSource([0.1]))
        self.builder.Connect(
            wsg1_position.get_output_port(),
            self.station.GetInputPort("wsg1.position"),
        )

        wsg2_position = self.builder.AddSystem(ConstantVectorSource([0.1]))
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
        plant_context = self.diagram.GetMutableSubsystemContext(
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
        iiwa1 = self.plant.GetModelInstanceByName("iiwa1")
        self.plant.SetPositions(plant_context, iiwa1, q0)
        self.plant.SetVelocities(plant_context, iiwa1, np.zeros(7))
        wsg1 = self.plant.GetModelInstanceByName("wsg1")
        self.plant.SetPositions(plant_context, wsg1, [-0.05, 0.05])
        self.plant.SetVelocities(plant_context, wsg1, [0, 0])

        iiwa2 = self.plant.GetModelInstanceByName("iiwa2")
        self.plant.SetPositions(plant_context, iiwa2, q0)
        self.plant.SetVelocities(plant_context, iiwa2, np.zeros(7))
        wsg2 = self.plant.GetModelInstanceByName("wsg2")
        self.plant.SetPositions(plant_context, wsg2, [-0.05, 0.05])
        self.plant.SetVelocities(plant_context, wsg2, [0, 0])    

        self.diagram.ForcedPublish(self.context)
    
    def init_simulator(self):
        simulator = Simulator(self.diagram, self.context)
        simulator.set_target_realtime_rate(1.0)
        simulator.AdvanceTo(0.1);
    
    def set_pose(self, robot_num, pose: RigidTransform):
        if robot_num == 0:
            diff_ik_source = self.pose1_source
        elif robot_num == 1:
            diff_ik_source = self.pose2_source
        
        diff_ik_source.set_pose(pose)

if __name__ == '__main__':
    meshcat = StartMeshcat()
    rng = np.random.default_rng(145)  # this is for python
    generator = RandomGenerator(rng.integers(0, 1000))  # this is for c++
    
    game = Connect4Game(meshcat)
    while True:
        pass
    
