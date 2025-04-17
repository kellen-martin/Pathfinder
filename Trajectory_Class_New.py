import numpy as np
import pandas as pd
import sys
sys.path.append(r"C:\\Users\\18475\\Desktop\\Projects\\GMAT\\api")
from load_gmat import gmat
import openmdao.api as om
import time as t


class Trajectory(om.ExplicitComponent):
    """
    This class is used to calculate the delta v required for each burn 
    """

    def setup(self):

        # Time vector as an input
        # [EarthWait EMDeltaT MarsWait MEDeltaT]
        self.add_input("EDB", val = np.zeros(3))
        self.add_input("MAB", val=np.zeros(3))
        self.add_input("MDB", val = np.zeros(3))
        self.add_input("EAB", val=np.zeros(3))
        self.add_input("ts", val = np.zeros(3))
        self.add_input("t_start", val=0)

        # delta-v magnitudes  as an output
        self.add_output("delta_v", val = 0.0)
        self.add_output("MarsRMAG")
        self.add_output("MarsVX")
        self.add_output("MarsVY")
        self.add_output("MarsVZ")
        self.add_output("EarthRMAG")
        self.add_output("EarthVX")
        self.add_output("EarthVY")
        self.add_output("EarthVZ")

    def setup_partials(self):
        self.declare_partials('*', '*', method = 'fd')

    def compute(self, inputs, outputs):
        """
        Uses GMAT to find the required delta-v vectors, find their magnitudes and
        the total required delta_v
        """
        ts = inputs['ts']
        t_start = inputs['t_start']
        EarthDepartureBurn = inputs['EDB']
        MarsArrivalBurn = inputs['MAB']
        MarsDepartureBurn = inputs['MDB']
        EarthArrivalBurn = inputs['EAB']
        """
        Call GMAT to calculate the delta-v vectors
        delta-vs = [delta_v1, delta_v2, ..., delta_vn]
        """
    
        # Load the GMAT Script
        gmat.LoadScript('C:\\Users\\18475\\Desktop\\Projects\\GMAT\\missions\\Pathfinder_EM.script')
        # Set Times in GMAT
        start_date = 30770 # UTCMod Julian Date April 4, 2025
        EarthWait = t_start + start_date
        EMDeltaT = ts[0]
        MarsWait = ts[1]
        MEDeltaT = ts[2]

        # Set Times (Earth-Mars)
        gmat.GetObject('Pathfinder').SetField("Epoch", str(float(EarthWait)))
        gmat.GetObject('EMDeltaT').SetField("Value", EMDeltaT)

        # Set Earth Departure Burn
        gmat.GetObject('EarthDepartureBurn').SetField("Element1", EarthDepartureBurn[0])
        gmat.GetObject('EarthDepartureBurn').SetField("Element2", EarthDepartureBurn[1])
        gmat.GetObject('EarthDepartureBurn').SetField("Element3", EarthDepartureBurn[2])

        # Set Mars Arrival Burn
        gmat.GetObject('MarsArrivalBurn').SetField("Element1", MarsArrivalBurn[0])
        gmat.GetObject('MarsArrivalBurn').SetField("Element2", MarsArrivalBurn[1])
        gmat.GetObject('MarsArrivalBurn').SetField("Element3", MarsArrivalBurn[2])

        # Run GMAT
        gmat.RunScript()
   
        # Get Vars (Earth to Mars)
        MarsRMAG = gmat.GetRuntimeObject("MarsRMAG").GetField("Value")
        MarsVX = gmat.GetRuntimeObject("MarsVX").GetField("Value")
        MarsVY = gmat.GetRuntimeObject("MarsVY").GetField("Value")
        MarsVZ = gmat.GetRuntimeObject("MarsVZ").GetField("Value")
        
        ### Mars-Earth ###
        gmat.LoadScript('C:\\Users\\18475\\Desktop\\Projects\\GMAT\\missions\\Pathfinder_ME.script')

        Mars_Departure = EarthWait + EMDeltaT + MarsWait

        # Set Times
        gmat.GetObject('Pathfinder').SetField("Epoch", str(float(Mars_Departure)))
        gmat.GetObject('MEDeltaT').SetField("Value", MEDeltaT)

        # Set Mars Departure Burn
        gmat.GetObject('MarsDepartureBurn').SetField("Element1", MarsDepartureBurn[0])
        gmat.GetObject('MarsDepartureBurn').SetField("Element2", MarsDepartureBurn[1])
        gmat.GetObject('MarsDepartureBurn').SetField("Element3", MarsDepartureBurn[2])

        # Set Mars Arrival Burn
        gmat.GetObject('EarthArrivalBurn').SetField("Element1", EarthArrivalBurn[0])
        gmat.GetObject('EarthArrivalBurn').SetField("Element2", EarthArrivalBurn[1])
        gmat.GetObject('EarthArrivalBurn').SetField("Element3", EarthArrivalBurn[2])

        # Run GMAT
        gmat.RunScript()

        # Get Vars (Earth to Mars)
        EarthRMAG = gmat.GetRuntimeObject("EarthRMAG").GetField("Value")
        EarthVX = gmat.GetRuntimeObject("EarthVX").GetField("Value")
        EarthVY = gmat.GetRuntimeObject("EarthVY").GetField("Value")
        EarthVZ = gmat.GetRuntimeObject("EarthVZ").GetField("Value")

        # Calculate delta-v
        delta_v = np.linalg.norm(EarthDepartureBurn) + np.linalg.norm(MarsArrivalBurn) + np.linalg.norm(MarsDepartureBurn) + np.linalg.norm(EarthArrivalBurn)

        # Outputs
        outputs['delta_v'] = delta_v
        outputs['MarsRMAG'] = MarsRMAG
        outputs['MarsVX'] = MarsVX
        outputs['MarsVY'] = MarsVY
        outputs['MarsVZ'] = MarsVZ
        outputs['EarthRMAG'] = EarthRMAG
        outputs['EarthVX'] = EarthVX
        outputs['EarthVY'] = EarthVY
        outputs['EarthVZ'] = EarthVZ




if __name__ == "__main__":
    """
    Test of Trajectory Component
    """
    """
    # Genertate Initial Guess
    gmat.LoadScript('C:\\Users\\18475\\Desktop\\Projects\\GMAT\\missions\\Pathfinder_Test3.script')
    ts_inital = np.array([208.0, 500.0, 270.0])
    t_start_initial = 608.0

    start_date = 30770 # UTCMod Julian Date April 4, 2025
    EarthWait = t_start_initial + start_date
    EMDeltaT = ts_inital[0]
    MarsWait = ts_inital[1]        
    MEDeltaT = ts_inital[2]
    gmat.GetObject('Pathfinder').SetField("Epoch", str(float(EarthWait)))
    gmat.GetObject('EMDeltaT').SetField("Value", EMDeltaT)
    gmat.GetObject('MarsWait').SetField("Value", MarsWait)
    gmat.GetObject('MEDeltaT').SetField("Value", MEDeltaT)

    # Run GMAT Script
    print("GMAT Start")
    gmat.RunScript()
    print("GMAT run completed, Solve Time: ")

    # Extract Delta-V Vectors
    report_path = "C:\\Users\\18475\\Desktop\\Projects\\GMAT\\output\\DeltaVs.txt"
    df = pd.read_csv(report_path, sep="\t", header=None) 
    last_row = df.iloc[-1]
    delta_vs_flat = np.array([float(x) for dv in last_row for x in dv.split()])
    burns = delta_vs_flat.reshape(6, 3)

    # Initial Burn Guesses
    EarthDepartureBurn_Initial = burns[1,:]
    MarsArrivalBurn_Initial = burns[2,:]
    MarsDepatureBurn_Initial = burns[4,:]
    EarthArrivalBurn_Initial = burns[5,:]
    """

    # Initial Guess from last Run
    ts_inital = np.array([248.55026636, 496.87138656, 269.94184923])
    t_start_initial = 603.24347764
    EarthDepartureBurn_Initial = np.array([-1.53354101,  4.81383612,  0.21110436])
    MarsArrivalBurn_Initial = np.array([-3.82042382,  0.01523331,  0.06887896])
    MarsDepatureBurn_Initial = np.array([ 3.76004347,  0.23218221, -1.3525757 ])
    EarthArrivalBurn_Initial = np.array([ 1.45271153, -6.6920759 , -7.55663744])
    # Set Up Problem
    Traj_prob = om.Problem()
    Traj_prob.model.add_subsystem("Trajectory", Trajectory(), promotes=['*'])
 
    Traj_prob.driver = om.ScipyOptimizeDriver()
    Traj_prob.driver.options["optimizer"] = "SLSQP"
    Traj_prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]

    # Design Variables
    Traj_prob.model.add_design_var("ts", lower=0, upper=2*365)
    Traj_prob.model.add_design_var("t_start", lower=0, upper=3650)
    Traj_prob.model.add_design_var("EDB")
    Traj_prob.model.add_design_var("MAB")
    Traj_prob.model.add_design_var("MDB")
    Traj_prob.model.add_design_var("EAB")

    # Objectives and Constraints
    Traj_prob.model.add_objective('delta_v')
    Traj_prob.model.add_constraint('MarsRMAG', upper=13000, lower=11000)
    Traj_prob.model.add_constraint("MarsVX", upper=0.1, lower=-0.1)
    Traj_prob.model.add_constraint("MarsVY", upper=0.1, lower=-0.1)
    Traj_prob.model.add_constraint("MarsVZ", upper=0.1, lower=-0.1)
    Traj_prob.model.add_constraint('EarthRMAG', upper=13000, lower=11000)
    Traj_prob.model.add_constraint("EarthVX", upper=0.1, lower=-0.1)
    Traj_prob.model.add_constraint("EarthVY", upper=0.1, lower=-0.1)
    Traj_prob.model.add_constraint("EarthVZ", upper=0.1, lower=-0.1)

    # Initial Guess
    Traj_prob.model.set_input_defaults("ts",val=ts_inital)
    Traj_prob.model.set_input_defaults("t_start",val=t_start_initial)
    Traj_prob.model.set_input_defaults("EDB", val=EarthDepartureBurn_Initial)
    Traj_prob.model.set_input_defaults("MAB", val=MarsArrivalBurn_Initial)
    Traj_prob.model.set_input_defaults("MDB", val=MarsDepatureBurn_Initial)
    Traj_prob.model.set_input_defaults("MAB", val=EarthArrivalBurn_Initial)

    Traj_prob.setup()
    Traj_prob.set_solver_print(level=0)
    Traj_prob.run_driver()

    # Traj_prob.set_val('ts', np.array([100, 200, 150, 200]))
    
    """
    Traj_prob.run_model()

    delta_v = Traj_prob.get_val('delta_v')
    print(delta_v)
    """


 


