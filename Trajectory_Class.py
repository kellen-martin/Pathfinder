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
        self.add_input("ts", val = np.zeros(3))
        self.add_input("t_start", val=0)

        # delta-v magnitudes  as an output
        self.add_output("delta_v", val = 0.0)

    def setup_partials(self):
        self.declare_partials('*', '*', method = 'fd')

    def compute(self, inputs, outputs):
        """
        Uses GMAT to find the required delta-v vectors, find their magnitudes and
        the total required delta_v
        """
        ts = inputs['ts']
        t_start = inputs['t_start']
        """
        Call GMAT to calculate the delta-v vectors
        delta-vs = [delta_v1, delta_v2, ..., delta_vn]
        """
    
        # Load the GMAT Script
        gmat.LoadScript('C:\\Users\\18475\\Desktop\\Projects\\GMAT\\missions\\Pathfinder_Test3.script')
        # Set Times in GMAT
        start_date = 30770 # UTCMod Julian Date April 4, 2025
        EarthWait = t_start + start_date
        EMDeltaT = ts[0]
        MarsWait = ts[1]
        MEDeltaT = ts[2]
        gmat.GetObject('Pathfinder').SetField("Epoch", str(float(EarthWait)))
        gmat.GetObject('EMDeltaT').SetField("Value", EMDeltaT)
        gmat.GetObject('MarsWait').SetField("Value", MarsWait)
        gmat.GetObject('MEDeltaT').SetField("Value", MEDeltaT)

        # Run GMAT Script
        print("GMAT Start")
        t_start = t.perf_counter()
        gmat.RunScript()
        t_end = t.perf_counter()
        time = (t_end - t_start)/60
        print("GMAT run completed, Solve Time: ", time)

        # Extract Delta-V Vectors
        report_path = "C:\\Users\\18475\\Desktop\\Projects\\GMAT\\output\\DeltaVs.txt"
        df = pd.read_csv(report_path, sep="\t", header=None) 
        last_row = df.iloc[-1]
        delta_vs_flat = np.array([float(x) for dv in last_row for x in dv.split()])
        burns = delta_vs_flat.reshape(6, 3)

        # Calculate the magnitudes of the delta_v vectors
        burn_mags = np.linalg.norm(burns, axis=1)

        # Calculate the sum of the delta_v magnitudes
        delta_v = sum(burn_mags)
        outputs['delta_v'] = delta_v

class Fuel_Burn(om.ExplicitComponent):
    """
    This class calculates the fuel required for the mission in kg
    """

    def setup(self):
        # Inputs are the total delta_v and the dry mass of the s/c
        self.add_input('delta_v', val = 0.0)
        self.add_input('m_d', val = 0.0)

        # Output is the total fuel burn in kg 
        self.add_output('F', val = 0.0)

    def setup_partials(self):
        self.declare_partials('*','*', method= 'fd')  

    def compute(self, inputs, outputs):
        """
        Computes the required fuel burn by the equation
        F = m_d(exp(delta_v/(Isp*g0) - 1)
        """
        delta_v = inputs['delta_v']
        m_d = inputs['m_d']

        # Constants
        g0 = 9.8    # [m/s^2] change to km?
        Isp = 380   # [s]

        # Compute fuel burn
        F = m_d*(np.exp(delta_v/(Isp*g0)) - 1)
        outputs['F'] = F

if __name__ == "__main__":
    """
    Test of Trajectory Component
    """
    
    Traj_prob = om.Problem()
    Traj_prob.model.add_subsystem("Trajectory", Trajectory(), promotes=['*'])
 
    Traj_prob.driver = om.ScipyOptimizeDriver()
    Traj_prob.driver.options["optimizer"] = "SLSQP"
    Traj_prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]

    Traj_prob.model.add_design_var("ts", lower=10, upper=2*365)
    Traj_prob.model.add_design_var("t_start", lower=0, upper=3650)
    Traj_prob.model.add_objective('delta_v')
    Traj_prob.model.set_input_defaults("ts",val=np.array([208, 500, 270]))
    Traj_prob.model.set_input_defaults("t_start",val=608.0)
    Traj_prob.setup()
    Traj_prob.set_solver_print(level=0)
    Traj_prob.run_driver()

    # Traj_prob.set_val('ts', np.array([100, 200, 150, 200]))
    
    """
    Traj_prob.run_model()

    delta_v = Traj_prob.get_val('delta_v')
    print(delta_v)
    """


 


