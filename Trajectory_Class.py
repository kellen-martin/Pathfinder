import numpy as np
import pandas as pd
import sys
sys.path.append(r"C:\\Users\\18475\\Desktop\\Projects\\GMAT\\api")
from load_gmat import gmat
import openmdao.api as om

class Trajectory(om.ExplicitComponent):
    """
    This class is used to calculate the delta v required for each burn 
    """

    def setup(self):

        # Time vector as an input
        self.add_input("ts", val = np.zeros(5)) #each burn time

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

        """
        Call GMAT to calculate the delta-v vectors
        delta-vs = [delta_v1, delta_v2, ..., delta_vn]
        """
        # Load the GMAT Script
        gmat.LoadScript('C:\\Users\\18475\\Desktop\\Projects\\GMAT\\missions\\Pathfinder_Test.script')

        gmat.GetObject('EarthWait').SetField("Value", 10)
        gmat.GetObject('EMDeltaT').SetField("Value", 200)
        gmat.GetObject('MarsWait').SetField("Value", 100)
        gmat.GetObject('EMDeltaT').SetField("Value", 200)

        # Run GMAT Script
        gmat.RunScript()

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