import numpy as np
import openmdao.api as om

class Trajectory(om.ExplicitComponent):
    """
    This class is used to calculate the delta v required for each burn 
    """

    def setup(self):

        # Time vector as an input
        self.add_input("ts", val = np.zeros(10)) #each burn time

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
        n = len(ts)

        delta_vs = np.zeros(n,3)
        """
        Call GMAT to calculate the delta-v vectors
        delta-vs = [delta_v1, delta_v2, ..., delta_vn]
        """
        
        # Calculate the magnitudes of the delta_v vectors
        delta_vs_mag = np.zeros(n)
        for i in range(n):
            delta_vs_mag[i] = np.linalg.norm(delta_vs[i,:])

        # Calculate the sum of the delta_v magnitudes
        delta_v = sum(delta_vs_mag)
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