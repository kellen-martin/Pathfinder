import numpy as np
import pandas as pd
import sys
sys.path.append(r"C:\\Users\\18475\\Desktop\\Projects\\GMAT\\api")
from load_gmat import gmat
import openmdao.api as om

#This is simillar as IDF code structure

#Trajectory Block
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

#Fuel Burn Block
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


#Shield Block
class RadiShield(om.ExplicitComponent):
    def setup(self):
        #set object function varaible here 
        self.add_input('m_s', val=0.0) #Shield Mass
        self.add_input('m_d', val=0.0) #Total Dry Mass
        self.add_input('ts', val=np.zeros(5)) #Time Vectors
        # ask how to intorduce the grobal inputs but constants (i.e. t_trasit and t_mars)? 
        self.add_output('Df', val=0.0) #Radiation after Shield
        self.add_output('dmass', val=0.0) #Constraint Value for m_s

    def setup_partials(self):
        # Finite difference all partials. *can use exact if want
        self.declare_partials("*", "*", method="fd")
    
    def compute(self, inputs, outputs):
        #set the input for design variables
        m_d = inputs['m_d']
        m_s = inputs['m_s']
        t = inputs['t']
        t_transit = abs(t[1]-t[0])+abs(t[4]-t[3]) #Earth to Mars Transition time
        t_mars = abs(t[3]-t[1]) #Stay Time in Mars
        D0=660*t_transit+634*t_mars #Original Radiation Amount
        myu=2.025
        p=0.97 #kg/m^3
        As=11670 #m^3
        outputs['Df'] = D0*np.exp(-myu*(m_s/(p*As)))
        outputs['dmass'] = 0.2*m_d-m_s


#circle set here
class ModelGroup(om.Group):

    def setup(self):
        #add system that is not in loop
        self.add_subsystem(
            "Traj",
            Trajectory(),
            promotes=["*"],
        )

        self.add_subsystem(
            "Fuel",
            Fuel_Burn(),
            promotes=["*"],
        )

        self.add_subsystem(
            "Radi",
            RadiShield(),
            promotes=["*"],
        )


prob = om.Problem()
prob.model = ModelGroup()

#Set the optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["maxiter"] = 100
prob.driver.options["tol"] = 1e-6
prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
# prob.driver.options["print_opt_prob"] = True

#set the grobal input
prob.model.set_input_defaults("m_s",val=0.0)
prob.model.set_input_defaults("m_d",val=0.0)
prob.model.set_input_defaults("ts",val=np.zeros(5))
prob.model.set_input_defaults("delta_v",val=0.0)

#set the design varaibles, objectives, constrints
prob.model.add_design_var("m_s", lower=0, upper=200000)
prob.model.add_design_var("m_d", lower=0, upper=200000)
prob.model.add_design_var("ts", lower=0) #Add upper limit if necessary
prob.model.add_design_var("delta_v",lower=0) #Add upper limit if necessary
prob.model.add_objective("Df")
prob.model.add_constraint("dmass", lower=0)
#Add other constraints if necessary 

# Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
prob.model.approx_totals()

prob.setup()
prob.set_solver_print(level=0)


prob.run_driver()

print("minimum found at")
print(prob.get_val("m_s"))
print(prob.get_val("m_d"))
print(prob.get_val("ts"))
print(prob.get_val("delta_v"))

print("minumum objective")
print(prob.get_val("Df"))

