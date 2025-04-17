import numpy as np
import pandas as pd
import sys
sys.path.append(r"C:\\Users\\18475\\Desktop\\Projects\\GMAT\\api")
from load_gmat import gmat
import openmdao.api as om

#This is similar as IDF code structure

#Dry_mass Block
class Dry_mass(om.ExplicitComponent):
    def setup(self):
        #set object function varaible here 
        self.add_input('m_s', val=0.0) #Shield Mass
        self.add_output('m_d', val=0.0) #Dry Mass

    def setup_partials(self):
        # Finite difference between all partials. *can use exactly if want
        self.declare_partials("*", "*", method="fd")
    
    def compute(self, inputs, outputs):
        #set the input
        m_s = inputs['m_s']
        m_hull=160000.0 #kg
       
        outputs['m_d'] = m_hull+m_s

#Trajectory Block
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
        g0 = .0098    # [km/s^2]
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
        self.add_input('ts', val=np.zeros(3)) #Time Vectors
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
        t = inputs['ts']
        t_transit = (t[0] + t[2])/365 #Earth to Mars Transition time
        t_mars = t[1]/365 #Stay Time in Mars
        D0=660*t_transit+634*t_mars #Original Radiation Amount
        DR=50*(t_transit+t_mars) #Limitation Radiation Amount
        myu=2.025
        p=0.97 #kg/m^3
        As=11670 #m^3
        outputs['Df'] = abs(DR-D0*np.exp(-myu*(m_s/(p*As))))


#circle set here
class ModelGroup(om.Group):

    def setup(self):
        #add system that is not in loop
        
        self.add_subsystem(
            "Drym",
            Dry_mass(),
            promotes=["*"],
        )
        
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
prob.model.set_input_defaults("m_s",val=20000.0) #due to initial value is sensitive  
prob.model.set_input_defaults("ts",val=np.array([208, 500, 270]))
prob.model.set_input_defaults("t_start",val=608.0)

# generate Initial guess 
# Initial Guess from last Run
ts_inital = np.array([248.55026636, 496.87138656, 269.94184923])
t_start_initial = 603.24347764
EarthDepartureBurn_Initial = np.array([-1.53354101,  4.81383612,  0.21110436])
MarsArrivalBurn_Initial = np.array([-3.82042382,  0.01523331,  0.06887896])
MarsDepatureBurn_Initial = np.array([ 3.76004347,  0.23218221, -1.3525757 ])
EarthArrivalBurn_Initial = np.array([ 1.45271153, -6.6920759 , -7.55663744])

prob.model.set_input_defaults("ts",val=ts_inital)
prob.model.set_input_defaults("t_start",val=t_start_initial)
prob.model.set_input_defaults("EDB", val=EarthDepartureBurn_Initial)
prob.model.set_input_defaults("MAB", val=MarsArrivalBurn_Initial)
prob.model.set_input_defaults("MDB", val=MarsDepatureBurn_Initial)
prob.model.set_input_defaults("MAB", val=EarthArrivalBurn_Initial)

#set the design varaibles, objectives, constrints
prob.model.add_design_var("m_s", lower=0, upper=200000)
#prob.model.add_design_var("m_d", lower=0, upper=200000)
prob.model.add_design_var("ts", lower=0, upper=2*365)
prob.model.add_design_var("t_start", lower=0, upper=3650)
prob.model.add_design_var("EDB")
prob.model.add_design_var("MAB")
prob.model.add_design_var("MDB")
prob.model.add_design_var("EAB")

prob.model.add_objective("Df")
prob.model.add_constraint("m_s", lower=0, upper=40000)
prob.model.add_constraint("F", lower=0, upper=98000000)
prob.model.add_constraint('MarsRMAG', upper=13000, lower=11000)
prob.model.add_constraint("MarsVX", upper=0.1, lower=-0.1)
prob.model.add_constraint("MarsVY", upper=0.1, lower=-0.1)
prob.model.add_constraint("MarsVZ", upper=0.1, lower=-0.1)
prob.model.add_constraint('EarthRMAG', upper=13000, lower=11000)
prob.model.add_constraint("EarthVX", upper=0.1, lower=-0.1)
prob.model.add_constraint("EarthVY", upper=0.1, lower=-0.1)
prob.model.add_constraint("EarthVZ", upper=0.1, lower=-0.1)
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
print(prob.get_val("F"))

