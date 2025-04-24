import numpy as np
import pandas as pd
import openmdao.api as om
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

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
# Define stumpff function for lambert solver
def stumpff_c2(psi):
    if psi > 1e-6:
        return (1 - np.cos(np.sqrt(psi))) / psi
    elif psi < -1e-6:
        return (1 - np.cosh(np.sqrt(-psi))) / psi
    else:
        return 0.5

def stumpff_c3(psi):
    if psi > 1e-6:
        return (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi ** 1.5)
    elif psi < -1e-6:
        return (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / ((-psi) ** 1.5)
    else:
        return 1/6

# Universal Lambert's Problem solver using bisection
def UV_Lambert_Bisect(r1_vec, r2_vec, mu, delta_t, tol=1e-4, max_iter=1000):
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    theta = np.arccos(np.dot(r1_vec, r2_vec) / (r1 * r2))

    # Auto-detect long-way using cross product
    cross = np.cross(r1_vec, r2_vec)
    if cross[2] < 0:  # Negative z-component means long-way (retrograde-like)
        theta = 2 * np.pi - theta

    c_theta = np.cos(theta)
    A = np.sign(np.sin(theta)) * np.sqrt(abs(r1 * r2 * (1 + c_theta)))

    psi_low, psi_up = -4*np.pi**2, 4*np.pi**2
    psi = 0
    n = 0

    while n < max_iter:
        c2 = stumpff_c2(psi)
        c3 = stumpff_c3(psi)
        try:
            y = r1 + r2 + (A * (psi * c3 - 1) / np.sqrt(c2))
            chi = np.sqrt(y / c2)
            dt_guess = (chi ** 3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        except:
            psi += 1e-4
            n += 1
            continue

        if abs(dt_guess - delta_t) < tol:
            break

        if dt_guess < delta_t:
            psi_low = psi
        else:
            psi_up = psi

        psi = 0.5 * (psi_low + psi_up)
        n += 1

    f = 1 - y / r1
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g
    return v1, v2, n

class Trajectory(om.ExplicitComponent):
    """
    This class is used to calculate the total delta v required for the mission
    """

    def setup(self):

        # Time vector as an input
        # [EarthWait EMDeltaT MarsWait MEDeltaT]
        self.add_input("ts", val = np.zeros(3))
        self.add_input("t_start", val=0)

        # delta-v magnitudes  as an output
        self.add_output("delta_v1", val = 0.0)
        self.add_output("delta_v2", val = 0.0)
        self.add_output("EDB", val=np.zeros(3))
        self.add_output("MAB", val=np.zeros(3))
        self.add_output("MDB", val=np.zeros(3))
        self.add_output("EAB", val=np.zeros(3))

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
        Calls Lambert's Problem Solver to find delta-v required for the given times
        """
        mu_sun = 1.32715e11  # [km^3/s^2]
    
        # Set Times in GMAT
        start_date = Time('2025-04-20', scale='tdb').jd  # Today's Julian date
        EarthDepart = t_start + start_date
        EMDeltaT = ts[0]
        MarsWait = ts[1]
        MEDeltaT = ts[2]

        # Convert Transfer Times to [s]
        EMDeltaT_s = EMDeltaT * 86400
        MEDeltaT_s = MEDeltaT * 86400

        # Find the dates of the burns
        t_EDB = EarthDepart
        t_MAB = t_EDB + EMDeltaT
        t_MDB = t_MAB + MarsWait
        t_EAB = t_MDB + MEDeltaT

        # Find Planet Positions and Velocities for transfer
        def get_state(body, jd):
            with solar_system_ephemeris.set('jpl'):
                pos, vel = get_body_barycentric_posvel(body, Time(jd, format='jd'))
            return pos.xyz.to_value('km').flatten(), vel.xyz.to_value('km/s').flatten()
        
        R_E1, V_E1 = get_state('earth', t_EDB)
        R_M1, V_M1 = get_state('mars', t_MAB)
        R_M2, V_M2 = get_state('mars', t_MDB)
        R_E2, V_E2 = get_state('earth', t_EAB)

        # Solve Lambert's Problem
        v1_E, v2_M, _ = UV_Lambert_Bisect(R_E1, R_M1, mu_sun, EMDeltaT_s)
        v1_M, v2_E, _ = UV_Lambert_Bisect(R_M2, R_E2, mu_sun, MEDeltaT_s)
    
        # Compute Burns
        EDB = v1_E - V_E1
        MAB = V_M1 - v2_M
        MDB = v1_M - V_M2
        EAB = V_E2 - v2_E

        # Calculate the sum of the delta_v magnitudes
        delta_v1 = np.linalg.norm(EDB) + np.linalg.norm(MAB)
        delta_v2 = np.linalg.norm(MDB) + np.linalg.norm(EAB)

        outputs['delta_v1'] = delta_v1
        outputs['delta_v2'] = delta_v2
        outputs['EDB'] = EDB
        outputs['MAB'] = MAB
        outputs['MDB'] = MDB
        outputs['EAB'] = EAB

#Fuel Burn Block
class Fuel_Burn(om.ExplicitComponent):
    """
    This class calculates the fuel required for the mission in kg
    """

    def setup(self):
        # Inputs are the total delta_v and the dry mass of the s/c
        self.add_input('delta_v1', val = 0.0)
        self.add_input('delta_v2', val=0.0)
        self.add_input('m_d', val = 0.0)

        # Output is the total fuel burn in kg 
        self.add_output('F1', val = 0.0)
        self.add_output('F2', val = 0.0)
        self.add_output("F_total", val = 0.0)

    def setup_partials(self):
        self.declare_partials('*','*', method= 'fd')  

    def compute(self, inputs, outputs):
        """
        Computes the required fuel burn by the equation
        F = m_d(exp(delta_v/(Isp*g0) - 1)
        """
        delta_v1 = inputs['delta_v1']
        delta_v2 = inputs['delta_v2']
        m_d = inputs['m_d']

        # Constants
        g0 = .0098    # [km/s^2]
        Isp = 380   # [s]

        # Compute fuel burn
        F1 = m_d*(np.exp(delta_v1/(Isp*g0)) - 1)
        F2 = m_d*(np.exp(delta_v2/(Isp*g0)) - 1)
        F_total = F1 + F2
        outputs['F1'] = F1
        outputs['F2'] = F2
        outputs['F_total'] = F_total



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
        t_total = t_transit + t_mars
        D0=660*t_transit+634*t_mars + 20*t_total*(11-t_total) #Original Radiation Amount (mSv) with SEP High Radiation Includes
        DR= 150 #Limitation Radiation Amount (mSv) (1/4 of NASA astronaut career radiation Limit)
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



prob.setup()

prob.model.set_val("m_s",val=40000.0)
prob.model.set_val("ts",val=np.array([240, 450, 270]))
prob.model.set_val("t_start",val=604.0)

prob.run_model()

print("Initial Guess")
print("Shelid Mass :", prob.get_val("m_s"))
print("Dry Mass :", prob.get_val("m_d"))
print("Df: ", prob.get_val("Df"))
print("T Start: ", prob.get_val("t_start"))
print("Times :", prob.get_val("ts"))
print("Earth-Mars Leg delta-v :", prob.get_val("delta_v1"))
print("Earth-Mars Fuel Burn", prob.get_val("F1"))
print("Mars-Earth Leg delta-v :", prob.get_val("delta_v2"))
print("Mars-Earth Fuel Burn", prob.get_val("F2"))

print("EDB: ", prob.get_val('EDB'))
print("MAB: ", prob.get_val('MAB'))
print("MDB: ", prob.get_val('MDB'))
print("EAB: ", prob.get_val('EAB'))

print("Objective")
print(prob.get_val("F_total"))