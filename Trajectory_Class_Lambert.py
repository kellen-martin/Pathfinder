import numpy as np
import openmdao.api as om
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

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
    c_theta = np.dot(r1_vec, r2_vec) / (r1 * r2)
    A = np.sqrt(r1 * r2 * (1 + c_theta))

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
    This class is used to calculate the delta v required for each burn 
    """

    def setup(self):

        # Time vector as an input
        # [EarthWait EMDeltaT MarsWait MEDeltaT]
        self.add_input("ts", val = np.zeros(3))
        self.add_input("t_start", val=0)

        # delta-v magnitudes  as an output
        self.add_output("delta_v", val = 0.0)
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
        start_date = Time('2025-04-20', scale='tdb').jd  # Julian date
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
        delta_v = np.linalg.norm(EDB) + np.linalg.norm(MAB) + np.linalg.norm(MDB) + np.linalg.norm(EAB)

        outputs['delta_v'] = delta_v
        outputs['EDB'] = EDB
        outputs['MAB'] = MAB
        outputs['MDB'] = MDB
        outputs['EAB'] = EAB

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
    Traj_prob.model.set_input_defaults("ts",val=np.array([208, 450, 270]))
    Traj_prob.model.set_input_defaults("t_start",val=604.0)
    Traj_prob.setup()
    Traj_prob.set_solver_print(level=0)
    Traj_prob.run_driver()

    print("minimum found at")
    print(Traj_prob.get_val('t_start'))
    print(Traj_prob.get_val("ts"))

    print("minumum objective")
    print(Traj_prob.get_val("delta_v"))

    print("Burns")
    print(Traj_prob.get_val("EDB"))
    print(Traj_prob.get_val("MAB"))
    print(Traj_prob.get_val("MDB"))
    print(Traj_prob.get_val("EAB"))