import numpy as np
import openmdao.api as om

#This is simillar as IDF code structure

#Example Block
class Aerodynamic(om.ExplicitComponent):

    def setup(self):

        # Global Design Variable
        self.add_input("theta", val=np.zeros(2))
        # Local Design Variable
        self.add_input("d", val=np.ones(2))
        # Coupling parameter (output for this block)
        self.add_output("gamma", val=np.ones(2))

    def setup_partials(self):
        # Finite difference all partials. *can use exact if want
        self.declare_partials("*", "*", method="fd")
    
    #Set the equation
    def compute(self, inputs, outputs):
        #Extract from input related to the setup section 
        theta = inputs["theta"]
        d = inputs["d"]

        A = np.zeros((2, 2))
        b = np.zeros((2, 1))
        A[0, 0] = (theta[0] + d[0]) ** 2 + 3
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[1, 1] = (theta[1] + d[1]) ** 2 + 5
        b[0, 0] = theta[0] + d[0]
        b[1, 0] = theta[1] + d[1]

        gamma = np.dot(np.linalg.inv(A), b)

        #output value is defined here
        outputs["gamma"] = gamma.flatten()

#Shield Block
class RadiShield(om.ExplicitComponent):
    def setup(self):
        #set object function varaible here 
        self.add_input('m_s', val=0.0)
        self.add_input('m_d', val=0.0)
        self.add_input('t', val=np.zeros(10))
        # ask how to intorduce the grobal inputs but constants (i.e. t_trasit and t_mars)? 
        self.add_output('f', val=0.0)
        self.add_output('dmass', val=0.0)

    def setup_partials(self):
        # Finite difference all partials. *can use exact if want
        self.declare_partials("*", "*", method="fd")
    
    def compute(self, inputs, outputs):
        #set the input for design variables
        m_d = inputs['m_d']
        m_s = inputs['m_s']
        t = inputs['t']
        t_transit = abs(t[1]-t[0])+abs(t[9]-t[8])
        t_mars = abs(t[8]-t[1])
        D0=660*t_transit+634*t_mars
        myu=2.025
        p=0.97 #kg/m^3
        As=11670 #m^3
        outputs['f'] = D0*np.exp(-myu*(m_s/(p*As)))
        outputs['dmass'] = abs(0.2*m_d-m_s)


#circle set here
class ModelGroup(om.Group):

    def setup(self):
        #set the circle for  components
        #cycle = self.add_subsystem("cycle", om.Group(), promotes=["*"])
        #cycle.add_subsystem("aero", Aerodynamic(), promotes=["*"])
        #cycle.add_subsystem("struct", Structure(), promotes=["*"])

        # cycle.set_input_defaults('gamma', 1.0)
        # cycle.set_input_defaults('d', np.array([5.0, 2.0]))

        # Nonlinear Block Gauss Seidel is a gradient free solver
        #cycle.nonlinear_solver = om.NonlinearBlockGS()
        
        #add system that is not in loop
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
prob.model.set_input_defaults("t",val=np.zeros(10))
prob.model.set_input_defaults("delta_v",val=0.0)

#set the design varaibles, objectives, constrints
prob.model.add_design_var("m_s", lower=0, upper=100000)
prob.model.add_design_var("m_d", lower=0, upper=100000)
prob.model.add_design_var("t", lower=0, upper=10000)
prob.model.set_input_defaults("delta_v",val=0.0)
prob.model.add_objective("f")
prob.model.add_constraint("dmass", equals=0)

# Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
prob.model.approx_totals()

prob.setup()
prob.set_solver_print(level=0)


prob.run_driver()

print("minimum found at")
print(prob.get_val("m_s"))
print(prob.get_val("m_d"))
print(prob.get_val("t"))
print(prob.get_val("delta_v"))

print("minumum objective")
print(prob.get_val("f"))
