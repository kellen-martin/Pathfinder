from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=True)

x.add_system("opt", OPT, r"\text{Optimizer}")
x.add_system('D4', FUNC, "m_d = m_{hull} + m_s")
x.add_system('D5', FUNC, "D_f = |D_R - D_0 e^{-\mu \\frac{m_s}{\\rho A_s}}|")
x.add_system("solver", SOLVER, r"\text{Lambert's Solver}")
x.add_system('D1', FUNC, "\Delta v_1 = ||\Delta v_{EDB}|| + ||\Delta v_{MAB}||")
x.add_system('D2', FUNC, "\Delta v_2 = ||\Delta v_{MDB}|| + ||\Delta v_{EAB}||")
x.add_system('D3', FUNC, "F_{1,2} = m_d(e^{\Delta v_{1,2}/Isp*g_0} - 1)")
x.add_system("D6", FUNC, "F_{burn} = F_1 + F_2")




x.connect("opt", "solver", "t_{start}, [t]_i^3")
x.connect("opt", "D5", "m_s, [t]_i^3")
x.connect("opt", "D4", "m_s")
x.connect("solver", "D1", "\Delta v_{EDB}, \Delta v_{MAB}")
x.connect("solver", "D2", "\Delta v_{MDB}, \Delta v_{EAB}")
x.connect("D1", "D3", "\Delta v_1")
x.connect("D2", "D3", "\Delta v_2")
x.connect("D3", "D6", "F_1, F_2")
x.connect("D3", "opt", "F_1, F_2")
x.connect("D6", "opt", "F_{burn}")
x.connect("D4", "opt", "m_d")
x.connect("D4", "D3", "m_d")
x.connect("D5", "opt", "D_f")


x.add_output("opt", "t_{start}^*, [t^*]_i^{3}, m_s^*", side=LEFT)
x.add_output("solver", "[\Delta v^*]_i^{4} ", side=LEFT)
x.add_output("D1", "\Delta v_1^*", side=LEFT)
x.add_output("D2", "\Delta v_2^*", side=LEFT)
x.add_output("D3", "F_1^*, F_2^*", side=LEFT)
x.add_output("D6", "F_{burn}^*", side=LEFT)
x.add_output("D5", "D_f^*", side=LEFT)
x.add_output("D4", "m_d^*", side=LEFT)

x.write("mdf")