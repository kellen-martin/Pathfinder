from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=True)

x.add_system("opt", OPT, r"\text{Optimizer}")
x.add_system('D4', FUNC, "m_d = m_{hull} + m_s")
x.add_system("solver", SOLVER, r"\text{GMAT}")
x.add_system('D1', FUNC, "\Delta v = \sum_{i=1}^n||\Delta v_i||")
x.add_system('D2', FUNC, "F_{burn} = m_d(e^{\Delta v/Isp*g_0} - 1)")
x.add_system('D3', FUNC, "f(m_s) = D_0 e^{-\mu \\frac{m_s}{\\rho A_s}}")




x.connect("opt", "solver", "[t]_i^n")
x.connect("opt", "D3", "m_s, [t]_i^n")
x.connect("opt", "D4", "m_s")
x.connect("solver", "D1", "[\Delta v]_i^n")
x.connect("D1", "D2", "\Delta v")
x.connect("D3", "opt", "f(m_s)")
x.connect("D2", "opt", "F_{burn}")
x.connect("D4", "D2", "m_d")
x.connect("D4", "opt", "m_d")
x.connect("D4", "D3", "m_d")


x.add_output("opt", "[t]_i^{n*}, m_s^*", side=LEFT)
x.add_output("solver", "[\Delta v]_i^{n*} ", side=LEFT)
x.add_output("D1", "\Delta v^*", side=LEFT)
x.add_output("D2", "F_{burn}^*", side=LEFT)
x.add_output("D3", "f(m_s)^*", side=LEFT)
x.add_output("D4", "m_d^*", side=LEFT)

x.write("mdf")