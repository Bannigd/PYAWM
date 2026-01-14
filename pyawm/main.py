import sympy as sp
from sympy.solvers.ode.systems import dsolve_system
from sympy.vector import CoordSys3D, curl
from sympy.utilities.autowrap import autowrap

from itertools import product

from preview_wrappers import preview_collection

def list_subs(eqs: list, subs: dict, eval=False):
    if eval:
        return [eq.subs(subs).expand().simplify() for eq in eqs]
    return [eq.subs(subs) for eq in eqs]


# TODO: provide additional info about vars/functions
t       = sp.Symbol("t")
x       = sp.Symbol("x")
y       = sp.Symbol("y")
z       = sp.Symbol("z")
phi     = sp.Function("varphi")(y,z)
eta     = sp.Function("eta")
gamma   = sp.Function("gamma")
c       = sp.Symbol("c")
mu      = sp.Symbol("mu")
epsilon = sp.Symbol("varepsilon")
omega   = sp.Symbol("omega")

class Domain:
    def __init__(self):

        self.E = sp.Function("E")
        self.H = sp.Function("H")
 
        for comp in [x,y,z]:
            self.__setattr__(self.E.name+comp.name, sp.Function(f"{self.E}^{comp}")(x,y,z,t))
            self.__setattr__(self.H.name+comp.name, sp.Function(f"{self.H}^{comp}")(x,y,z,t))
       
    def construct_series_sum(self, U_syms, order):
        # creating Ex0, Ey0, ..., Hx0, ... class attributes for easy access
        U_comps = [U.name+i.name for U, i in product(U_syms, [x,y,z])]
        for U in U_comps:
            for s in range(order+1):
                self.__setattr__(f"{U}{s}",sp.Function(f"{U}_{s}"))
                
        for U in U_comps:
            series_attr_name = f"{U}_series" 
            self.__setattr__(series_attr_name, [])
            for s in range(order+1):
                tmp = 0
                for k in range(s+1):
                    F = (self.__getattribute__(U+str(k))) # e.g. Ex0
                    tmp += F(x,y,z)*(sp.I*omega)**(-k)
                self.__getattribute__(series_attr_name).append(tmp)
        

    def construct_maxwell_equations(self):
        """
        creates Maxwell's equations in differential form before any manipulations
        """
        R = CoordSys3D("R")

        # Ex(R.x,R.y,R.z)*R.i + ...
        for comp in [x,y,z]:
            self.__setattr__(self.E.name+comp.name, sp.Function(f"{self.E}^{comp}")(x,y,z,t))
            self.__setattr__(self.H.name+comp.name, sp.Function(f"{self.H}^{comp}")(x,y,z,t))

        vfield_E = sum(
            [F.func(R.x, R.y, R.z, t)*unit
             for F, comp, unit in zip([self.Ex, self.Ey, self.Ez],[x, y, z], [R.i, R.j, R.k])], 0*R.i)

        vfield_H = sum(
            [F.func(R.x, R.y, R.z, t)*unit
             for F, comp, unit in zip([self.Hx, self.Hy, self.Hz],[x, y, z], [R.i, R.j, R.k])], 0*R.i)

        curl_E = curl(vfield_E) + mu / c      * vfield_H.diff(t)
        curl_H = curl(vfield_H) - epsilon / c * vfield_E.diff(t)

        # cleaning equations
        eqs = list_subs(list(curl_E.doit().to_matrix(R)) +
                        list(curl_H.doit().to_matrix(R)),
                        {R.x:x,R.y:y,R.z:z})

        return eqs

    def construct_equations_wrt_series(self, order):
        # E_series, H_series = self.get_series_sum(order)

        # U_comps = [self.__getattribute__(U.name+comp.name) for U, comp in product([self.E, self.H], [x,y,z])]
        self.construct_series_sum([self.E, self.H], order)
        eqs_0 = list_subs(self._maxwell_equations,
                          {
                              self.Ex: self.Ex_series[order]*sp.exp(sp.I*omega*t-sp.I*omega/c*phi),
                              self.Ey: self.Ey_series[order]*sp.exp(sp.I*omega*t-sp.I*omega/c*phi),
                              self.Ez: self.Ez_series[order]*sp.exp(sp.I*omega*t-sp.I*omega/c*phi),
                              self.Hx: self.Hx_series[order]*sp.exp(sp.I*omega*t-sp.I*omega/c*phi),
                              self.Hy: self.Hy_series[order]*sp.exp(sp.I*omega*t-sp.I*omega/c*phi),
                              self.Hz: self.Hz_series[order]*sp.exp(sp.I*omega*t-sp.I*omega/c*phi),
                          })
        return eqs_0

    def solve_zero_order(self):
        """
        Main method for solving awm problem in zero order. Does a lot of things inside.
        """
        self._maxwell_equations = self.construct_maxwell_equations()
        # sp.pprint(self._maxwell_equations)

        # preview_collection(self._maxwell_equations)
        preview_collection(self.construct_equations_wrt_series(0))
        preview_collection(self.construct_equations_wrt_series(1))        


if __name__ == "__main__":
    d = Domain()
    d.solve_zero_order()
    pass
