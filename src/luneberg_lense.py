import sympy as sp
from sympy.abc import c, epsilon, mu, omega, x, y, z
from sympy.solvers.ode.systems import dsolve_system
from sympy.vector import CoordSys3D, Del
from sympy.utilities.autowrap import autowrap

import numpy as np
from scipy.optimize import bisect
import matplotlib.pyplot as plt

from preview_wrappers import preview_collection, save_latex_as_image
from solveHomoSLAE import solve_all
from pyawm import *

if __name__ == "__main__":
    # setup sympy symbols
    R = CoordSys3D("")
    delop = Del()  # nabla
    phi = sp.Function("phi")
    eta = sp.Function("eta")
    gamma = sp.Function("gamma")
    max_order = 1  # order of expansion
    E_symbols, H_symbols, E_vec_comps, H_vec_comps, E_asympt, H_asympt = (
        gen_vector_field_symbols(R, order=max_order)
    )

    # contsruct Maxwell's equations
    eqs_0, alg_eqs_0, diff_eqs_0 = gen_maxwell_eqs(
        R,
        delop,
        phi,
        E_symbols,
        H_symbols,
        E_vec_comps,
        H_vec_comps,
        E_asympt,
        H_asympt,
        order=0,
    )
    eqs_0 = list_subs(
        eqs_0, gen_vars_subs([E_symbols[0], H_symbols[0]], [x, y, z], [x])
    )
    alg_eqs_0 = list_subs(
        alg_eqs_0, gen_vars_subs([E_symbols[0], H_symbols[0]], [x, y, z], [x])
    )
    diff_eqs_0 = list_subs(
        diff_eqs_0, gen_vars_subs([E_symbols[0], H_symbols[0]], [x, y, z], [x])
    )

    # solve ODE system
    diff_sols_0 = dsolve_system(
        diff_eqs_0,
        funcs=[
            E_symbols[0][1](x),
            E_symbols[0][2](x),
            H_symbols[0][1](x),
            H_symbols[0][2](x),
        ],
        t=x,
    )[0]

    assert check_sols(diff_eqs_0, diff_sols_0) == [True, True, True, True]
    print("0: Found and checked solution to zero-order method")

    sols_0 = [
        alg_eqs_0[0].subs({eq.lhs: eq.rhs for eq in diff_sols_0}),
        diff_sols_0[0],
        diff_sols_0[1],
        alg_eqs_0[1].subs({eq.lhs: eq.rhs for eq in diff_sols_0}),
        diff_sols_0[2],
        diff_sols_0[3],
    ]

    sols_0 = list_subs(
        sols_0,
        {
            omega / c * sp.sqrt(-epsilon * mu + sp.diff(phi(z), z) ** 2): gamma(z),
            -epsilon * mu + sp.diff(phi(z), z) ** 2: eta(z),
        },
        eval=False,
    )

    # remove denominator to shorten expressions
    sols_0 = list_subs(
        sols_0,
        {
            sp.Symbol("C1"): sp.Symbol("C1") * sp.sqrt(eta(z)),
            sp.Symbol("C2"): sp.Symbol("C2") * sp.sqrt(eta(z)),
            sp.Symbol("C3"): sp.Symbol("C3") * epsilon,
            sp.Symbol("C4"): sp.Symbol("C4") * epsilon,
        },
        eval=False,
    )

    save_latex_as_image(sols_0, "general_solution_zero_order")

    # Construct solutions for different layers
    sols_0_layers = {
        layer: layered_sols(sols_0, layer, order=0) for layer in ["c", "l", "f", "s"]
    }

    # Now solving for 2D waveguide with smoothly irregular transition, x=h(z)

    # boundry conditions

    # for 4 layer slab waveguide:
    #     cover
    #     ---- x=h_l=h_f+dh
    #     lens
    #     ---- x=h_f
    #     film
    #     ---- x=0
    #     substrate

    
    h_l = sp.Function("h_1")
    h_f = sp.Function("h_2")
    border_func = lambda f: R.x - f(R.z)

    E_boundry_cl_0 = gen_boundry_conds(
        R, delop, E_vec_comps[0], border_func(h_l), h_l(z), ["c", "l"], order=0
    )

    H_boundry_cl_0 = gen_boundry_conds(
        R, delop, H_vec_comps[0], border_func(h_l), h_l(z), ["c", "l"], order=0
    )

    E_boundry_lf_0 = gen_boundry_conds(
        R, delop, E_vec_comps[0], border_func(h_f), h_f(z), ["l", "f"], order=0
    )

    H_boundry_lf_0 = gen_boundry_conds(
        R, delop, H_vec_comps[0], border_func(h_f), h_f(z), ["l", "f"], order=0
    )

    E_boundry_fs_0 = gen_boundry_conds(
        R, delop, E_vec_comps[0], border_func(h_f), 0, ["f", "s"], order=0
    )

    H_boundry_fs_0 = gen_boundry_conds(
        R, delop, H_vec_comps[0], border_func(h_f), 0, ["f", "s"], order=0
    )

    boundry_eqs_0 = list_subs(
        E_boundry_cl_0 + H_boundry_cl_0 + E_boundry_lf_0 + H_boundry_lf_0 + E_boundry_fs_0 + H_boundry_fs_0,
        {
            eq.lhs: eq.rhs
            for eq in list_subs(sols_0_layers["c"] + sols_0_layers["l"], {x: h_l(z)}, eval=False)
            +list_subs(sols_0_layers["l"] + sols_0_layers["f"], {x: h_f(z)}, eval=False)
            +list_subs(sols_0_layers["f"] + sols_0_layers["s"], {x: 0}, eval=False)

        },
        eval=False,
    )

    # order is intentional to produce block-diagonal matrix
    sym_coeffs_0 = [
        # TE
        sp.Symbol("A_0^c"),
        sp.Symbol("A_0^l"),
        sp.Symbol("B_0^l"),
        sp.Symbol("A_0^f"),
        sp.Symbol("B_0^f"),
        sp.Symbol("A_0^s"),

        # TM
        sp.Symbol("B_0^c"),
        sp.Symbol("C_0^l"),
        sp.Symbol("D_0^l"),
        sp.Symbol("C_0^f"),
        sp.Symbol("D_0^f"),
        sp.Symbol("B_0^s"),
    ]

    # reorder equations to block-diagonal matrix of coefficients
    # new_ord = [1, 2, 5, 6, 0, 3, 4, 7]
    new_order = [1, 2, 5, 6, 10, 9, 0, 3, 4, 7, 8, 11]
    boundry_eqs_0 = [boundry_eqs_0[i] for i in new_order]
    M_0, _ = sp.linear_eq_to_matrix(boundry_eqs_0, sym_coeffs_0)



    M_0_TE = M_0[:6, :6]
    sym_coeffs_0_TE = sym_coeffs_0[:6]

    # eqs_0_TE = [
    #     sp.Eq(eq.collect(sym_coeffs_0_TE, sp.combsimp), 0)
    #     for eq in M_0_TE * sp.matrices.Matrix(sym_coeffs_0_TE)
    # ]

    # sol_coeffs_0_TE = solve_all(eqs_0_TE, sym_coeffs_0_TE)

    # sol_coeffs_0_TE = [
    #     {
    #         coeff: expr.expand().collect(sp.exp(sp.Wild("w")), sp.simplify)
    #         for coeff, expr in sol.items()
    #     }
    #     for sol in sol_coeffs_0_TE
    # ]
    # print(f"0: found coefficients for TE-mode:{sym_coeffs_0_TE}")
    # save_latex_as_image(sol_coeffs_0_TE, "coeffs_0_TE_all")
    
    # boundry_check_all_solutions(eqs_0_TE, sym_coeffs_0_TE, sol_coeffs_0_TE)
    # print("0: check all variants of the solutions to boundry equations TE mode")

    M_0_TM = M_0[6:, 6:]
    sym_coeffs_0_TM = sym_coeffs_0[6:]
    # eqs_0_TM = [
    #     sp.Eq(eq.collect(sym_coeffs_0_TM, sp.combsimp), 0)
    #     for eq in M_0_TM * sp.matrices.Matrix(sym_coeffs_0_TM)
    # ]

    # sol_coeffs_0_TM = solve_all(eqs_0_TM, sym_coeffs_0_TM)

    # sol_coeffs_0_TM = [
    #     {
    #         coeff: expr.expand().collect(sp.exp(sp.Wild("w")), sp.combsimp)
    #         for coeff, expr in sol.items()
    #     }
    #     for sol in sol_coeffs_0_TM
    # ]

    # print(f"0: found coefficients for TM-mode:{sym_coeffs_0_TM}")

    # NOTE: unable to finish with default algortims. Manually checked for 3rd
    # and 4th sets.
    # bouindry_check_all_solutions(eqs_0_TM, sym_coeffs_0_TM, sol_coeffs_0_TM)
    # print("0: check all variants of the solutions to boundry equations TM mode")

    # save_latex_as_image(sol_coeffs_0_TM, "coeffs_0_TM_all")

    # Численный расчет
    layers = ['c', 'f', 'l', 's']
    symbolic_subs = {}
    for layer in layers:
        epsilon_layer = add_layer_index(epsilon, layer)
        mu_layer = add_layer_index(mu, layer)        
        symbolic_subs[add_layer_index(eta, layer)(z)] = -epsilon_layer * mu_layer + sp.diff(phi(z), z) ** 2
        symbolic_subs[add_layer_index(gamma, layer)(z)] = omega / c * sp.sqrt(-epsilon_layer * mu_layer + sp.diff(phi(z), z) ** 2)

    num_lambda = sp.Float(0.55)
    beta = sp.Symbol('beta', complex=True)
    dh = sp.Symbol(r'dh', real=True)
    numeric_parameters = {
        sp.Symbol('epsilon_c') : sp.Float(1.0)**2,
        sp.Symbol('epsilon_l') : sp.Float(3.61),
        sp.Symbol('epsilon_f') : sp.Float(1.565)**2,
        sp.Symbol('epsilon_s') : sp.Float(1.47)**2,
        h_l(z)                 : 2*num_lambda+dh,    # форма линзы h_l=h_f+dh
        h_f(z)                 : 2*num_lambda,
        sp.Symbol('mu_c')      : sp.Float(1.0),
        sp.Symbol('mu_l')      : sp.Float(1.0),
        sp.Symbol('mu_f')      : sp.Float(1.0),
        sp.Symbol('mu_s')      : sp.Float(1.0),
        omega/c                : 2*sp.pi/num_lambda,
        sp.diff(phi(z),z)**2   : beta**2,
    }
       
    sp.utilities.codegen.COMPLEX_ALLOWED = True
    det_M_0_TE_num = autowrap(M_0_TE.subs(symbolic_subs).subs(numeric_parameters).doit().to_DM().det(), backend='f2py')
    def get_determinant_TE(beta, dh):
        det = det_M_0_TE_num(beta, dh)
        return det.imag+det.real

    # M_0_TE_num = sp.lambdify([beta, dh], M_0_TE.subs(symbolic_subs).subs(numeric_parameters).doit(), 
    #                                modules=[{'sqrt':np.emath.sqrt},'numpy'])
    # def get_determinant_TE(beta, dh):
    #     M = M_0_TE_num(beta, dh)
    #     det = np.linalg.det(M)
    #     return det.imag+det.real

    lens_index_path = "IndexProf_2000_points.txt"
    if True: # do we need to recompute lens profile?
        print(f"computing lens profile from index profile in {lens_index_path}")
        with open(lens_index_path, 'r') as f:
            lines = f.readlines()[2:]
            data = []
            for line in lines:
                data.append(list(map(float,line.split())))

            data = data[::-1] # начинаем с r=1 -- край линзы



        # beta_0 это к.ф.з. для трехслойного волновода, на границе линзы, когда dh=0 
        beta_0 = bisect(lambda b: get_determinant_TE(b, 0), 1.54, 1.6, xtol=1e-16)

        dh_num = [0.]
        beta_num = [beta_0]
        for i in range(1, len(data)):
            beta_num.append(beta_0*data[i][1]) # beta=beta_0*n_eff
            dh_num.append(bisect(lambda dh: get_determinant_TE(beta_num[i], dh), dh_num[i-1], dh_num[i-1]+.2, xtol=1e-16))
    
    
        plt.figure(figsize=(20,6))
        plt.plot(np.linspace(-1,1, len(dh_num)*2), list(map(lambda dh: dh+2*num_lambda, dh_num+dh_num[::-1])))
        plt.savefig("../images/lens_profile.png")

        # saving luneberg lens profile 
        with open('luneberg_lens_profile.txt', 'w') as f:
            f.write('r, n_eff(r,f), dh, beta\n')
            for i in range(len(data)):
                f.write(f'{data[i][0]},{data[i][1]},{dh_num[i]},{beta_num[i]}\n')
