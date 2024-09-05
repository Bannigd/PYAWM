from datetime import date, datetime
from functools import reduce
from itertools import chain, product
from pathlib import Path

import sympy as sp
from sympy import latex
from sympy.abc import c, epsilon, mu, omega, t, x, y, z
from sympy.printing import preview
from sympy.solvers.ode.systems import dsolve_system
from sympy.vector import CoordSys3D, Del

from solveHomoSLAE import solve_all


def gen_vars_subs(U_symbols, args_before, args_after):
    return {
        U(*args_before): U(*args_after) for U in list(chain.from_iterable(U_symbols))
    }


def gen_subs_diff(R, U_comps):
    return {
        value.diff(key): 0
        for key, value in list(product([R.y, R.z], list(U_comps.components.values())))
    }


def list_subs(eqs: list, subs: dict, eval=True):
    if eval is True:
        return list(map(lambda eq: eq.subs(subs).expand().simplify(), eqs))
    else:
        return list(map(lambda eq: eq.subs(subs), eqs))


def check_sols(eqs, sols):
    sols_subs = {sol.lhs: sol.rhs for sol in sols}
    return list_subs(eqs, sols_subs)


def gen_vector_field_symbols(R, order=0):
    E_symbols = [
        [sp.Function(f"E^{comp}_{i}") for comp in [x, y, z]] for i in range(order + 1)
    ]
    E_vec_comps = [
        Ex(R.x, R.y, R.z) * R.i + Ey(R.x, R.y, R.z) * R.j + Ez(R.x, R.y, R.z) * R.k
        for Ex, Ey, Ez in E_symbols
    ]
    H_symbols = [
        [sp.Function(f"H^{comp}_{i}") for comp in [x, y, z]] for i in range(order + 1)
    ]
    H_vec_comps = [
        Hx(R.x, R.y, R.z) * R.i + Hy(R.x, R.y, R.z) * R.j + Hz(R.x, R.y, R.z) * R.k
        for Hx, Hy, Hz in H_symbols
    ]
    E_asympt = [
        sum(
            map(
                lambda k, vec: vec / (sp.I * omega) ** k,
                range(s + 1),
                E_vec_comps,
            ),
            0 * R.i,
        )
        for s in range(order + 1)
    ]  # 0*R.i determines return type
    H_asympt = [
        sum(
            map(
                lambda k, vec: vec / (sp.I * omega) ** k,
                range(s + 1),
                H_vec_comps,
            ),
            0 * R.i,
        )
        for s in range(order + 1)
    ]
    return E_symbols, H_symbols, E_vec_comps, H_vec_comps, E_asympt, H_asympt


def gen_maxwell_eqs(
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
    prev_order_eqs=None,
):
    curl_E = (
        (
            (
                delop.cross(
                    E_asympt[order]
                    * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                )
                + mu
                / c
                * (
                    H_asympt[order]
                    * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                ).diff(t)
            )
            .doit()
            .to_matrix(R)
            / sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
        )
        .expand()
        .subs(gen_subs_diff(R, E_vec_comps[order]))
    )

    curl_H = (
        (
            (
                delop.cross(
                    H_asympt[order]
                    * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                )
                - epsilon
                / c
                * (
                    E_asympt[order]
                    * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                ).diff(t)
            )
            .doit()
            .to_matrix(R)
            / sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
        )
        .expand()
        .subs(gen_subs_diff(R, H_vec_comps[order]))
    )

    maxwell_eqs = list_subs(
        [sp.Eq(eq, 0) for eq in list(curl_E) + list(curl_H)],
        {R.x: x, R.y: y, R.z: z},
        eval=False,
    )

    if prev_order_eqs is not None:
        maxwell_eqs = [
            sp.Eq((maxwell_eqs[i].lhs - prev_order_eqs[i].lhs).expand(), 0)
            for i in range(6)
        ]

    maxwell_alg_eqs = [
        sp.Eq(
            E_symbols[order][0](x, y, z),
            sp.solve(maxwell_eqs[3], E_symbols[order][0](x, y, z))[0],
        ),
        sp.Eq(
            H_symbols[order][0](x, y, z),
            sp.solve(maxwell_eqs[0], H_symbols[order][0](x, y, z))[0],
        ),
    ]

    maxwell_diff_eqs = list(
        filter(
            lambda x: x is not (sp.sympify(True) or 0),
            list_subs(maxwell_eqs, {eq.lhs: eq.rhs for eq in maxwell_alg_eqs}),
        )
    )
    return maxwell_eqs, maxwell_alg_eqs, maxwell_diff_eqs


def layered_sols(sols, layer_symbol, order) -> list:
    # TODO: add a param to signal what exponents (with `+` or `-`) to omit
    # WARNING: order of substituting symbols here is "hard-coded" for author's convenience
    match layer_symbol:
        case "c":
            # x -> x-h(z) eases calculation
            sols = [
                sol.replace(
                    sp.exp, lambda arg: sp.exp(arg.subs({x: x - sp.Function("h")(z)}))
                )
                for sol in sols
            ]

            sols = list_subs(
                sols,
                {
                    sp.Symbol("C1"): sp.Symbol(f"A_{order}^{layer_symbol}"),
                    sp.Symbol("C2"): 0,
                    sp.Symbol("C3"): sp.Symbol(f"B_{order}^{layer_symbol}"),
                    sp.Symbol("C4"): 0,
                },
                eval=False,
            )
        case "f":
            sols = list_subs(
                sols,
                {
                    sp.Symbol("C1"): sp.Symbol(f"A_{order}^{layer_symbol}"),
                    sp.Symbol("C2"): sp.Symbol(f"B_{order}^{layer_symbol}"),
                    sp.Symbol("C3"): sp.Symbol(f"C_{order}^{layer_symbol}"),
                    sp.Symbol("C4"): sp.Symbol(f"D_{order}^{layer_symbol}"),
                },
                eval=False,
            )
            pass
        case "s":
            sols = list_subs(
                sols,
                {
                    sp.Symbol("C1"): 0,
                    sp.Symbol("C2"): sp.Symbol(f"A_{order}^{layer_symbol}"),
                    sp.Symbol("C3"): 0,
                    sp.Symbol("C4"): sp.Symbol(f"B_{order}^{layer_symbol}"),
                },
                eval=False,
            )
    return [
        sol.xreplace(
            {
                sol.lhs: comp,
                epsilon: sp.Symbol(f"epsilon_{layer_symbol}"),
                mu: sp.Symbol(f"mu_{layer_symbol}"),
                sp.Function("eta")(z): sp.Function(f"eta_{layer_symbol}")(z),
                sp.Function("gamma")(z): sp.Function(f"gamma_{layer_symbol}")(z),
            }
        )
        for sol, comp in zip(
            sols,
            [
                sp.Function(f"E^{{x,{layer_symbol}}}_{order}")(x),
                sp.Function(f"E^{{y,{layer_symbol}}}_{order}")(x),
                sp.Function(f"E^{{z,{layer_symbol}}}_{order}")(x),
                sp.Function(f"H^{{x,{layer_symbol}}}_{order}")(x),
                sp.Function(f"H^{{y,{layer_symbol}}}_{order}")(x),
                sp.Function(f"H^{{z,{layer_symbol}}}_{order}")(x),
            ],
        )
    ]


def gen_boundry_conds(R, delop, U, border_func, border_func_value, layers, order):
    normal = delop(border_func).doit()
    char_U = str(list(U.components.values())[0])[0]
    nU_1 = normal.cross(U).subs(
        {
            U.components[R.i]: sp.Function(f"{char_U}^{{x,{layers[0]}}}_{order}")(x),
            U.components[R.j]: sp.Function(f"{char_U}^{{y,{layers[0]}}}_{order}")(x),
            U.components[R.k]: sp.Function(f"{char_U}^{{z,{layers[0]}}}_{order}")(x),
        }
    )

    nU_2 = normal.cross(U).subs(
        {
            U.components[R.i]: sp.Function(f"{char_U}^{{x,{layers[1]}}}_{order}")(x),
            U.components[R.j]: sp.Function(f"{char_U}^{{y,{layers[1]}}}_{order}")(x),
            U.components[R.k]: sp.Function(f"{char_U}^{{z,{layers[1]}}}_{order}")(x),
        }
    )

    boundry_eqs = (nU_1 - nU_2).subs({x: border_func_value, R.y: y, R.z: z})

    return [sp.Eq(boundry_eqs.components[arg], 0) for arg in [R.j, R.k]]


def save_latex_as_image(equations, filename):
    image_time_prefix = datetime.today().strftime("%y-%m-%d--%H-%M-%S--")
    today = date.today().isoformat()
    image_today_path = Path(f"./images/{today}")
    image_today_path.mkdir(parents=True, exist_ok=True)

    with open(
        f"{str(image_today_path)}/{image_time_prefix}{filename}.png",
        "wb",
    ) as out:
        if isinstance(equations, list):
            output = reduce(
                lambda x, y: x + y,
                [latex(eq, mode="equation*") for eq in equations],
                "",
            )
        else:
            output = latex(equations, mode="equation*")
        preview(
            output,
            viewer="BytesIO",
            outputbuffer=out,
            dvioptions=["-D 200"],
            euler=False,
        )


def main():
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
    print("Found and checked solution to zero-order method")

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

    # save_latex_as_image(sols_0, "general_solution_zero_order")

    # Construct solutions for different layers
    sols_0_layers = {
        layer: layered_sols(sols_0, layer, order=0) for layer in ["c", "f", "s"]
    }

    # Now solving for 2D waveguide with smoothly irregular transition, x=h(z)

    # boundry conditions
    h = sp.Function("h")
    border_func = R.x - h(R.z)

    E_boundry_cf_0 = gen_boundry_conds(
        R, delop, E_vec_comps[0], border_func, h(z), ["c", "f"], order=0
    )

    H_boundry_cf_0 = gen_boundry_conds(
        R, delop, H_vec_comps[0], border_func, h(z), ["c", "f"], order=0
    )

    E_boundry_fs_0 = gen_boundry_conds(
        R, delop, E_vec_comps[0], border_func, 0, ["f", "s"], order=0
    )

    H_boundry_fs_0 = gen_boundry_conds(
        R, delop, H_vec_comps[0], border_func, 0, ["f", "s"], order=0
    )

    boundry_eqs_0 = list_subs(
        E_boundry_cf_0 + H_boundry_cf_0 + E_boundry_fs_0 + H_boundry_fs_0,
        {
            eq.lhs: eq.rhs
            for eq in list_subs(
                sols_0_layers["c"] + sols_0_layers["f"], {x: h(z)}, eval=False
            )
            + list_subs(sols_0_layers["f"] + sols_0_layers["s"], {x: 0}, eval=False)
        },
        eval=False,
    )

    # order is intentional to produce block-diagonal matrix
    coeffs_0 = [
        # TE
        sp.Symbol("A_0^c"),
        sp.Symbol("A_0^f"),
        sp.Symbol("B_0^f"),
        sp.Symbol("A_0^s"),
        # TM
        sp.Symbol("B_0^c"),
        sp.Symbol("C_0^f"),
        sp.Symbol("D_0^f"),
        sp.Symbol("B_0^s"),
    ]

    # reorder equations to block-diagonal matrix of coefficients
    new_ord = [1, 2, 5, 6, 0, 3, 4, 7]
    boundry_eqs_0 = [boundry_eqs_0[i] for i in new_ord]
    M_0, _ = sp.linear_eq_to_matrix(boundry_eqs_0, coeffs_0)

    M_0_TE = M_0[:4, :4]
    coeffs_0_TE = coeffs_0[:4]
    M_0_TM = M_0[4:, 4:]
    coeffs_0_TM = coeffs_0[4:]

    # sp.pprint(M_0)
    eqs_0_TE = [
        sp.Eq(eq.collect(coeffs_0_TE, sp.combsimp), 0)
        for eq in M_0_TE * sp.matrices.Matrix(coeffs_0_TE)
    ]
    eqs_0_TM = [
        sp.Eq(eq.collect(coeffs_0_TM, sp.combsimp), 0)
        for eq in M_0_TM * sp.matrices.Matrix(coeffs_0_TM)
    ]
    # sp.pprint(eqs_0_TE)
    sol_coeffs_0_TE = solve_all(eqs_0_TE, coeffs_0_TE)
    sol_coeffs_0_TM = solve_all(eqs_0_TM, coeffs_0_TM)
    sol_coeffs_0_TE = [
        {
            coeff: expr.expand().collect(sp.exp(sp.Wild("w")), sp.simplify)
            for coeff, expr in sol.items()
        }
        for sol in sol_coeffs_0_TE
    ]
    sol_coeffs_0_TM = [
        {
            coeff: expr.expand().collect(sp.exp(sp.Wild("w")), sp.simplify)
            for coeff, expr in sol.items()
        }
        for sol in sol_coeffs_0_TM
    ]

    save_latex_as_image(sol_coeffs_0_TE, "coeffs_0_TE_all")
    save_latex_as_image(sol_coeffs_0_TM, "coeffs_0_TM_all")
    print(
        "found coefficients from boundry conditions for TE- and TM-mode in zeroth order"
    )

    # TODO: check if expressions found are indeed solutions to the system
    # after substitution one of four equations has to contain Det(M),
    # which has to be zero for non-trivial solution to exist
    # (manually done in jupyter, will add here some other time)
    # ref: https://doi.org/10.1134/S0361768822020049

    return
    eqs_1, alg_eqs_1, diff_eqs_1 = gen_maxwell_eqs(
        R,
        delop,
        phi,
        E_symbols,
        H_symbols,
        E_vec_comps,
        H_vec_comps,
        E_asympt,
        H_asympt,
        order=1,
        prev_order_eqs=list_subs(
            eqs_0, gen_vars_subs([E_symbols[0], H_symbols[0]], [x], [x, y, z])
        ),
    )

    derivs_subs = {
        sp.Derivative(E_symbols[0][0](x, z), z): sp.Function("S_1")(x, z),
        sp.Derivative(H_symbols[0][1](x, z), z): sp.Function("S_2")(x, z),
        sp.Derivative(H_symbols[0][0](x, z), z): sp.Function("S_3")(x, z),
        sp.Derivative(E_symbols[0][1](x, z), z): sp.Function("S_4")(x, z),
    }

    eqs_1 = list_subs(
        eqs_1,
        gen_vars_subs([E_symbols[1], H_symbols[1]], [x, y, z], [x])
        | gen_vars_subs([E_symbols[0], H_symbols[0]], [x, y, z], [x, z])
        | derivs_subs,
    )
    alg_eqs_1 = list_subs(
        alg_eqs_1,
        gen_vars_subs([E_symbols[1], H_symbols[1]], [x, y, z], [x])
        | gen_vars_subs([E_symbols[0], H_symbols[0]], [x, y, z], [x, z])
        | derivs_subs,
    )
    diff_eqs_1 = list_subs(
        diff_eqs_1,
        gen_vars_subs([E_symbols[1], H_symbols[1]], [x, y, z], [x])
        | gen_vars_subs([E_symbols[0], H_symbols[0]], [x, y, z], [x, z])
        | derivs_subs,
    )

    # save_latex_as_image(diff_eqs_1, "diff_eqs_1")

    sols_1 = dsolve_system(
        # list_subs(diff_eqs_1, derivs_subs),
        diff_eqs_1,
        funcs=[
            E_symbols[1][1](x),
            E_symbols[1][2](x),
            H_symbols[1][1](x),
            H_symbols[1][2](x),
        ],
        t=x,
    )[0]
    print("Found solution to first-order method")

    # NOTE: A *LOT* of compute time to check
    # assert check_sols(diff_eqs_1, sols_1) == [True, True, True, True]
    # print("Checked solution to first-order method")

    sols_1 = list_subs(
        sols_1,
        {
            omega / c * sp.sqrt(-epsilon * mu + sp.diff(phi(z), z) ** 2): gamma(z),
            -epsilon * mu + sp.diff(phi(z), z) ** 2: eta(z),
        },
        eval=False,
    )

    sols_1 = [
        eq.replace(sp.Integral, lambda *args: sp.simplify(sp.Integral(*args)))
        for eq in sols_1
    ]

    # save_latex_as_image(sols_1, "general_solution_first_order")


if __name__ == "__main__":
    main()
