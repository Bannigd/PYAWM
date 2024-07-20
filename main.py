from itertools import chain, product

import sympy as sp
from sympy.abc import c, epsilon, mu, omega, t, x, y, z
from sympy.printing import preview
from sympy.solvers.ode.systems import dsolve_system
from sympy.vector import CoordSys3D, Del


def gen_vars_subs(U_symbols):
    return {U(x, y, z): U(x) for U in list(chain.from_iterable(U_symbols))}


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
    sols_subs = {sol.lhs.func(x, y, z): sol.rhs for sol in sols}
    check = list_subs(eqs, sols_subs)
    return check


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
        [sp.Eq(eq, 0) for eq in list(curl_E) + list(curl_H)], {R.x: x, R.y: y, R.z: z}
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
    # WARNING:order of symbols here is "hard-coded" for author's convenience
    match layer_symbol:
        case "c":
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
                sp.Function(f"eta")(z): sp.Function(f"eta_{layer_symbol}")(z),
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


def gen_boundry_conds(R, delop, U, border_func, func_value, layers, order):
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

    boundry_eqs = (nU_1 - nU_2).subs({x: func_value, R.y: y, R.z: z})

    return [sp.Eq(boundry_eqs.components[arg], 0) for arg in [R.j, R.k]]


def main():
    # setup sympy symbols
    R = CoordSys3D("")
    delop = Del()  # nabla
    phi = sp.Function("phi")
    eta = sp.Function("eta")
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

    # eqs_1, alg_eqs_1, diff_eqs_1 = gen_maxwell_eqs(
    #     R,
    #     delop,
    #     phi,
    #     E_symbols,
    #     H_symbols,
    #     E_vec_comps,
    #     H_vec_comps,
    #     E_asympt,
    #     H_asympt,
    #     order=1,
    #     prev_order_eqs=eqs_0,
    # )

    vars_subs_0 = gen_vars_subs([E_symbols[0], H_symbols[0]])
    vars_subs_1 = gen_vars_subs([E_symbols[1], H_symbols[1]])

    # solve ODE system
    diff_sols_0 = dsolve_system(
        list_subs(diff_eqs_0, vars_subs_0),
        funcs=[
            E_symbols[0][1](x),
            E_symbols[0][2](x),
            H_symbols[0][1](x),
            H_symbols[0][2](x),
        ],
        t=x,
    )[0]
    assert check_sols(diff_eqs_0, diff_sols_0) == [True, True, True, True]

    sols_0 = [
        alg_eqs_0[0].subs({eq.lhs.func(x, y, z): eq.rhs for eq in diff_sols_0}),
        diff_sols_0[0],
        diff_sols_0[1],
        alg_eqs_0[1].subs({eq.lhs.func(x, y, z): eq.rhs for eq in diff_sols_0}),
        diff_sols_0[2],
        diff_sols_0[3],
    ]
    sols_0 = list_subs(
        sols_0, {sp.sqrt(-epsilon * mu + sp.diff(phi(z), z) ** 2): eta(z)}
    )
    # sols_1 = dsolve_system(
    #     list_subs(diff_eqs_1, vars_subs_1),
    #     funcs=[
    #         E_symbols[1][1](x),
    #         E_symbols[1][2](x),
    #         H_symbols[1][1](x),
    #         H_symbols[1][2](x),
    #     ],
    #     t=x,
    # )[0]
    # assert check_sols(diff_eqs_1, sols_1) == [True, True, True, True]

    # preview(sols_1[0], output="png", dvioptions=["-D 600"], euler=False)
    # preview(sols_1[1], output="png", dvioptions=["-D 600"], euler=False)
    # preview(sols_1[2], output="png", dvioptions=["-D 600"], euler=False)
    # preview(sols_1[3], output="png", dvioptions=["-D 600"], euler=False)

    # Construct solutions for different layers
    sols_0_layers = {
        layer: layered_sols(sols_0, layer, order=0) for layer in ["c", "f", "s"]
    }

    # Now solving for 2D waveguide with smoothly irregular transition, x=h(z)

    # boundry conditions
    h = sp.Function("h")
    border_func = R.x - h(R.z)
    E_boundry_cf = gen_boundry_conds(
        R, delop, E_vec_comps[0], border_func, h(z), ["c", "f"], order=0
    )

    H_boundry_cf = gen_boundry_conds(
        R, delop, H_vec_comps[0], border_func, h(z), ["c", "f"], order=0
    )

    E_boundry_fs = gen_boundry_conds(
        R, delop, E_vec_comps[0], border_func, 0, ["f", "s"], order=0
    )

    H_boundry_fs = gen_boundry_conds(
        R, delop, H_vec_comps[0], border_func, 0, ["f", "s"], order=0
    )

    boundry_eqs = list_subs(
        E_boundry_cf + H_boundry_cf + E_boundry_fs + H_boundry_fs,
        {eq.lhs: eq.rhs for eq in (sols_0_layers["c"] + sols_0_layers["f"])},
    )
    coeffs = [
        sp.Symbol(f"A_0^c"),
        sp.Symbol(f"B_0^c"),
        sp.Symbol(f"A_0^f"),
        sp.Symbol(f"B_0^f"),
        sp.Symbol(f"C_0^f"),
        sp.Symbol(f"D_0^f"),
        sp.Symbol(f"A_0^s"),
        sp.Symbol(f"B_0^s"),
    ]


if __name__ == "__main__":
    main()
