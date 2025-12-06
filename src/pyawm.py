from itertools import chain, product

import sympy as sp
from sympy.abc import c, epsilon, mu, omega, t, x, y, z


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
    if eval:
        return [eq.subs(subs).expand().simplify() for eq in eqs]
    return [eq.subs(subs) for eq in eqs]

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
            lambda x: x is not (sp.true or 0),
            list_subs(maxwell_eqs, {eq.lhs: eq.rhs for eq in maxwell_alg_eqs}),
        )
    )
    return maxwell_eqs, maxwell_alg_eqs, maxwell_diff_eqs


def layered_sols(sols, layer_symbol, order) -> list:
    # TODO: add a param to signal what exponents (with `+` or `-`) to omit
    #
    # NOTE: in some layers we make a substitution of x->x-h_i(z). 
    # The layers to do this subs are chosen manually, and for now i dont know how to automate this criteria.
    # Maybe just provide a flag for each layer to do or not to do this substitution.
    # 
    # WARNING: order of substituting symbols here is "hard-coded" for author's convenience
    if order > 0:
        prev_U_subs = {
            sp.Function("E^x_0")(x, z): sp.Function(f"E^{{x,{layer_symbol}}}_{0}")(
                x, z
            ),
            sp.Function("E^y_0")(x, z): sp.Function(f"E^{{y,{layer_symbol}}}_{0}")(
                x, z
            ),
            sp.Function("E^z_0")(x, z): sp.Function(f"E^{{z,{layer_symbol}}}_{0}")(
                x, z
            ),
            sp.Function("H^x_0")(x, z): sp.Function(f"H^{{x,{layer_symbol}}}_{0}")(
                x, z
            ),
            sp.Function("H^y_0")(x, z): sp.Function(f"H^{{y,{layer_symbol}}}_{0}")(
                x, z
            ),
            sp.Function("H^z_0")(x, z): sp.Function(f"H^{{z,{layer_symbol}}}_{0}")(
                x, z
            ),
        }
    else:
        prev_U_subs = {}

    U_p_subs = {
        sp.Function(f"{U}^{{{i}}}_p")(x, z): sp.Function(
            f"{U}^{{{i},{layer_symbol}}}_p"
        )(x, z)
        for U, i in product(["E", "H"], [x, y, z])
    }

    match layer_symbol:
        case "c":
            # x -> x - h(z) eases calculation (only for 0-order for now)
            if order == 0:
                sols = [
                    sol.replace(
                        sp.exp,
                        lambda arg: sp.exp(arg.subs({x: x - sp.Function("h_1")(z)})),
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
                }
                | prev_U_subs
                | U_p_subs,
                eval=False,
            )

        case "l":
            if order == 0:
                sols = [
                    sol.replace(
                        sp.exp,
                        lambda arg: sp.exp(arg.subs({x: x - sp.Function("h_2")(z)})),
                    )
                    for sol in sols
                ]

            sols = list_subs(
                sols,
                {
                    sp.Symbol("C1"): sp.Symbol(f"A_{order}^{layer_symbol}"),
                    sp.Symbol("C2"): sp.Symbol(f"B_{order}^{layer_symbol}"),
                    sp.Symbol("C3"): sp.Symbol(f"C_{order}^{layer_symbol}"),
                    sp.Symbol("C4"): sp.Symbol(f"D_{order}^{layer_symbol}"),
                }
                | prev_U_subs
                | U_p_subs,
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
                }
                | prev_U_subs
                | U_p_subs,
                eval=False,
            )

        case "s":
            sols = list_subs(
                sols,
                {
                    sp.Symbol("C1"): 0,
                    sp.Symbol("C2"): sp.Symbol(f"A_{order}^{layer_symbol}"),
                    sp.Symbol("C3"): 0,
                    sp.Symbol("C4"): sp.Symbol(f"B_{order}^{layer_symbol}"),
                }
                | prev_U_subs
                | U_p_subs,
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
                sp.Function("gammatilde")(z): sp.Function(f"gammatilde_{layer_symbol}")(
                    z
                ),
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
    char_U = str(next(iter(U.components.values())))[0]
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


def boundry_check_all_solutions(eqs, vars, all_sols):
    """
    we need to show, that equation is True or is a multiple of Determinant
    ref: https://doi.org/10.1134/S0361768822020049
    """
    M, b = sp.linear_eq_to_matrix(eqs, vars)

    if b != sp.zeros(len(vars), 1):
        raise ValueError("`eqs` must be a homogeneous system")

    eqs_lhs = [eq.lhs - eq.rhs for eq in eqs]

    det = M.det(method="berkowitz").expand().combsimp()

    for i, sol in enumerate(all_sols):
        system = list_subs(eqs_lhs, sol, eval=False)
        system = [
            eq.expand()
            .combsimp()
            .subs({det: sp.Symbol("|M|"), -det: -sp.Symbol("|M|")})
            for eq in system
        ]

        print(f"for solution {i+1}:")
        for eq in system:
            if eq == 0:
                print("\tOK")
            elif eq.has(sp.Symbol("|M|")):
                print("\tOK (has det(M))")
        # preview_collection(system)


def add_layer_index(symbol, layer):
    if symbol.is_Function:
        return sp.Function(f'{symbol.name}_{layer}')
    if symbol.is_Symbol:        
        return sp.Symbol(f'{symbol.name}_{layer}')
    raise TypeError("symbol provided is not sympy Function or Symbol (`is_Function` and `is_Symbol` checks failed).") 
