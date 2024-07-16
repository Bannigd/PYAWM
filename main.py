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


def list_subs(eqs: list, subs: dict):
    return list(map(lambda eq: eq.subs(subs).expand().simplify(), eqs))


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
                + epsilon
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

    # WARNING: only for zero order for now
    maxwell_alg_eqs = {
        H_symbols[order][0](x, y, z): sp.solve(
            maxwell_eqs[0], H_symbols[order][0](x, y, z)
        )[0],
        E_symbols[order][0](x, y, z): sp.solve(
            maxwell_eqs[3], E_symbols[order][0](x, y, z)
        )[0],
    }

    maxwell_diff_eqs = list(
        filter(
            lambda x: x is not (sp.sympify(True) or 0),
            list_subs(maxwell_eqs, maxwell_alg_eqs),
        )
    )
    return maxwell_eqs, maxwell_alg_eqs, maxwell_diff_eqs


def main():
    # setup sympy symbols
    R = CoordSys3D("")
    delop = Del()  # nabla
    phi = sp.Function("phi")
    max_order = 1  # order of expansion
    E_symbols, H_symbols, E_vec_comps, H_vec_comps, E_asympt, H_asympt = (
        gen_vector_field_symbols(R, order=max_order)
    )

    # contsruct Maxwell's equations
    maxwell_eqs, maxwell_alg_eqs, maxwell_diff_eqs = gen_maxwell_eqs(
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

    vars_subs_0 = gen_vars_subs([E_symbols[0], H_symbols[0]])
    vars_subs_1 = gen_vars_subs([E_symbols[1], H_symbols[1]])

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
        prev_order_eqs=maxwell_eqs,
    )

    preview(
        list_subs(diff_eqs_1, vars_subs_1),
        output="png",
        dvioptions=["-D 600"],
        euler=False,
    )

    # solve ODE system
    sols = dsolve_system(
        list_subs(diff_eqs_1, vars_subs_1),
        funcs=[
            E_symbols[1][1](x),
            E_symbols[1][2](x),
            H_symbols[1][1](x),
            H_symbols[1][2](x),
        ],
        t=x,
    )[0]

    preview(sols[0].simplify(), output="png", dvioptions=["-D 600"], euler=False)
    # preview(sols[1], output="png", dvioptions=["-D 600"], euler=False)
    # preview(sols[2], output="png", dvioptions=["-D 600"], euler=False)
    # preview(sols[3], output="png", dvioptions=["-D 600"], euler=False)

    # check if solution is correct
    sols_subs = {sol.lhs: sol.rhs for sol in sols}
    checked = [sol.subs(sols_subs).simplify() for sol in sols]
    assert checked == [True, True, True, True]


if __name__ == "__main__":
    main()
