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
    return list(map(lambda eq: eq.subs(subs), eqs))


# def construct_maxwell_eqs():
#     """
#     constructs and returns maxwell equations (only vector part)
#     """
#     R = CoordSys3D("")
#     delop = Del()  # nabla
#     Ex, Ey, Ez = sp.symbols("E_x E_y E_z", cls=sp.Function)
#     Hx, Hy, Hz = sp.symbols("H_x H_y H_z", cls=sp.Function)
#     E_vfield = (
#         Ex(R.x, R.y, R.z, t) * R.i
#         + Ey(R.x, R.y, R.z, t) * R.j
#         + Ez(R.x, R.y, R.z, t) * R.k
#     )
#     H_vfield = (
#         Hx(R.x, R.y, R.z, t) * R.i
#         + Hy(R.x, R.y, R.z, t) * R.j
#         + Hz(R.x, R.y, R.z, t) * R.k
#     )
#     curl_E = (delop.cross(E_vfield) + mu / c * H_vfield.diff(t)).to_matrix(R)
#     curl_H = (delop.cross(H_vfield) - epsilon / c * E_vfield.diff(t)).to_matrix(R)
#     return curl_E, curl_H
#


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
    E_asympt = sum(
        map(
            lambda k, vec: vec / (sp.I * omega) ** k,
            range(order + 1),
            E_vec_comps,
        ),
        0 * R.i,
    )  # 0*R.i determines return type
    H_asympt = sum(
        map(
            lambda k, vec: vec / (sp.I * omega) ** k,
            range(order + 1),
            H_vec_comps,
        ),
        0 * R.i,
    )
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
):
    curl_E = (
        (
            (
                delop.cross(
                    E_asympt * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                )
                + mu
                / c
                * (
                    H_asympt * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                ).diff(t)
            )
            .doit()
            .to_matrix(R)
            / sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
        )
        .expand()
        .subs(gen_subs_diff(R, E_vec_comps[0]))
    )

    curl_H = (
        (
            (
                delop.cross(
                    H_asympt * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                )
                + epsilon
                / c
                * (
                    E_asympt * sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
                ).diff(t)
            )
            .doit()
            .to_matrix(R)
            / sp.exp(sp.I * omega * t - sp.I * omega / c * phi(R.z))
        )
        .expand()
        .subs(gen_subs_diff(R, H_vec_comps[0]))
    )
    maxwell_eqs = list_subs(
        [sp.Eq(eq, 0) for eq in list(curl_E) + list(curl_H)], {R.x: x, R.y: y, R.z: z}
    )
    maxwell_alg_eqs = {
        H_symbols[0][0](x, y, z): sp.solve(maxwell_eqs[0], H_symbols[0][0](x, y, z))[0],
        E_symbols[0][0](x, y, z): sp.solve(maxwell_eqs[3], E_symbols[0][0](x, y, z))[0],
    }

    maxwell_diff_eqs = list(
        filter(
            lambda x: x is not sp.sympify(True), list_subs(maxwell_eqs, maxwell_alg_eqs)
        )
    )
    vars_subs = gen_vars_subs([E_symbols[0], H_symbols[0]])

    maxwell_diff_eqs = list_subs(maxwell_diff_eqs, vars_subs)
    return maxwell_alg_eqs, maxwell_diff_eqs


def main():
    # setup sympy symbols
    R = CoordSys3D("")
    delop = Del()  # nabla
    phi = sp.Function("phi")
    order = 0  # order of expansion
    E_symbols, H_symbols, E_vec_comps, H_vec_comps, E_asympt, H_asympt = (
        gen_vector_field_symbols(R, order=order)
    )

    # contsruct Maxwell's equations
    maxwell_alg_eqs, maxwell_diff_eqs = gen_maxwell_eqs(
        R,
        delop,
        phi,
        E_symbols,
        H_symbols,
        E_vec_comps,
        H_vec_comps,
        E_asympt,
        H_asympt,
    )

    # solve ODE system
    sols = dsolve_system(
        maxwell_diff_eqs,
        funcs=[
            E_symbols[0][1](x),
            E_symbols[0][2](x),
            H_symbols[0][1](x),
            H_symbols[0][2](x),
        ],
        t=x,
    )[0]

    # preview([curl_E, curl_H], output="png", dvioptions=["-D 600"], euler=False)
    # preview([maxwell_diff_eqs, sols], output="png", dvioptions=["-D 600"], euler=False)

    # check if solution is correct
    sols_subs = {sol.lhs: sol.rhs for sol in sols}
    checked = [sol.subs(sols_subs).simplify() for sol in sols]
    assert checked == [True, True, True, True]


if __name__ == "__main__":
    main()
