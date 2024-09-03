# algorithm is described in https://doi.org/10.1134/S0361768822020049

from itertools import chain

import sympy as sp

from main import save_latex_as_image


def solve_subsystem(eqs: list[sp.Eq], vars: list[sp.Symbol]) -> list:
    # first is to reduce system to one equation
    coeff1 = eqs[0].lhs.coeff(vars[0])
    coeff2 = eqs[1].lhs.coeff(vars[0])
    single_eq = eqs[1].lhs * coeff1 - eqs[0].lhs * coeff2
    single_eq = single_eq.simplify().collect(vars)

    # solve for 2 remaining variables
    # need minus because `single_eq` has form: a1*x1 + a2*x2=0, and method needs a1*x1 - a2*x2=0
    sol = {
        vars[1]: -single_eq.coeff(vars[2]),
        vars[2]: single_eq.coeff(vars[1]),
    }

    # find third var, which was cancelled earlier

    sol[vars[0]] = sp.solve(eqs[0].subs(sol), vars[0])[0]

    return sol


def solve_all(eqs, vars):
    # find non-zero variables in each equation
    all_sols = list()
    vars_in_eq = []
    for eq_idx in eqs:
        vars_in_eq.append([var for var in vars if var in eq_idx.lhs.atoms()])

    # find pairs with the same non-zero vars
    eq_pairs_idx = list()

    for i in range(eqs.__len__()):
        for j in range(i + 1, eqs.__len__()):
            if vars_in_eq[i] == vars_in_eq[j]:
                eq_pairs_idx.append([i, j])

    all_permutations = subsystems_permutations(eq_pairs_idx)

    for pair, eq_idx in all_permutations:
        sol = dict()
        curr_vars = vars_in_eq[pair[0]]
        sol = solve_subsystem([eqs[i] for i in pair], curr_vars)
        last_var = (set(vars) - set(curr_vars)).pop()
        sol[last_var] = sp.solve(eqs[eq_idx].subs(sol), last_var)[0]
        all_sols.append(sol)

    return all_sols


def subsystems_permutations(eq_pairs):
    """
    contructs list of type [[eq1, eq2], eq3], where [eq1, eq2] -- pair of
    equations (as index) with same variables, eq3 - equation (as index) needed
    to find last variable
    """
    res = list()
    for pair in eq_pairs:
        remainder_pair = list(chain.from_iterable([p for p in eq_pairs if p != pair]))
        for eq in remainder_pair:
            res.append([pair, eq])

    return res


def main():
    M = sp.Matrix(
        [[sp.Symbol(f"m_{{{j},{i}}}") for i in range(1, 5)] for j in range(1, 5)]
    )
    M = M.as_mutable()

    # now system(matrix) has two subsystems 2x3
    M[0, 3] = 0
    M[1, 3] = 0
    M[2, 0] = 0
    M[3, 0] = 0

    # vector of variables
    X = sp.Matrix([sp.Symbol(f"x_{{{i}}}") for i in range(1, 5)])
    X = X.as_mutable()

    slae = M * X
    eqs = [sp.Eq(eq, 0) for eq in list(slae)]

    all_sols = solve_all(eqs, list(X))
    for sol in all_sols:
        sp.pprint(
            [
                sp.Eq(
                    eq.lhs.factor()
                    .subs(sol)
                    .simplify()
                    .factor()
                    .subs({M.det(): sp.Symbol("D")}),
                    0,
                )
                for eq in eqs
            ]
        )


if __name__ == "__main__":
    main()
