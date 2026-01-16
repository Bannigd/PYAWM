from datetime import date, datetime
from functools import reduce
from pathlib import Path

from sympy import latex, Eq
from sympy.printing import preview


def preview_collection(equations):
    if isinstance(equations, list):
        out = reduce(
            lambda x, y: x + y,
            [latex(eq, mode="equation*") for eq in equations],
            "",
        )
    if isinstance(equations, dict):
        out = reduce(
            lambda x, y: x + y,
            [latex(Eq(lhs, rhs), mode="equation*") for lhs, rhs in equations.items()],
            "",
        )
    else:
        out = latex(equations, mode="equation*")
    preview(
        out,
        output="png",
        dvioptions=["-D 200"],
        euler=False,
    )


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
