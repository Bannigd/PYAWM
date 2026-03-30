# WIP

Эта ветка находится в разработке и в планируется стать основной когда, будут реализованы все возмжности из версии в main.
Здесь активно используются классы для группирования атрибутов, а также setattr и getattr методы, которые упрощают
интерктивную работу с получающимися формулами.

# pyawm/main.py

Модуль символьного решения уравнений Максвелла для многослойных волноведущих структур.

## Классы

### `Layer` (Enum)
Типы слоёв волновода:
- `COVER` ("c") —  верхний слой
- `FILM` ("f") —  основной слой волновода
- `LENS` ("l") —  слой линзы (для линзы Люнеберга)
- `SUBSTRATE` ("s") —  подложка

Алиасы: `C`, `F`, `L`, `S`

### `Waveguide`
Описание геометрии волновода:
```python
Waveguide(field_vars, layers, geometry, phi_vars)
```
- `field_vars` —  переменные поля (например для 2D волновода: x, z)
- `layers` —  список слоёв `[Layer.SUBSTRATE, Layer.FILM, Layer.COVER]`
- `geometry` —  кортежи `(left_layer, right_layer, border_func)` для границ
- `phi_vars` —  переменные функции фазы поля

### `Domain`
Основной решатель. Принимает `Waveguide`.

**Методы:**
- `solve_general_form_zero()` —  решение в нулевом приближении
- `solve_general_form_first()` —  решение в первом приближении
- `construct_boundary_equations(order)` —  построение граничных условий

## Пример использования

```python
from pyawm.main import Layer, Waveguide, Domain
import sympy as sp

field_vars = [x, z]
phi_vars = [z]
layers = [Layer.SUBSTRATE, Layer.FILM, Layer.COVER]

geometry = [
    (Layer.FILM,      Layer.COVER,     sp.Function("h_f")),
    (Layer.SUBSTRATE, Layer.FILM,      sp.Function("h_s"))
]

horn = Waveguide(field_vars, layers, geometry, phi_vars)
d = Domain(horn)
d.solve_general_form_zero()
d.solve_general_form_first()
```
