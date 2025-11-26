from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl
from em.formalgeo.tools import show_gc, draw_gc
from pprint import pprint


# from sympy import symbols, sqrt, nonlinsolve
#
# Qx = symbols('Q.x')
# Qy = symbols('Q.y')
# eq1 = sqrt((Qx + 0.05) ** 2 + (Qy - 0.2) ** 2) - 1.5  # 超越方程，解集为空
# eq2 = ((Qx + 0.05) ** 2 + (Qy - 0.2) ** 2) ** 0.5 - 1.5  # 解集为ConditionSet
# eq3 = (Qx + 0.05) ** 2 + (Qy - 0.2) ** 2 - 1.5 ** 2  # 解集为FiniteSet
# print(eq1)
# result1 = nonlinsolve([eq1], [Qx, Qy])
# print(type(result1), result1)
# print()
# print(eq2)
# result2 = nonlinsolve([eq2], [Qx, Qy])
# print(type(result2), result2)
# print()
# print(eq3)
# result3 = nonlinsolve([eq3], [Qx, Qy])
# print(type(result3), result3)
# exit(0)


def solve(gdl, example, problem_id):
    gc = GeometricConfiguration(parse_gdl(load_json(gdl)))
    example = load_json(example)[str(problem_id)]

    for construction in example['constructions']:
        applied = gc.construct(construction)
        print(applied, construction)
        # if construction == 'Point(P):FreePoint(P)':
        #     draw_geometric_figure(gc, f'../../../data/outputs/geometric_figure_{pid}.png')
        #     exit(0)

    for theorem in example['theorems']:
        applied = gc.apply(theorem)
        print(applied, theorem)
        # if theorem == 'Point(P):FreePoint(P)':
        #     show(gc)
        #     exit(0)

    print()
    show_gc(gc)
    draw_gc(gc, f'../../../data/outputs/geometric_figure_{problem_id}.png')


if __name__ == '__main__':
    # gdl_filename = '../../../data/gdl/gdl-xiaokai.json'
    gdl_filename = '../../../data/gdl/gdl-yuchang.json'
    # example_filename = '../../../data/gdl/gc-xiaokai.json'
    example_filename = '../../../data/gdl/gc-yuchang.json'
    pid = 3

    pprint(parse_gdl(load_json(gdl_filename)))

    # solve(gdl_filename, example_filename, pid)

