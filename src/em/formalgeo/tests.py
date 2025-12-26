import copy
import time
from sympy import symbols, sqrt, nonlinsolve
from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, save_json, get_vocab
from em.formalgeo.tools import show_gc, draw_gc, get_hypergraph
from sympy import *
from pprint import pprint
from random import Random


def test1():
    gc = GeometricConfiguration(parse_gdl(load_json(gdl_filename)))
    problem = load_json(example_filename)[str(pid)]
    for i in problem["constructions"]:
        gc.construct(i)
    a = parse_gdl(load_json(gdl_filename))
    fact_is_change = 1
    number_of_iterations = 0
    ttime = []
    print("fact_is_change", fact_is_change)

    while number_of_iterations < 1 and fact_is_change == 1:
        fact_is_change = 0
        oldfact = copy.deepcopy(gc.facts)
        oldoperations = copy.deepcopy(gc.operations)
        # print("*******", len(oldfact))
        for name, j in a["Theorems"].items():
            if name not in ():
                print("当前第", number_of_iterations + 1, "次尝试应用定理", name)
                start = time.time()
                b = gc.apply(name)
                if b:
                    # print("*******##", len(oldfact))
                    # print("*******$$$", len(gc.facts))
                    print("$$$$$$$当前可以应用定理", name, "$$$$$$$")
                    # print("changegc.facts", gc.facts[len(oldfact):])
                    # print("changegc.operations", gc.operations[len(oldoperations):])
                    fact_is_change = 1
                    oldfact = copy.deepcopy(gc.facts)
                    oldoperations = copy.deepcopy(gc.operations)
                end = time.time()
                print("第", number_of_iterations + 1, "次尝试应用定理", name, "所用时间", end - start, "秒")
                ttime.append(f"第{number_of_iterations + 1}次尝试应用定理{name}所用时间{end - start}秒")
                print("%%%%%%%%%%%%%%%%")
        print("fact_is_change", fact_is_change)
        number_of_iterations = number_of_iterations + 1

    show_gc(gc)
    draw_gc(gc, f'../../../data/outputs/test1_figure_{pid}.png')


def test2():
    gc = GeometricConfiguration(parse_gdl(load_json(gdl_filename)))
    example = load_json(example_filename)[str(pid)]

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
    show_gc(gc, target=example['targets'][0])
    draw_gc(gc, f'../../../data/outputs/test2_figure_{pid}.png')
    save_json(get_hypergraph(gc), f'../../../data/outputs/test2_hypergraph_{pid}.json')
    save_json(get_hypergraph(gc, serialize=True), f'../../../data/outputs/test2_serialized_hypergraph_{pid}.json')


def test3():
    Qx = symbols('Q.x')
    Qy = symbols('Q.y')
    eq1 = sqrt((Qx + 0.05) ** 2 + (Qy - 0.2) ** 2) - 1.5  # 超越方程，解集为空
    eq2 = ((Qx + 0.05) ** 2 + (Qy - 0.2) ** 2) ** 0.5 - 1.5  # 解集为ConditionSet
    eq3 = (Qx + 0.05) ** 2 + (Qy - 0.2) ** 2 - 1.5 ** 2  # 解集为FiniteSet
    print(eq1)
    result1 = nonlinsolve([eq1], [Qx, Qy])
    print(type(result1), result1)
    print()
    print(eq2)
    result2 = nonlinsolve([eq2], [Qx, Qy])
    print(type(result2), result2)
    print()
    print(eq3)
    result3 = nonlinsolve([eq3], [Qx, Qy])
    print(type(result3), result3)
    exit(0)


def test4():
    pprint(parse_gdl(load_json(gdl_filename)))


def test5():
    a, b, c, t = symbols('a b c t')
    print(type(nonlinsolve([t - a - b, a + b], [t, a, b])), nonlinsolve([t - a - b, a + b], [t, a, b]))
    print(type(nonlinsolve([t - a - b, t - a + b], [t, a, b])), nonlinsolve([t - a - b, t - a + b], [t, a, b]))
    print(type(nonlinsolve([t + 1, t - 1], [t, a, b])), nonlinsolve([t + 1, t - 1], [t, a, b]))


def test6():
    vocab = get_vocab(parse_gdl(load_json(gdl_filename)))
    print(len(vocab), vocab)


def test7():
    a, b = symbols('a b')
    result_a = nonlinsolve([a - 3, a + 2], [a])
    result_b = nonlinsolve([b - 3], [b])
    print("result_a:", result_a)
    print("type(result_a):", type(result_a))
    print("type(EmptySet):", type(EmptySet))
    print("result_a is EmptySet:", result_a is EmptySet)
    print("result_a is FiniteSet:", result_a is FiniteSet)
    print("type(result_a) is EmptySet:", type(result_a) is EmptySet)
    print("type(result_a) is type(EmptySet):", type(result_a) is type(EmptySet))
    print("type(result_a) == type(EmptySet):", type(result_a) == type(EmptySet))
    print()
    print("result_b:", result_b)
    print("type(result_b):", type(result_b))
    print("type(FiniteSet):", type(FiniteSet))
    print("result_b is FiniteSet:", result_b is FiniteSet)
    print("type(result_b) is FiniteSet:", type(result_b) is FiniteSet)
    print("type(result_b) is type(FiniteSet):", type(result_b) is type(FiniteSet))
    print("type(result_b) == type(FiniteSet):", type(result_b) == type(FiniteSet))


def test8():
    pprint(parse_gdl(load_json(gdl_filename)))


def test9():
    random = Random(42)  # 固定种子

    new_random = copy.copy(random)

    print("原始实例:", random.random())  # 0.025010...
    print("原始实例:", random.random())  # 0.025010...
    print("新实例:  ", new_random.random())  # 0.275029...
    print("新实例:  ", new_random.random())  # 0.275029...


# gdl_filename = '../../../data/gdl/gdl-xiaokai.json'
gdl_filename = '../../../data/gdl/gdl-yuchang.json'
# example_filename = '../../../data/gdl/gc-xiaokai.json'
example_filename = '../../../data/gdl/gc-yuchang.json'
pid = 3

if __name__ == '__main__':
    test9()
