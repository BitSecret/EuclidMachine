from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, show_gc, draw_gc
import copy
import time
import json

gc = GeometricConfiguration(parse_gdl(load_json('../../../data/gdl/gdl-yuchang.json')))
problem = load_json('../../../data/gdl/gc-yuchang.json')["3"]
for i in problem["constructions"]:
    gc.construct(i)
a = parse_gdl(load_json('../../../data/gdl/gdl-yuchang.json'))
fact_is_change = 1
number_of_iterations = 0
ttime = []
print("fact_is_change", fact_is_change)

while (number_of_iterations < 1 and fact_is_change == 1):
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
draw_gc(gc, '../../../data/outputs/gc.png')
