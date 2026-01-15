from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl
from em.formalgeo.tools import show_gc, draw_gc,get_hypergraph
import copy
import time
import json
from pprint import pprint
from pprint import pformat
from sympy import symbols
#cd EuclidMachine#
#python /home/lengmen/yyc/Projects/EuclidMachine/src/em/inductor/solver.py#
def get_entities(gc):
    entities_dict = {}

    for entity in ['Point', 'Line', 'Circle']:
        if len(gc.ids_of_predicate[entity]) == 0:
            continue

        entities_dict[entity] = {}

        for fact_id in gc.ids_of_predicate[entity]:
            name = gc.facts[fact_id][1][0]

            if entity == 'Point':
                values = [(round(float(gc.value_of_para_sym[symbols(f'{name}.x')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{name}.y')]), 4))]
            elif entity == 'Line':
                values = [(round(float(gc.value_of_para_sym[symbols(f'{name}.k')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{name}.b')]), 4))]
            else:
                values = [(round(float(gc.value_of_para_sym[symbols(f'{name}.u')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{name}.v')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{name}.r')]), 4))]

            entities_dict[entity][name] = values

    return entities_dict


def test_problem(data,num):
    problem = GeometricConfiguration(parse_gdl(load_json('/home/lengmen/szh/PythonWorkSpace/EuclidMachine/data/new_gdl/gdl-yuchang(1).json')))
    #problem = GeometricConfiguration(parse_gdl(load_json('/home/lengmen/yyc/Projects/EuclidMachine/src/em/inductor/mygdl_useless.json')))
    for i in data["constructions"]:
        problem.construct(i)
    a = parse_gdl(load_json('/home/lengmen/szh/PythonWorkSpace/EuclidMachine/data/new_gdl/gdl-yuchang(1).json'))
    #a = load_json('/home/lengmen/yyc/Projects/EuclidMachine/src/em/inductor/parsed_gdl.json')
    fact_is_change=1
    number_of_iterations=0
    ttime=[]
    print("fact_is_change", fact_is_change)
    #print("oldproblem.facts", problem.facts)
    #print("oldproblem.operations", problem.operations)
    #show_gc(problem)
    while(number_of_iterations<40 and fact_is_change==1):
        fact_is_change=0
        oldfact=copy.deepcopy(problem.facts)
        oldoperations=copy.deepcopy(problem.operations)
        #print("*******", len(oldfact))
        for name, j in a["Theorems"].items():
            if name not in ():
                print("当前第",number_of_iterations+1,"次尝试应用定理",name)
                start = time.time()
                b=problem.apply(name)
                if b:
                    #print("*******##", len(oldfact))
                    #print("*******$$$", len(problem.facts))
                    print("$$$$$$$当前可以应用定理", name,"$$$$$$$")
                    #print("changeproblem.facts", problem.facts[len(oldfact):])
                    #print("changeproblem.operations", problem.operations[len(oldoperations):])
                    fact_is_change=1
                    oldfact = copy.deepcopy(problem.facts)
                    oldoperations = copy.deepcopy(problem.operations)
                end = time.time()
                print("第",number_of_iterations+1,"次尝试应用定理",name,"所用时间",end - start, "秒")
                ttime.append(f"第{number_of_iterations+1}次尝试应用定理{name}所用时间{end - start}秒")
                print("%%%%%%%%%%%%%%%%")
        print("fact_is_change", fact_is_change)
        number_of_iterations=number_of_iterations+1


    #print("newproblem.facts", problem.facts)
    #print("newproblem.operations", problem.operations)
    show_gc(problem)
    draw_gc(problem,"png")
    #print("problem.groups",problem.groups)
    #pprint(get_hypergraph(problem),width=400)
    get_h = pformat(get_hypergraph(problem,False), width=400)
    file_path1 = f"newproblemfacts{num}.json"
    # file_path1 ="changeofproblem.json"
    with open(file_path1, "w", encoding="utf-8") as f:
        for item in problem.facts:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path1}")

    file_path2 = f"newproblemoperations{num}.json"
    # file_path2 ="newconstruction.json"
    with open(file_path2, "w", encoding="utf-8") as f:
        for item in problem.operations:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path2}")

    file_path3 = f"time{num}.json"
    # file_path2 ="newconstruction.json"
    with open(file_path3, "w", encoding="utf-8") as f:
        for item in ttime:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path3}")

    file_path4 = f"get_hypergraph{num}.json"
    with open(file_path4, "w", encoding="utf-8") as f:
        f.write(get_h)

    file_path5 = f"newproblemgroups{num}.json"
    # file_path2 ="newconstruction.json"
    with open(file_path5, "w", encoding="utf-8") as f:
        for item in problem.groups:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path5}")

    file_path6 = f"newproblementities{num}.json"
    entities=get_entities(problem)
    with open(f"newproblementities{num}.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    all_start=time.time()
    file_path = "/home/lengmen/szh/PythonWorkSpace/EuclidMachine/data/new_gdl/test_theo(1).json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    #save_readable_parsed_gdl(parse_gdl(load_json('gdl-yuchang.json')), 'parsed_gdl.json')
    #save_readable_parsed_gdl(parse_gdl(load_json('/home/lengmen/yyc/Projects/EuclidMachine/src/em/inductor/mygdl_useless.json')), '/home/lengmen/yyc/Projects/EuclidMachine/src/em/inductor/parsed_gdl.json')
    test_problem(data["2"],2)
    all_end = time.time()
    print("所用时间", all_end - all_start, "秒")

