import copy
import json
import time
from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, draw_gc, get_hypergraph
from pprint import pformat

# --- 构图函数 ---
def construct_from_list(problem, construction_steps):
    print(f"开始执行 {len(construction_steps)} 步构图...")
    for i, step in enumerate(construction_steps):
        success = problem.construct(step)
        if not success:
            print(f"  [Step {i + 1}] 失败: {step}")
            return False
    print("构图序列执行完毕。")
    hypergraph = get_hypergraph(problem)

    with open('data/hypergraph-3-raw.json', 'w', encoding='utf-8') as f:
        json.dump(hypergraph, f, ensure_ascii=False, indent=4)

    return True


# --- 2. 暴力求解器 (完全复刻图片逻辑) ---
class BruteForceSolver:
    def __init__(self, problem, theorem_lib_path):
        self.problem = problem
        # 加载 JSON 并解析，获取定理列表
        self.raw_json = load_json(theorem_lib_path)
        self.parsed_gdl = parse_gdl(self.raw_json)
        # 这里对应图片里的 solver.parsed_gdl["Theorems"]
        self.theorems = self.parsed_gdl.get('Theorems', {})
        print(f"求解器初始化完成，加载了 {len(self.theorems)} 个定理。")

    def solve(self, max_iterations=20):
        print("=== 开始暴力枚举求解 (Image Logic) ===")

        # 对应图片里的 oldfact 和 number_of_iterations 逻辑
        number_of_iterations = 0
        fact_is_change = 1  # 初始设为 1 以进入循环

        # 外层循环：只要上一轮有 Fact 变化，就继续跑下一轮
        while fact_is_change and number_of_iterations < max_iterations:
            fact_is_change = 0  # 本轮开始前重置为 0

            # 记录本轮开始前的状态 (对应图片 oldfact)
            # oldfact = copy.deepcopy(self.problem.facts)
            # oldoperations = copy.deepcopy(self.problem.operations) if hasattr(self.problem, 'operations') else []

            print(f"\n[Epoch {number_of_iterations + 1}] 开始扫描全库定理...")

            for name, thm_data in self.theorems.items():
                if name in ():
                    continue

                flag = self.problem.apply(name)

                if flag:
                    print(f"当前可以应用定理:{name}")

                    # 标记发生了变化
                    fact_is_change = 1

                    # 图片逻辑：每次成功都更新 oldfact
                    # (注：这样写其实效率较低，但在调试时能精准看到每一步的变化)
                    # oldfact = copy.deepcopy(self.problem.facts)
                    # if hasattr(self.problem, 'operations'):
                    #     oldoperations = copy.deepcopy(self.problem.operations)

            if fact_is_change == 0:
                print(">>> 不动点达成：本轮遍历所有定理后无新事实生成。")
                break

            number_of_iterations += 1

        print(f"\n=== 求解结束 ===")
        print(f"总共执行轮数: {number_of_iterations}")
        print(f"最终 Facts 数量: {len(self.problem.facts)}")
        return self.problem


# --- 3. 运行测试 ---
def run_test():
    # 路径配置
    gdl_path = 'E:\PythonWorkSpace\EuclidMachine\data\\new_gdl\gdl-yuchang.json'

    # 初始化问题
    raw_gdl = load_json(gdl_path)
    parsed_gdl = parse_gdl(raw_gdl)
    problem = GeometricConfiguration(parsed_gdl)

    # 构图步骤
    constructions = [
        "Point(A):FreePoint(A)",
      "Point(B):FreePoint(B)",
      "Point(C):PointLeftSegment(C,A,B)",
      "Line(a):PointOnLine(B,a)&PointOnLine(C,a)",
      "Line(b):PointOnLine(A,b)&PointOnLine(C,b)",
      "Line(c):PointOnLine(A,c)&PointOnLine(B,c)",
      "Point(P):FreePoint(P)",
      "Point(Q):SegmentEqualSegment(P,Q,A,B)",
      "Line(r):PointOnLine(P,r)&PointOnLine(Q,r)",
      "Line(q):PointOnLine(P,q)&AngleEqualAngle(b,c,q,r)",
      "Point(R):SegmentEqualSegment(A,C,P,R)&PointLeftSegment(R,P,Q)&PointOnLine(R,q)",
      "Line(p):PointOnLine(Q,p)&PointOnLine(R,p)",
      "Line(l):PointOnLine(P,l)&AngleEqualAngle(b,c,r,l)",
      "Line(m):PointOnLine(Q,m)&AngleEqualAngle(m,r,c,a)",
      "Point(E):PointOnLine(E,l)&PointOnLine(E,m)",
      "Line(n):PointOnLine(E,n)&PointOnLine(R,n)",
      "Point(D):PointOnLine(D,r)&PointOnLine(D,n)"
    ]

    # 执行构图
    if construct_from_list(problem, constructions):
        # 初始化并运行求解器
        solver = BruteForceSolver(problem, gdl_path)
        # 可选：保存结果
        draw_gc(problem, 'data/problem-3.png')
        start_time = time.time()

        solver.solve(max_iterations=15)

        end_time = time.time()
        print(f"推理用时: {end_time - start_time}")

        # file_path1 = f"data/newproblemfacts_violence.json"
        # with open(file_path1, "w", encoding="utf-8") as f:
        #     for item in problem.facts:
        #         f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
        # print(f"已生成新文件: {file_path1}")
        #
        # file_path2 = f"data/newproblemoperations_violence.json"
        # with open(file_path2, "w", encoding="utf-8") as f:
        #     for item in problem.operations:
        #         f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
        # print(f"已生成新文件: {file_path2}")
        #
        # file_path3 = f"data/newproblemgroups_violence.json"
        # with open(file_path3, "w", encoding="utf-8") as f:
        #     for item in problem.groups:
        #         f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
        # print(f"已生成新文件: {file_path3}")
        #
        # get_h = pformat(get_hypergraph(problem, False), width=400)
        # file_path4 = f"data/get_hypergraph_violence.json"
        # with open(file_path4, "w", encoding="utf-8") as f:
        #     f.write(get_h)

        hypergraph = get_hypergraph(problem)
        with open('data/hypergraph-3-solve.json', 'w', encoding='utf-8') as f:
            json.dump(hypergraph, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_test()