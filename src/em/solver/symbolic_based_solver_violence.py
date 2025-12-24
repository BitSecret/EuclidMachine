import copy
import json
import time
from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, draw_gc, get_hypergraph


# --- 1. 保持原有的构图函数 ---
def construct_from_list(problem, construction_steps):
    print(f"开始执行 {len(construction_steps)} 步构图...")
    for i, step in enumerate(construction_steps):
        success = problem.construct(step)
        if not success:
            print(f"  [Step {i + 1}] 失败: {step}")
            return False
    print("构图序列执行完毕。")
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
            oldfact = copy.deepcopy(self.problem.facts)
            oldoperations = copy.deepcopy(self.problem.operations) if hasattr(self.problem, 'operations') else []

            print(f"\n[Epoch {number_of_iterations + 1}] 开始扫描全库定理...")

            for name, thm_data in self.theorems.items():
                if name in ():
                    continue

                print("当前第", number_of_iterations + 1, "次尝试应用定理", name)

                b = self.problem.apply(name)

                if b:
                    print("$$$$$$ 当前可以应用定理", name, "$$$$$$")

                    # 标记发生了变化
                    fact_is_change = 1

                    # 图片逻辑：每次成功都更新 oldfact
                    # (注：这样写其实效率较低，但在调试时能精准看到每一步的变化)
                    oldfact = copy.deepcopy(self.problem.facts)
                    if hasattr(self.problem, 'operations'):
                        oldoperations = copy.deepcopy(self.problem.operations)

                    print("%%%%%%%%%%%%%%%%%")

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
        "Point(D):FreePoint(D)",
        "Point(B):FreePoint(B)",
        "Point(C):PointLeftSegment(C,D,B)",
        "Line(s):PointOnLine(C,s)&PointOnLine(B,s)",
        "Line(l):PointOnLine(D,l)&PointOnLine(B,l)",
        "Line(m):PointOnLine(C,m)&LinesParallel(l,m)",
        "Line(i):PointOnLine(D,i)&PointOnLine(C,i)",
        "Line(j):PointOnLine(B,j)&LinesParallel(i,j)",
        "Point(E):PointOnLine(E,j)&PointOnLine(E,m)"
    ]

    # 执行构图
    if construct_from_list(problem, constructions):
        # 初始化并运行求解器
        solver = BruteForceSolver(problem, gdl_path)
        solver.solve(max_iterations=10)

        # 可选：保存结果
        # draw_gc(problem, 'data/output-brute.png')


if __name__ == "__main__":
    run_test()