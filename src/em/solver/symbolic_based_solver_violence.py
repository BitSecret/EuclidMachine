import copy
import json
import time
import logging
import os
from datetime import datetime

from sympy import symbols

from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, draw_gc, get_hypergraph
from pprint import pformat


# --- 日志配置 ---
def setup_logging():
    # 确保日志目录存在
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'solver_run_{timestamp}.log')

    # 配置 logging
    # format参数决定了日志每一行的格式，这里加上了具体时间
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 写入文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    return logging.getLogger()


# 初始化 logger
logger = setup_logging()

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

# --- 构图函数 ---
def construct_from_list(problem, construction_steps):
    logger.info(f"开始执行 {len(construction_steps)} 步构图...")
    for i, step in enumerate(construction_steps):
        success = problem.construct(step)
        if not success:
            logger.info(f"  [Step {i + 1}] 失败: {step}")
            return False
    logger.info("构图序列执行完毕。")
    hypergraph = get_hypergraph(problem)

    # 确保目录存在
    os.makedirs('data', exist_ok=True)

    with open('data/hypergraph-2-raw.json', 'w', encoding='utf-8') as f:
        json.dump(hypergraph, f, ensure_ascii=False, indent=4)

    return True


# --- 2. 暴力求解器  ---
class BruteForceSolver:
    def __init__(self, problem, theorem_lib_path):
        self.problem = problem
        # 加载 JSON 并解析，获取定理列表
        self.raw_json = load_json(theorem_lib_path)
        self.parsed_gdl = parse_gdl(self.raw_json)
        # 这里对应图片里的 solver.parsed_gdl["Theorems"]
        self.theorems = self.parsed_gdl.get('Theorems', {})
        logger.info(f"求解器初始化完成，加载了 {len(self.theorems)} 个定理。")

    def solve(self, max_iterations=20):
        logger.info("=== 开始暴力枚举求解 (Image Logic) ===")

        # 对应图片里的 oldfact 和 number_of_iterations 逻辑
        number_of_iterations = 0
        fact_is_change = 1  # 初始设为 1 以进入循环

        # 外层循环：只要上一轮有 Fact 变化，就继续跑下一轮
        while fact_is_change and number_of_iterations < max_iterations:
            fact_is_change = 0  # 本轮开始前重置为 0

            # 记录本轮开始前的状态 (对应图片 oldfact)
            # oldfact = copy.deepcopy(self.problem.facts)
            # oldoperations = copy.deepcopy(self.problem.operations) if hasattr(self.problem, 'operations') else []

            logger.info(f"\n[Epoch {number_of_iterations + 1}] 开始扫描全库定理...")

            for name, thm_data in self.theorems.items():
                if name in ():
                    continue

                flag = self.problem.apply(name)

                if flag:
                    logger.info(f"当前可以应用定理:{name}")

                    # 标记发生了变化
                    fact_is_change = 1

                    # 图片逻辑：每次成功都更新 oldfact
                    # (注：这样写其实效率较低，但在调试时能精准看到每一步的变化)
                    # oldfact = copy.deepcopy(self.problem.facts)
                    # if hasattr(self.problem, 'operations'):
                    #     oldoperations = copy.deepcopy(self.problem.operations)

            if fact_is_change == 0:
                logger.info(">>> 不动点达成：本轮遍历所有定理后无新事实生成。")
                break

            number_of_iterations += 1

        logger.info(f"\n=== 求解结束 ===")
        logger.info(f"总共执行轮数: {number_of_iterations}")
        logger.info(f"最终 Facts 数量: {len(self.problem.facts)}")
        return self.problem


# --- 3. 运行测试 ---
def run_test():
    # 路径配置
    gdl_path = '/home/lengmen/szh/PythonWorkSpace/EuclidMachine/data/new_gdl/gdl-yuchang.json'

    # 初始化问题
    # 注意：需要确保这些路径存在，或者加一些 try-except 处理
    try:
        raw_gdl = load_json(gdl_path)
    except FileNotFoundError:
        logger.error(f"文件未找到: {gdl_path}")
        return

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
        "Point(Q):SegmentEqualSegment(A,B,P,Q)",
        "Point(R):SegmentEqualSegment(A,C,P,R)&SegmentEqualSegment(B,C,Q,R)&PointLeftSegment(R,P,Q)",
        "Line(p):PointOnLine(Q,p)&PointOnLine(R,p)",
        "Line(r):PointOnLine(P,r)&PointOnLine(Q,r)",
        "Line(q):PointOnLine(P,q)&PointOnLine(R,q)",
        "Line(l):PointOnLine(Q,l)&AngleEqualAngle(c,a,p,l)",
        "Line(m):PointOnLine(R,m)&AngleEqualAngle(m,p,a,b)",
        "Point(S):PointOnLine(S,l)&PointOnLine(S,m)",
        "Line(n):PointOnLine(P,n)&PointOnLine(S,n)",
        "Point(D):PointOnLine(D,n)&PointOnLine(D,p)",
        "Line(i):PointOnLine(A,i)&PointOnLine(P,i)",
        "Line(j):PointOnLine(D,j)&PointOnLine(B,j)"
    ]

    # 执行构图
    if construct_from_list(problem, constructions):
        # 初始化并运行求解器
        solver = BruteForceSolver(problem, gdl_path)

        # 确保输出目录存在
        os.makedirs('data', exist_ok=True)
        draw_gc(problem, 'data/problem-2.png')

        start_time = time.time()

        solver.solve(max_iterations=15)

        end_time = time.time()
        logger.info(f"推理用时: {end_time - start_time:.4f} 秒")

        file_path1 = f"data/newproblemfacts2.json"
        with open(file_path1, "w", encoding="utf-8") as f:
            for item in problem.facts:
                f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
        logger.info(f"已生成新文件: {file_path1}")

        file_path2 = f"data/newproblemoperations2.json"
        with open(file_path2, "w", encoding="utf-8") as f:
            for item in problem.operations:
                f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
        logger.info(f"已生成新文件: {file_path2}")

        file_path3 = f"data/newproblemgroups2.json"
        with open(file_path3, "w", encoding="utf-8") as f:
            for item in problem.groups:
                f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
        logger.info(f"已生成新文件: {file_path3}")

        get_h = pformat(get_hypergraph(problem, False), width=400)
        file_path4 = f"data/get_hypergraph_violence.json"
        with open(file_path4, "w", encoding="utf-8") as f:
            f.write(get_h)

        file_path6 = f"data/newproblementities2.json"
        entities = get_entities(problem)
        with open(file_path6, "w", encoding="utf-8") as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)

        hypergraph = get_hypergraph(problem)
        with open('data/hypergraph-2-solve.json', 'w', encoding='utf-8') as f:
            json.dump(hypergraph, f, ensure_ascii=False, indent=4)
        logger.info("Hypergraph 数据已保存至 data/hypergraph-2-solve.json")


if __name__ == "__main__":
    run_test()