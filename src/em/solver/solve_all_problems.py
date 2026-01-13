import copy
import json
import time
import logging
import os
from datetime import datetime
from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, draw_gc, get_hypergraph
from pprint import pformat


# --- 日志配置 ---
def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'solver_run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


logger = setup_logging()


# --- 构图函数 (增加了 pid 参数) ---
def construct_from_list(problem, construction_steps, pid):
    logger.info(f"Problem-{pid}: 开始执行 {len(construction_steps)} 步构图...")
    for i, step in enumerate(construction_steps):
        # 简单的错误捕获，防止单步构图崩溃
        try:
            success = problem.construct(step)
            if not success:
                logger.error(f"  [Problem-{pid} Step {i + 1}] 构图失败: {step}")
                return False
        except Exception as e:
            logger.error(f"  [Problem-{pid} Step {i + 1}] 构图异常: {step} | Error: {e}")
            return False

    logger.info(f"Problem-{pid}: 构图序列执行完毕。")
    hypergraph = get_hypergraph(problem)

    os.makedirs('data', exist_ok=True)

    # 文件名加上 pid
    filename = f'data/hypergraph-{pid}-raw.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hypergraph, f, ensure_ascii=False, indent=4)

    return True


# --- 2. 暴力求解器 ---
class BruteForceSolver:
    def __init__(self, problem, theorem_lib_path):
        self.problem = problem
        self.raw_json = load_json(theorem_lib_path)
        self.parsed_gdl = parse_gdl(self.raw_json)
        self.theorems = self.parsed_gdl.get('Theorems', {})
        logger.info(f"求解器初始化完成，加载了 {len(self.theorems)} 个定理。")

    def solve(self, max_iterations=20, pid="unknown"):
        logger.info(f"=== [Problem-{pid}] 开始暴力枚举求解 ===")

        number_of_iterations = 0
        fact_is_change = 1

        while fact_is_change and number_of_iterations < max_iterations:
            fact_is_change = 0
            logger.info(f"\n[Problem-{pid} Epoch {number_of_iterations + 1}] 开始扫描全库定理...")

            for name, thm_data in self.theorems.items():
                # 可以在这里排除一些不需要的定理
                if name in ():
                    continue

                flag = self.problem.apply(name)

                if flag:
                    # logger.info(f"当前可以应用定理:{name}") # 如果日志太多可以注释掉
                    fact_is_change = 1

            if fact_is_change == 0:
                logger.info(f">>> [Problem-{pid}] 不动点达成：本轮遍历所有定理后无新事实生成。")
                break

            number_of_iterations += 1

        logger.info(f"\n=== [Problem-{pid}] 求解结束 ===")
        logger.info(f"总共执行轮数: {number_of_iterations}")
        logger.info(f"最终 Facts 数量: {len(self.problem.facts)}")
        return self.problem


# --- 3. 运行测试 (主循环逻辑) ---
def run_test():
    gdl_path = '/home/lengmen/szh/PythonWorkSpace/EuclidMachine/data/new_gdl/gdl-yuchang.json'

    problem_json_path = '/home/lengmen/szh/PythonWorkSpace/EuclidMachine/data/test_problem/problem.json'

    # 1. 加载 GDL (只需加载一次原始文件，但每次通过 parse_gdl 解析给新的 problem 使用)
    try:
        raw_gdl = load_json(gdl_path)
    except FileNotFoundError:
        logger.error(f"GDL文件未找到: {gdl_path}")
        return

    # 2. 加载题目集
    try:
        problems_data = load_json(problem_json_path)
        logger.info(f"成功加载题目文件，共 {len(problems_data)} 道题。")
    except FileNotFoundError:
        logger.error(f"题目文件未找到: {problem_json_path}")
        return

    # 确保输出目录存在
    os.makedirs('data', exist_ok=True)

    # 3. 循环遍历每一道题
    # problems_data 是字典，key是 "1", "2"... value是内容
    for pid, p_data in problems_data.items():
        logger.info("=" * 60)
        logger.info(f"正在处理题目 ID: {pid} | 描述: {p_data.get('descriptions', '')}")
        logger.info("=" * 60)

        # 每次循环都要重新解析 GDL 也就是创建一个新的 GeometricConfiguration 环境
        # 避免上一题的状态(facts/points)污染这一题
        parsed_gdl = parse_gdl(raw_gdl)
        problem = GeometricConfiguration(parsed_gdl)

        constructions = p_data.get('constructions', [])

        if not constructions:
            logger.warning(f"题目 {pid} 没有构图步骤，跳过。")
            continue

        # 执行构图
        # 传入 pid 以区分保存的文件名
        if construct_from_list(problem, constructions, pid):

            # 初始化求解器
            solver = BruteForceSolver(problem, gdl_path)

            # 绘制初始构图
            draw_gc(problem, f'data/problem-{pid}.png')

            start_time = time.time()

            # 执行求解
            solver.solve(max_iterations=15, pid=pid)

            end_time = time.time()
            logger.info(f"题目 {pid} 推理用时: {end_time - start_time:.4f} 秒")

            # 保存结果 Hypergraph
            hypergraph = get_hypergraph(problem)
            result_file = f'data/hypergraph-{pid}-solve.json'
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(hypergraph, f, ensure_ascii=False, indent=4)
            logger.info(f"Hypergraph 数据已保存至 {result_file}")

        else:
            logger.error(f"题目 {pid} 构图失败，跳过求解步骤。")


if __name__ == "__main__":
    run_test()