import json
import random
import time
from collections import defaultdict, deque
from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, draw_gc, get_hypergraph, show_gc
from pprint import pformat

def construct_from_list(problem, construction_steps):
    """
    遍历列表，逐条执行构图语句。
    :param problem: GeometricConfiguration 对象
    :param construction_steps: 包含构图语句的字符串列表
    """
    print(f"开始执行 {len(construction_steps)} 步构图...")

    for i, step in enumerate(construction_steps):
        # 执行单步构图
        success = problem.construct(step)

        if success:
            print(f"  [Step {i + 1}/{len(construction_steps)}] 成功: {step}")
        else:
            print(f"  [Step {i + 1}/{len(construction_steps)}] 失败: {step}")
            # 如果某一步失败（例如无解），通常后面的步骤也无法继续，建议中断
            print(">>> 构图中断：无法满足约束或解不存在。")
            return False

    print("构图序列执行完毕。")
    return True


def test_problem_dynamic():
    # 1. 初始化空问题
    gdl_path = 'E:\PythonWorkSpace\EuclidMachine\data\\new_gdl\gdl-yuchang.json'
    problem = GeometricConfiguration(parse_gdl(load_json(gdl_path)))

    # 2. 定义构图序列
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

    # 3. 调用循环函数执行构图
    success = construct_from_list(problem, constructions)

    if success:
        print(f"问题初始状态的facts长度: {len(problem.facts)}")
        draw_gc(problem, 'data/output-1.png')

        hypergraph = get_hypergraph(problem, serialize=False)
        with open('data/hypergraph-1-raw.json', 'w', encoding='utf-8') as f:
            json.dump(hypergraph, f, ensure_ascii=False, indent=4)

    return problem


class GoalFreeSolver:
    def __init__(self, problem, theorem_lib_path):
        """
        初始化求解器
        :param problem: GeometricConfiguration对象
        :param theorem_lib_path: 定理库的本地路径 (.json)
        """
        self.problem = problem
        self.raw_json = load_json(theorem_lib_path)
        self.parsed_gdl = parse_gdl(self.raw_json)

        # --- 核心改进：构建索引 ---
        self.premise_index = defaultdict(set)  # { 'Parallel': {'theorem_1', 'theorem_5'}, ... }
        self.conclusion_index = defaultdict(set)  # { 'theorem_1': {'EqualAngle', ...}, ... }

        # 待处理的谓词队列 (Agenda)
        self.pending_predicates = set()

        # 构建索引并初始化队列
        self._build_indices()

    def _get_predicate_name(self, logic_form_str):
        """
        辅助函数：从 'Parallel(l,k)' 或 'Eq(Sub(...))' 中提取谓词名
        """
        if not logic_form_str:
            return ""
        # 截取第一个 '(' 之前的部分
        if '(' in logic_form_str:
            return logic_form_str.split('(')[0].strip()
        return logic_form_str.strip()

    def _build_indices(self):
        """
        根据 parsed_gdl['Theorems'] 构建倒排索引
        适配 'gpl' 结构 (替代原有的 premises 和 ee_checks)
        """
        print("正在构建定理索引 (Source: Parsed GDL - GPL Mode)...")

        theorems = self.parsed_gdl.get('Theorems', {})

        for thm_name, thm_data in theorems.items():
            # -------------------------------------------------------
            # 1. 处理 GPL (Geometric Premise List) - 这是新的前提源
            # -------------------------------------------------------
            # gpl 是一个元组，包含多个字典
            gpl_entries = thm_data.get('gpl', ())

            for entry in gpl_entries:
                # 获取 product 字段: ('PointOnLine', ('A', 'l'), ...)
                product = entry.get('product')

                # 确保 product 存在且非空
                if product and len(product) > 0:
                    # 第一个元素即为谓词名称，如 'PointOnLine'
                    pred_name = product[0]

                    # 放入索引
                    # 注意：这里不需要 _get_predicate_name 清洗，因为 product[0] 本身就是纯字符串
                    if isinstance(pred_name, str):
                        self.premise_index[pred_name].add(thm_name)

            # -------------------------------------------------------
            # 2. 处理 结论 (Conclusions) - 依然存在
            # -------------------------------------------------------
            # 结构示例: (('PointOnLine', ('C', 'l')),)
            conc_list = thm_data.get('conclusions', ())

            for c_item in conc_list:
                # c_item 是一个元组 ('Predicate', args...)
                if c_item and len(c_item) > 0:
                    pred_name = c_item[0]

                    if isinstance(pred_name, str):
                        self.conclusion_index[thm_name].add(pred_name)

        print(f"索引构建完成，共处理 {len(theorems)} 个定理。")

    def _init_pending_predicates(self):
        """
        根据 problem 当前的状态初始化谓词队列。
        兼容 'PointOnLine(A,l)' (实例) 和 'PointOnLine' (类型) 两种键格式。
        """
        try:
            # 获取 Configuration 对象内部的 parsed_gdl
            gdl = self.problem.parsed_gdl
            found_any = False

            # 我们需要扫描 Relations (几何关系) 和 Entities (基本实体)
            # Entities 对应 ee_check 索引的定理
            # Relations 对应 premise 索引的定理
            target_sections = ['Relations', 'Entities']

            for section in target_sections:
                if section in gdl and isinstance(gdl[section], dict):
                    # 遍历字典的键
                    for key in gdl[section].keys():
                        # 情况 A: 键是实例字符串 "PointOnLine(A,l)"
                        pred_name = self._get_predicate_name(key)

                        # 情况 B: 键本身就是谓词名 "PointOnLine" (防止直接读取了定义)
                        if not pred_name and key.isalnum():
                            pred_name = key

                        if pred_name:
                            self.pending_predicates.add(pred_name)
                            found_any = True

            # 兜底：如果初始化队列为空，回退到全量扫描模式
            if not found_any:
                print("Warning: 未能提取初始谓词，将执行全量扫描...")
                for pred in self.premise_index.keys():
                    self.pending_predicates.add(pred)
            else:
                # 打印激活的谓词用于调试
                display_preds = list(self.pending_predicates)[:]
                print(f"初始化完成，初始激活 {len(self.pending_predicates)} 类谓词: {display_preds}...")

        except Exception as e:
            print(f"Error during init: {e}")
            # 出错时回退
            for pred in self.premise_index.keys():
                self.pending_predicates.add(pred)

    def _refresh_predicates_from_facts(self):
        """
        从 problem.facts (动态事实库) 中提取当前所有存在的谓词类型。
        这确保了下一轮 Epoch 能利用上一轮生成的新知识。
        """
        # 清空旧队列，准备全量重新加载
        self.pending_predicates.clear()

        try:
            # problem.facts 是一个列表，每个元素是一个 fact
            # fact 的结构是 tuple/list，第0个元素是 predicate (str)
            # 例如: ('PointOnLine', 'A', ...)
            if hasattr(self.problem, 'facts'):
                for fact in self.problem.facts:
                    # 确保 fact 是可迭代的且有内容
                    if fact and len(fact) > 0:
                        pred_name = fact[0]
                        self.pending_predicates.add(pred_name)

            # 同时也扫描一下 Entities (为了那些依赖 ee_check 的基础定理)
            # 虽然 facts 里通常包含了实体定义，但为了保险起见保留这个
            if hasattr(self.problem, 'parsed_gdl') and 'Entities' in self.problem.parsed_gdl:
                for entity_type in self.problem.parsed_gdl['Entities'].keys():
                    self.pending_predicates.add(entity_type)

        except Exception as e:
            print(f"Warning: 从 facts 刷新谓词失败: {e}")
            # 失败时的兜底：还是读初始 GDL
            self._init_pending_predicates()

    def solve(self, max_epochs=10, max_steps_per_epoch=500):
        """
        改进后的求解过程：循环直到达到不动点（事实不再增加）。
        :param max_epochs: 最大循环轮数
        :param max_steps_per_epoch: 每一轮最大的推理步数
        """
        print("=== 开始数据驱动 Goal-free 求解 (Fixed-Point Mode) ===")

        total_effective_steps = 0

        for epoch in range(max_epochs):
            # 1. 记录本轮开始前的事实数量
            # problem 对象有一个 facts 列表属性
            start_fact_count = len(self.problem.facts)
            print(f"\n--- Epoch {epoch + 1} (当前事实数: {start_fact_count}) ---")

            # 每一轮开始前，强制刷新状态。
            # 将当前 facts 中所有的谓词（无论是初始的还是上一轮新生成的）都加入队列。
            # 这保证了 Epoch 2 能看到 Epoch 1 产生的所有新关系。
            self._refresh_predicates_from_facts()

            steps = 0
            epoch_effective = 0

            # 3. 执行单轮推理 (Inner Loop)
            while self.pending_predicates and steps < max_steps_per_epoch:
                steps += 1
                current_pred = self.pending_predicates.pop()

                # 查找相关定理
                candidate_theorems = list(self.premise_index.get(current_pred, []))

                if not candidate_theorems:
                    continue

                # 尝试应用
                for thm_name in candidate_theorems:
                    try:
                        # 直接把 Key 传给 apply
                        success = self.problem.apply(thm_name)

                        if success:
                            print(f'在第{steps}步成功应用定理{thm_name}')
                            epoch_effective += 1
                            total_effective_steps += 1

                            # 触发新结论的更新
                            output_preds = self.conclusion_index.get(thm_name, [])
                            for out_p in output_preds:
                                self.pending_predicates.add(out_p)

                    except Exception as e:
                        # 某些特定定理应用时可能会抛错，捕获它以保证循环继续
                        print(f"Error applying {thm_name}: {e}")
                        pass

            # 4. 检查是否达到不动点 (Termination Check)
            end_fact_count = len(self.problem.facts)
            fact_diff = end_fact_count - start_fact_count

            print(f"Epoch {epoch + 1} 结束: 执行 {steps} 步，新增 {fact_diff} 条事实。")

            if fact_diff == 0:
                print(">>> 系统达到稳定状态 (Fixed Point)，无新事实生成，停止求解。")
                break
            else:
                print(">>> 检测到事实扩充，将进入下一轮 Epoch 继续尝试...")

        print(f"\n=== 求解彻底结束 ===")
        print(f"最终事实数量: {len(self.problem.facts)}")
        print(f"总计有效推导步数: {total_effective_steps}")

        return self.problem


if __name__ == "__main__":
    # 1. 定理库路径
    lib_path = "E:\PythonWorkSpace\EuclidMachine\data\\new_gdl\gdl-yuchang.json"

    # 2. 题目的初始定义 (Construction steps)
    problem = test_problem_dynamic()
    # show_gc(problem)
    start = time.time()
    # 3. 初始化求解器
    solver = GoalFreeSolver(problem, lib_path)

    # 4. 运行 Goal-free 求解
    problem = solver.solve()

    end = time.time()
    print(f"推理时间:{end - start}")

    # show_gc(problem)
    # 获取超图
    hypergraph = get_hypergraph(problem)

    file_path1 = f"data/newproblemfacts_num.json"
    with open(file_path1, "w", encoding="utf-8") as f:
        for item in problem.facts:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path1}")

    file_path2 = f"data/newproblemoperations_num.json"
    with open(file_path2, "w", encoding="utf-8") as f:
        for item in problem.operations:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path2}")

    file_path3 = f"data/newproblemgroups_num.json"
    with open(file_path3, "w", encoding="utf-8") as f:
        for item in problem.groups:
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")
    print(f"已生成新文件: {file_path3}")

    get_h = pformat(get_hypergraph(problem, False), width=400)
    file_path4 = f"data/get_hypergraph_num.json"
    with open(file_path4, "w", encoding="utf-8") as f:
        f.write(get_h)

    # 将字典写入JSON文件
    with open('data/hypergraph-1-solve.json', 'w', encoding='utf-8') as f:
        json.dump(hypergraph, f, ensure_ascii=False, indent=4)

    # 获取序列化的超图
    hypergraph_serialize = get_hypergraph(problem, serialize=True)
    with open('data/hypergraph-1-serialize.json', 'w', encoding='utf-8') as f:
        json.dump(hypergraph_serialize, f, ensure_ascii=False, indent=4)