import json
import os
import re  # 引入正则模块用于提取ID
from collections import deque, defaultdict
import random

"""
    当前代码用来批量生成所有题目的定理预测数据
"""

# --- 配置区域 ---

# 1. 动作过滤
CONSTRUCTION_OPS = {
    "Point", "FreePoint", "Line", "FreeLine", "Circle", "FreeCircle",
    "PointOnLine", "PointOnCircle", "ClockwiseTriangle", "PointLeftSegment",
    "Equation"
}
LOGIC_ARTIFACTS = {"auto_extend", "multiple_forms"}
SKIPPED_PREDICTION_OPS = CONSTRUCTION_OPS | LOGIC_ARTIFACTS

# 2. 目标过滤
IGNORED_GOAL_TYPES = {
    "PointLeftSegment",
    "ClockwiseTriangle",
    "Equation",
}

# 3. 状态过滤
IGNORED_STATE_TYPES = {
    "FreePoint",
    "PointLeftSegment",
    "Equation",
    "ClockwiseTriangle"
}


def format_fact(fact_list):
    """格式化事实文本"""
    if not fact_list: return ""
    pred = fact_list[0]
    args = fact_list[1:]
    clean_args = [str(a).replace("'", "") for a in args]
    if len(clean_args) > 0:
        return f"{pred}({', '.join(clean_args)})"
    return pred


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_indices(hypergraph):
    """构建索引"""
    fact_producer = {}
    fact_consumers = defaultdict(set)
    all_produced_facts = set()

    for step_idx, (premise_ids, edge_id, conclusion_ids) in enumerate(hypergraph):
        for pid in premise_ids:
            fact_consumers[pid].add(step_idx)
        for cid in conclusion_ids:
            fact_producer[cid] = step_idx
            all_produced_facts.add(cid)

    return fact_producer, fact_consumers, all_produced_facts


def get_minimal_dependency_subgraph(target_fid, fact_producer, hypergraph_data):
    """获取最小依赖子图"""
    queue = deque([target_fid])
    involved_facts = {target_fid}
    involved_steps = set()

    while queue:
        current_fid = queue.popleft()
        producer_step_idx = fact_producer.get(current_fid)

        # 如果没有生产者，或者是初始条件，跳过
        if producer_step_idx is None:
            continue

        if producer_step_idx in involved_steps: continue
        involved_steps.add(producer_step_idx)
        premise_ids = hypergraph_data[producer_step_idx][0]
        for pid in premise_ids:
            if pid not in involved_facts:
                involved_facts.add(pid)
                queue.append(pid)
    return involved_steps, involved_facts


def get_multiple_topological_sorts(involved_steps, hypergraph_data, fact_producer, num_sequences=3):
    """生成多个有效的拓扑排序序列"""
    local_adj = defaultdict(list)
    base_in_degree = {step: 0 for step in involved_steps}
    step_list = list(involved_steps)

    for consumer_step in step_list:
        premises = hypergraph_data[consumer_step][0]
        for pid in premises:
            if pid in fact_producer:
                producer_step = fact_producer[pid]
                if producer_step in involved_steps:
                    local_adj[producer_step].append(consumer_step)
                    base_in_degree[consumer_step] += 1

    unique_sequences = set()
    results = []
    max_attempts = num_sequences * 10
    attempts = 0

    while len(results) < num_sequences and attempts < max_attempts:
        attempts += 1
        current_in_degree = base_in_degree.copy()
        candidates = [s for s in involved_steps if current_in_degree[s] == 0]
        current_sequence = []

        while candidates:
            random.shuffle(candidates)
            curr = candidates.pop(0)
            current_sequence.append(curr)

            for neighbor in local_adj[curr]:
                current_in_degree[neighbor] -= 1
                if current_in_degree[neighbor] == 0:
                    candidates.append(neighbor)

        if len(current_sequence) == len(involved_steps):
            seq_tuple = tuple(current_sequence)
            if seq_tuple not in unique_sequences:
                unique_sequences.add(seq_tuple)
                results.append(current_sequence)

    return results


def topological_sort_steps(involved_steps, hypergraph_data, fact_producer):
    """单次拓扑排序"""
    local_adj = defaultdict(list)
    in_degree = {step: 0 for step in involved_steps}
    step_list = list(involved_steps)

    for consumer_step in step_list:
        premises = hypergraph_data[consumer_step][0]
        for pid in premises:
            if pid in fact_producer:
                producer_step = fact_producer[pid]
                if producer_step in involved_steps:
                    local_adj[producer_step].append(consumer_step)
                    in_degree[consumer_step] += 1

    queue = deque([s for s in involved_steps if in_degree[s] == 0])
    sorted_sequence = []

    while queue:
        curr = queue.popleft()
        sorted_sequence.append(curr)
        for neighbor in local_adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return sorted_sequence


def generate_theorem_dataset(file_path, output_path, index, pid="unknown"):
    print(f"[{pid}] 开始生成标准数据集...")
    data = load_data(file_path)
    hypergraph = data['hypergraph']
    edges = data['edges']
    notes = data['notes']

    fact_producer, fact_consumers, all_produced_facts = build_indices(hypergraph)

    dataset = []
    processed_goals = 0

    leaf_facts = []
    for fid in all_produced_facts:
        if fid < index: continue

        fact_content = notes[fid]
        predicate = fact_content[0]
        if predicate in IGNORED_GOAL_TYPES: continue

        if fid not in fact_consumers:
            leaf_facts.append(fid)
            continue

        consumer_steps = fact_consumers[fid]
        is_consumed_by_real_theorem = False
        for step_idx in consumer_steps:
            edge_id = hypergraph[step_idx][1]
            op_name = edges[edge_id][0]
            if op_name not in SKIPPED_PREDICTION_OPS:
                is_consumed_by_real_theorem = True
                break

        if not is_consumed_by_real_theorem:
            leaf_facts.append(fid)

    print(f"[{pid}] 过滤后找到 {len(leaf_facts)} 个有效目标。")

    for goal_fid in leaf_facts:
        involved_steps, involved_facts = get_minimal_dependency_subgraph(goal_fid, fact_producer, hypergraph)
        execution_sequence = topological_sort_steps(involved_steps, hypergraph, fact_producer)

        current_state_ids = set()
        for step_idx in involved_steps:
            premises = hypergraph[step_idx][0]
            for pid_fact in premises:
                prod_step = fact_producer.get(pid_fact)
                if prod_step is None or prod_step not in involved_steps:
                    current_state_ids.add(pid_fact)

        goal_text = format_fact(notes[goal_fid])

        for i, step_idx in enumerate(execution_sequence):
            edge_id = hypergraph[step_idx][1]
            op_info = edges[edge_id]
            op_name = op_info[0]
            conclusions = hypergraph[step_idx][2]

            should_predict = op_name not in SKIPPED_PREDICTION_OPS

            if should_predict:
                state_text_list = []
                for fid in current_state_ids:
                    fact_data = notes[fid]
                    predicate = fact_data[0]
                    if predicate not in IGNORED_STATE_TYPES:
                        state_text_list.append(format_fact(fact_data))

                sample = {
                    "problem_id": pid,  # 增加题目ID标识
                    "goal": goal_text,
                    "current_state": state_text_list,
                    "target_action": op_name,
                    "当前状态和动作依赖的拓扑排序序列": execution_sequence[0: i + 1],
                    "当前叶子结点完整的拓扑排序序列": execution_sequence
                }
                dataset.append(sample)

            for cid in conclusions:
                current_state_ids.add(cid)

        processed_goals += 1

    print(f"[{pid}] 处理完成。共生成 {len(dataset)} 条有效训练样本。")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


def generate_theorem_dataset_multiple(file_path, output_path, index, pid="unknown"):
    print(f"[{pid}] 开始生成多路径增强数据集...")
    data = load_data(file_path)
    hypergraph = data['hypergraph']
    edges = data['edges']
    notes = data['notes']

    fact_producer, fact_consumers, all_produced_facts = build_indices(hypergraph)

    dataset = []
    processed_goals = 0

    leaf_facts = []
    for fid in all_produced_facts:
        if fid < index: continue
        fact_content = notes[fid]
        predicate = fact_content[0]
        if predicate in IGNORED_GOAL_TYPES: continue
        if fid not in fact_consumers:
            leaf_facts.append(fid)
            continue
        consumer_steps = fact_consumers[fid]
        is_consumed_by_real_theorem = False
        for step_idx in consumer_steps:
            edge_id = hypergraph[step_idx][1]
            op_name = edges[edge_id][0]
            if op_name not in SKIPPED_PREDICTION_OPS:
                is_consumed_by_real_theorem = True
                break
        if not is_consumed_by_real_theorem:
            leaf_facts.append(fid)

    for goal_fid in leaf_facts:
        involved_steps, involved_facts = get_minimal_dependency_subgraph(goal_fid, fact_producer, hypergraph)
        execution_sequences_list = get_multiple_topological_sorts(
            involved_steps, hypergraph, fact_producer, num_sequences=3
        )

        goal_text = format_fact(notes[goal_fid])

        for seq_idx, execution_sequence in enumerate(execution_sequences_list):
            current_state_ids = set()
            for step_idx in involved_steps:
                premises = hypergraph[step_idx][0]
                for pid_fact in premises:
                    prod_step = fact_producer.get(pid_fact)
                    if prod_step is None or prod_step not in involved_steps:
                        current_state_ids.add(pid_fact)

            for i, step_idx in enumerate(execution_sequence):
                edge_id = hypergraph[step_idx][1]
                op_info = edges[edge_id]
                op_name = op_info[0]
                conclusions = hypergraph[step_idx][2]
                should_predict = op_name not in SKIPPED_PREDICTION_OPS

                if should_predict:
                    state_text_list = []
                    for fid in current_state_ids:
                        fact_data = notes[fid]
                        predicate = fact_data[0]
                        if predicate not in IGNORED_STATE_TYPES:
                            state_text_list.append(format_fact(fact_data))

                    sample = {
                        #"problem_id": pid,
                        "goal": goal_text,
                        "current_state": state_text_list,
                        "target_action": op_name,
                        # "当前状态和动作依赖的拓扑排序序列": execution_sequence[0: i + 1],
                        # "当前叶子结点完整的拓扑排序序列": execution_sequence,
                        # "seq_id": f"{goal_fid}_{seq_idx}",
                    }
                    dataset.append(sample)

                for cid in conclusions:
                    current_state_ids.add(cid)

        processed_goals += 1

    print(f"[{pid}] 多路径增强处理完成。共生成 {len(dataset)} 条样本。")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


# --- 主循环逻辑 ---

def process_all_problems():
    data_dir = 'data'
    output_dir = 'tp_data'

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 获取 data 目录下所有文件
    try:
        all_files = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"错误: 找不到数据目录 '{data_dir}'")
        return

    # 正则表达式匹配 hypergraph-{id}-solve.json
    # 假设 id 可能是数字
    solve_file_pattern = re.compile(r'^hypergraph-(\d+)-solve\.json$')

    # 筛选出所有 solve 文件并按 ID 排序 (为了日志好看)
    solve_files = []
    for f in all_files:
        match = solve_file_pattern.match(f)
        if match:
            solve_files.append((match.group(1), f))

    # 按 ID 数字大小排序
    solve_files.sort(key=lambda x: int(x[0]))

    if not solve_files:
        print("未在 data 目录下找到任何 hypergraph-*-solve.json 文件。")
        return

    print(f"找到 {len(solve_files)} 个待处理的题目文件。")

    for pid, solve_filename in solve_files:
        print("-" * 50)
        print(f"正在处理题目 ID: {pid}")

        solve_path = os.path.join(data_dir, solve_filename)

        # 构造对应的 raw 文件名
        raw_filename = f"hypergraph-{pid}-raw.json"
        raw_path = os.path.join(data_dir, raw_filename)

        # 检查 raw 文件是否存在
        if not os.path.exists(raw_path):
            print(f"  [警告] 找不到对应的原始文件: {raw_filename}，跳过此题。")
            continue

        try:
            # 1. 加载 Raw 数据获取构图长度
            raw_data = load_data(raw_path)
            constructions_len = len(raw_data['notes'])
            print(f"  构图步骤 Fact 数量: {constructions_len}")

            # 2. 生成标准数据集
            output_single = os.path.join(output_dir, f'problem_{pid}.json')
            generate_theorem_dataset(solve_path, output_single, constructions_len, pid)

            # 3. 生成多路径数据集
            output_multi = os.path.join(output_dir, f'problem_{pid}_multiple.json')
            generate_theorem_dataset_multiple(solve_path, output_multi, constructions_len, pid)

        except Exception as e:
            print(f"  [错误] 处理题目 {pid} 时发生异常: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 50)
    print("所有题目批量处理完毕。")


if __name__ == "__main__":
    process_all_problems()