import json
from collections import deque, defaultdict
# --- 配置区域 ---

# 1. 动作过滤：这些算子会被执行以更新状态，但不会作为 target_action 被预测
CONSTRUCTION_OPS = {
    "Point", "FreePoint", "Line", "FreeLine", "Circle", "FreeCircle",
    "PointOnLine", "PointOnCircle", "ClockwiseTriangle", "PointLeftSegment",
    "Equation"
}
LOGIC_ARTIFACTS = {"auto_extend", "multiple_forms"}
SKIPPED_PREDICTION_OPS = CONSTRUCTION_OPS | LOGIC_ARTIFACTS

# 2. 目标过滤：这些类型的结论即使是叶子节点，也不作为 Goal (不训练模型去证明这些)
IGNORED_GOAL_TYPES = {
    "PointLeftSegment",
    "ClockwiseTriangle",
    "Equation",  # 数值计算通常不作为纯几何证明的目标
    # 可以在这里添加更多你想忽略的类型，例如 "Colliders" 等
}

# 3. 状态过滤（新增）：这些类型的谓词永远不出现在 current_state 中
IGNORED_STATE_TYPES = {
    "FreePoint",
    "PointLeftSegment",
    "Equation"
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
        if current_fid not in fact_producer: continue
        producer_step_idx = fact_producer[current_fid]
        if producer_step_idx in involved_steps: continue
        involved_steps.add(producer_step_idx)
        premise_ids = hypergraph_data[producer_step_idx][0]
        for pid in premise_ids:
            if pid not in involved_facts:
                involved_facts.add(pid)
                queue.append(pid)
    return involved_steps, involved_facts


def topological_sort_steps(involved_steps, hypergraph_data, fact_producer):
    """拓扑排序"""
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


def generate_theorem_dataset(file_path, output_path, index):
    data = load_data(file_path)
    hypergraph = data['hypergraph']
    edges = data['edges']
    notes = data['notes']

    fact_producer, fact_consumers, all_produced_facts = build_indices(hypergraph)

    dataset = []
    processed_goals = 0

    # --- 1. 识别目标 (增加了 IGNORED_GOAL_TYPES 过滤) ---
    leaf_facts = []
    for fid in all_produced_facts:
        # A. 基础过滤：索引检查
        if fid < index:
            continue

        # B. 类型过滤：检查谓词是否在黑名单中
        fact_content = notes[fid]
        predicate = fact_content[0]
        if predicate in IGNORED_GOAL_TYPES:
            continue

        # C. 拓扑过滤：检查是否为“逻辑上的叶子”
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

    print(f"过滤后找到 {len(leaf_facts)} 个有效目标 (已排除 {IGNORED_GOAL_TYPES})。")

    # --- 2. 生成数据 ---
    for goal_fid in leaf_facts:
        involved_steps, involved_facts = get_minimal_dependency_subgraph(goal_fid, fact_producer, hypergraph)
        execution_sequence = topological_sort_steps(involved_steps, hypergraph, fact_producer)

        # 初始化状态
        current_state_ids = set()
        for step_idx in involved_steps:
            premises = hypergraph[step_idx][0]
            for pid in premises:
                prod_step = fact_producer.get(pid)
                if prod_step is None or prod_step not in involved_steps:
                    current_state_ids.add(pid)

        goal_text = format_fact(notes[goal_fid])

        for step_idx in execution_sequence:
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
                    # 如果该谓词在黑名单中，就不加入 state_text_list
                    if predicate not in IGNORED_STATE_TYPES:
                        state_text_list.append(format_fact(fact_data))
                # state_text_list.sort() # 可选排序

                sample = {
                    "goal": goal_text,
                    "current_state": state_text_list,
                    "target_action": op_name
                }
                dataset.append(sample)

            for cid in conclusions:
                current_state_ids.add(cid)

        processed_goals += 1

    print(f"处理完成。共生成 {len(dataset)} 条有效训练样本。")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # 请确保这里的 index 参数是你 logic flow 开始的第一个 ID
    # 如果你是要过滤掉所有构图步骤，通常 constrcutions_len 是个不错的切分点
    data = load_data('data/hypergraph-1-raw.json')  # 加载原始数据以获取构图长度
    constructions_len = len(data['notes'])

    generate_theorem_dataset(
        'data/hypergraph-1-solve.json',
        'data/theorem_prediction_data-filtered.json',
        constructions_len
    )