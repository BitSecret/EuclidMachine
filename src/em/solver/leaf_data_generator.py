import json
from collections import deque, defaultdict
import random

"""
    当前代码用来生成每一道题目（经过推理之后收敛的超图）的叶子结点的定理预测数据
"""

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
}

# 3. 状态过滤：这些类型的谓词永远不出现在 current_state 中
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
    fact_producer = {}  # fact_id -> step_idx(hypergraph, 记录了每一个事实是由哪一个推理步骤产生的)
    fact_consumers = defaultdict(set) # fact_id -> {step_idx_1, step_idx_2, ...}(记录每一个事实被哪些推理步骤作为前提条件使用了)
    all_produced_facts = set() # 记录了所有通过推理步骤新生成的事实 ID

    for step_idx, (premise_ids, edge_id, conclusion_ids) in enumerate(hypergraph):
        for pid in premise_ids:
            fact_consumers[pid].add(step_idx)
        for cid in conclusion_ids:
            fact_producer[cid] = step_idx
            all_produced_facts.add(cid)

    return fact_producer, fact_consumers, all_produced_facts


def get_minimal_dependency_subgraph(target_fid, fact_producer, hypergraph_data):
    """获取最小依赖子图"""
    queue = deque([target_fid]) # 为了得到结论，必须执行的步骤集合。
    involved_facts = {target_fid} # 涉及到的所有事实（包括中间结论和已知条件）
    involved_steps = set()

    while queue:
        current_fid = queue.popleft()
        # if current_fid not in fact_producer: continue # 如果这个事实没有生产者（即不在 fact_producer 中）,说明它是题目一开始给的“已知条件”(Given)，不需要再往上找步骤了。
        producer_step_idx = fact_producer[current_fid]
        if producer_step_idx in involved_steps: continue
        involved_steps.add(producer_step_idx)
        premise_ids = hypergraph_data[producer_step_idx][0]
        for pid in premise_ids:
            if pid not in involved_facts:
                involved_facts.add(pid)
                queue.append(pid)
    return involved_steps, involved_facts

def get_multiple_topological_sorts(involved_steps, hypergraph_data, fact_producer, num_sequences=3):
    """
    生成多个有效的拓扑排序序列。
    原理：Kahn算法。在每一轮选择入度为0的节点时，不再使用固定的队列顺序，而是随机选择。
    """

    # --- 阶段 1：构建依赖图 (这部分是静态的，只用做一次) ---
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

    # --- 阶段 2：随机生成多条路径 ---
    unique_sequences = set()
    results = []

    # 为了防止死循环（例如图本身只有一个唯一解），设置最大尝试次数
    max_attempts = num_sequences * 10
    attempts = 0

    while len(results) < num_sequences and attempts < max_attempts:
        attempts += 1

        # 必须拷贝一份入度表，因为每次排序都会修改它
        current_in_degree = base_in_degree.copy()

        # 初始候选集：所有入度为0的节点
        candidates = [s for s in involved_steps if current_in_degree[s] == 0]

        current_sequence = []

        # 开始随机Kahn算法
        while candidates:
            # 【核心修改】：随机打乱候选集，或者随机选一个
            # 这模拟了“如果我有3个独立的定理可以用，我随机选一个先用”
            random.shuffle(candidates)

            # 取出一个执行
            curr = candidates.pop(0)
            current_sequence.append(curr)

            # 解锁下游
            for neighbor in local_adj[curr]:
                current_in_degree[neighbor] -= 1
                if current_in_degree[neighbor] == 0:
                    candidates.append(neighbor)

        # 验证生成的序列长度是否完整 (防止环或其他异常)
        if len(current_sequence) == len(involved_steps):
            # 转成 tuple 放入 set 去重
            seq_tuple = tuple(current_sequence)
            if seq_tuple not in unique_sequences:
                unique_sequences.add(seq_tuple)
                results.append(current_sequence)

    return results

def topological_sort_steps(involved_steps, hypergraph_data, fact_producer):
    """拓扑排序"""
    # --- 第一阶段：构建局部依赖图 ---
    local_adj = defaultdict(list) # 邻接表：记录 "步骤A -> 步骤B" 的指向
    in_degree = {step: 0 for step in involved_steps} # 入度表：记录每个步骤还缺几个前置步骤
    step_list = list(involved_steps)
    # 遍历每一个“消费者步骤” (Consumer)
    for consumer_step in step_list:
        # 看看这个步骤需要哪些原料 (Premises)
        premises = hypergraph_data[consumer_step][0]
        for pid in premises:
            # 查找这个原料是谁生产的 (Producer)
            if pid in fact_producer:
                producer_step = fact_producer[pid]
                # 【关键判断】：只有当生产者也在 involved_steps 里时，才算有效依赖。
                # 为什么？因为如果 producer_step 不在 involved_steps 里，
                # 说明 pid 是个"已知条件"或者"不需要证明的分支"，
                # 对于局部排序来说，pid 视为已经存在，不需要等待生产者。
                if producer_step in involved_steps:
                    # 建立关系：生产者 -> 消费者
                    local_adj[producer_step].append(consumer_step)
                    # 消费者的等待数 +1
                    in_degree[consumer_step] += 1

    # --- 第二阶段：执行排序 (Kahn算法) ---
    # 1. 寻找起点：所有入度为 0 的步骤
    # 这些步骤只依赖“题目已知条件”，不依赖“involved_steps 中的其他步骤”
    queue = deque([s for s in involved_steps if in_degree[s] == 0])
    sorted_sequence = [] # 拓扑排序的返回结果{存储的是involved_steps中的内容}, 当前算法只有一种结果
    while queue:
        # 2. 取出一个可以执行的步骤
        curr = queue.popleft()
        sorted_sequence.append(curr)
        # 3. 通知下游：你的一个前置条件已经搞定了
        for neighbor in local_adj[curr]:
            in_degree[neighbor] -= 1
            # 4. 如果下游步骤的所有前置条件都搞定了 (入度变0)，把它加入队列
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
        # A. 基础过滤：索引检查, 过滤构图过程中的产生的fact
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
                    # 如果该谓词在黑名单中，就不加入 state_text_list
                    if predicate not in IGNORED_STATE_TYPES:
                        state_text_list.append(format_fact(fact_data))
                # state_text_list.sort() # 可选排序

                sample = {
                    "goal": goal_text,
                    "current_state": state_text_list,
                    "target_action": op_name,
                    "当前状态和动作依赖的拓扑排序序列":execution_sequence[0: i+1],
                    "当前叶子结点完整的拓扑排序序列":execution_sequence # 序号对应超图中的边
                }
                dataset.append(sample)

            for cid in conclusions:
                current_state_ids.add(cid)

        processed_goals += 1

    print(f"处理完成。共生成 {len(dataset)} 条有效训练样本。")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

def generate_theorem_dataset_multiple(file_path, output_path, index):
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
        # A. 基础过滤：索引检查, 过滤构图过程中的产生的fact
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

        # 【修改点 1】：获取多条路径 (例如设定为 3 种)
        # 注意：如果子图很简单（线性依赖），可能实际只能返回 1 条，这是正常的
        execution_sequences_list = get_multiple_topological_sorts(
            involved_steps, hypergraph, fact_producer, num_sequences=3
        )

        goal_text = format_fact(notes[goal_fid])

        # 【修改点 2】：增加一层循环，遍历每一条生成的路径
        for seq_idx, execution_sequence in enumerate(execution_sequences_list):

            # 每个序列都需要从头初始化状态，不能共用
            current_state_ids = set()
            for step_idx in involved_steps:
                premises = hypergraph[step_idx][0]
                for pid in premises:
                    prod_step = fact_producer.get(pid)
                    if prod_step is None or prod_step not in involved_steps:
                        current_state_ids.add(pid)

            # 开始按照这一条特定的随机路径生成样本
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
                        "goal": goal_text,
                        "current_state": state_text_list,
                        "target_action": op_name,
                        "当前状态和动作依赖的拓扑排序序列": execution_sequence[0: i + 1],
                        "当前叶子结点完整的拓扑排序序列": execution_sequence,  # 序号对应超图中的边
                        "seq_id": f"{goal_fid}_{seq_idx}", # 为了调试方便,可以记录一下是第几种变体
                    }
                    dataset.append(sample)

                for cid in conclusions:
                    current_state_ids.add(cid)

        processed_goals += 1

    print(f"处理完成。共生成 {len(dataset)} 条有效训练样本。")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # 这里的 index 参数是 推理过程 开始的第一个 ID
    # 如果要过滤掉所有构图步骤, constrcutions_len 是个不错的切分点
    raw_data = load_data('data/hypergraph-1-raw.json')  # 加载原始数据以获取构图长度
    constructions_len = len(raw_data['notes'])

    generate_theorem_dataset(
        'data/hypergraph-1-solve_add_aux.json',
        'tp_data/problem_1_add_aux.json',
        constructions_len
    )

    generate_theorem_dataset_multiple(
        'data/hypergraph-1-solve_add_aux.json',
        'tp_data/problem_1_multiple_add_aux.json',
        constructions_len
    )