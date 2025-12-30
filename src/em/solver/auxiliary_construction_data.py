import json
import networkx as nx
import re


# 映射回文本
def map_indices_to_lines(indices, lines, notes):
    mapped_lines = []
    sig_to_line = {}
    for line in lines:
        try:
            # 兼容带有空格的情况 Line( s )
            t = line.split('(')[0].strip()
            n = line.split('(')[1].split(')')[0].strip()
            sig_to_line[(t, n)] = line
        except:
            pass

    sorted_indices = sorted(list(indices))
    for idx in sorted_indices:
        note = notes[idx]
        sig = (note[0], note[1])
        if sig in sig_to_line:
            mapped_lines.append(sig_to_line[sig])
    return mapped_lines


def normalize_string(s):
    """去除所有空格，用于字符串比较"""
    return s.replace(" ", "")


def parse_goal_string(goal_str):
    """
    解析 "Type(P1, P2...)" 为 ("Type", ["P1", "P2"...])
    """
    # 提取类型
    pred = goal_str.split('(')[0].strip()
    # 提取括号内的内容并按逗号分割
    match = re.search(r'\((.*?)\)', goal_str)
    if match:
        params = [p.strip() for p in match.group(1).split(',')]
    else:
        params = []
    return pred, params


def generate_auxiliary_training_data(dependent_entities_json_path, proof_json_path, construction_lines, goal):
    # 1. 加载数据
    with open(dependent_entities_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    notes = data['notes']
    deps = data['dependent_entities']

    # 2. 构建依赖图 (Edge: Child -> Parent)
    G = nx.DiGraph()
    for child_idx, parents in enumerate(deps):
        for p_idx in parents:
            G.add_edge(child_idx, p_idx)

    # ==========================================
    # 3. 确定“基础构图” (Basic Construction)
    # ==========================================

    # 解析输入的目标字符串
    target_pred, target_params = parse_goal_string(goal)
    target_params_set = set(target_params)  # 转集合，忽略点顺序

    goal_node_idx = -1

    # 不仅匹配类型，还要匹配参数
    for i in range(len(notes) - 1, -1, -1):
        note = notes[i]
        # note[0] 是类型，note[1:] 是参数列表
        if note[0] == target_pred:
            # 检查参数集合是否一致（防止中间步骤同类型干扰）
            if set(note[1:]) == target_params_set:
                goal_node_idx = i
                break

    if goal_node_idx == -1:
        print(f"Error: 在超图中未找到目标节点 {goal} (类型或参数不匹配)")
        return

    # 找到这些点在 notes 中的索引（用于回溯定义）
    target_entity_indices = []
    for p_name in target_params:
        for i, note in enumerate(notes):
            if note[0] == "Point" and note[1] == p_name:
                target_entity_indices.append(i)
                break

    basic_indices = set()
    for idx in target_entity_indices:
        basic_indices.add(idx)
        basic_indices.update(nx.descendants(G, idx))

    basic_construction_indices = {
        i for i in basic_indices
        if notes[i][0] in ['Point', 'Line', 'Circle']
    }

    basic_lines = map_indices_to_lines(basic_construction_indices, construction_lines, notes)

    # ==========================================
    # 4. 确定“有效构图” (Proof/Effective Construction)
    # ==========================================

    identifier_to_full_line = {}
    for line in construction_lines:
        if ':' in line:
            identifier = line.split(':')[0].strip()
        else:
            identifier = line.strip()
        identifier_to_full_line[identifier] = line

    proof_full_lines = set()

    with open(proof_json_path, 'r', encoding='utf-8') as f:
        tp_datas = json.load(f)

    found_state = False
    normalized_input_goal = normalize_string(goal)  # 预处理输入目标

    for orig_idx in range(len(tp_datas) - 1, -1, -1):
        tp_data = tp_datas[orig_idx]

        # 【修复 2】：使用去空格后的字符串进行比较，增强鲁棒性
        if normalize_string(tp_data['goal']) == normalized_input_goal:
            current_facts = tp_data['current_state']

            for fact in current_facts:
                if any(fact.startswith(prefix) for prefix in ['Point', 'Line', 'Circle']):
                    if fact in identifier_to_full_line:
                        proof_full_lines.add(identifier_to_full_line[fact])

            found_state = True
            break

    if not found_state:
        print(f"Warning: 未在 problem_1.json 中找到目标为 {goal} 的有效 current_state")
        # 如果找不到证明数据，就无法计算辅助线，直接返回
        return

    # 5. 计算差异 (辅助构图)
    auxiliary_lines = proof_full_lines - set(basic_lines)

    print("\n" + "=" * 40)
    print("【合成训练数据样本 (Fixed & Robust)】")
    print("=" * 40)
    print(f"Goal (Input): {notes[goal_node_idx]}")
    print("-" * 20)
    print(f"Basic Construction (Input : 数量: {len(basic_lines)}):")
    for l in basic_lines:
        print(f"  {l}")
    print("-" * 20)
    print(f"Auxiliary Construction (Output : 数量: {len(auxiliary_lines)}):")
    for l in list(auxiliary_lines):
        print(f"  {l}")
    print("=" * 40)


# --- 运行配置 ---
constructions = [
    "Point(D):FreePoint(D)",
    "Point(B):FreePoint(B)",
    "Point(C):PointLeftSegment(C,D,B)",
    "Line(s):PointOnLine(C,s)&PointOnLine(B,s)",
    "Line(l):PointOnLine(D,l)&PointOnLine(B,l)",
    "Line(m):PointOnLine(C,m)&LinesParallel(l,m)",
    "Line(i):PointOnLine(D,i)&PointOnLine(C,i)",
    "Line(j):PointOnLine(B,j)&LinesParallel(i,j)",
    "Point(E):PointOnLine(E,j)&PointOnLine(E,m)",
    "Line(k):PointOnLine(D,k)&PointOnLine(E,k)",
    "Line(p):PointOnLine(D,p)"
]

goal = "SegmentEqualSegment(B, E, D, C)"

generate_auxiliary_training_data('data/hypergraph-1-solve_add_aux.json', 'tp_data/problem_1_add_aux.json',
                                 constructions, goal)