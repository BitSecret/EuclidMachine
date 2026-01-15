import json
import networkx as nx
import re



# --- 映射回随机初始构图中的语句 ---

def map_indices_to_lines(indices, lines, notes):
    """
     1.构建映射：["Point", "D"] -> "Point(D):FreePoint(D)"
     2.遍历indices的索引(notes中的id)，从映射表中查找，返回相关的构图语句集合 mapped_lines
    """
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
    pred = goal_str.split('(')[0].strip()
    match = re.search(r'\((.*?)\)', goal_str)
    if match:
        params = [p.strip() for p in match.group(1).split(',')]
    else:
        params = []
    return pred, params


# --- 主逻辑函数 ---

def generate_all_training_data(dependent_entities_json_path, proof_json_path, construction_lines, output_file_path):
    print(f"正在加载超图数据: {dependent_entities_json_path} ...")
    # 1. 加载超图数据 (只加载一次)
    with open(dependent_entities_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    notes = data['notes']
    deps = data['dependent_entities']

    # 2. 构建依赖图 (Edge: Child -> Parent)
    G = nx.DiGraph()
    for child_idx, parents in enumerate(deps):
        for p_idx in parents:
            G.add_edge(child_idx, p_idx)

    # 3. 建立构图语句查找表
    identifier_to_full_line = {}
    for line in construction_lines:
        if ':' in line:
            identifier = line.split(':')[0].strip()
        else:
            identifier = line.strip()
        identifier_to_full_line[identifier] = line

    print(f"正在加载证明轨迹数据: {proof_json_path} ...")
    # 4. 加载所有证明目标数据
    with open(proof_json_path, 'r', encoding='utf-8') as f:
        tp_datas = json.load(f)

    all_training_samples = []

    print(f"开始处理 {len(tp_datas)} 个目标状态...")

    # =======================================================
    # 【修改点 1】: 遍历 proof_json 中的每一条数据 (每一个Goal)
    # =======================================================
    for index, tp_data in enumerate(tp_datas):
        current_goal_str = tp_data['goal']

        # --- A. 确定“基础构图” (Basic Construction) ---
        target_pred, target_params = parse_goal_string(current_goal_str)
        target_params_set = set(target_params)

        goal_node_idx = -1

        # 在超图中寻找目标节点
        for i in range(len(notes) - 1, -1, -1):
            note = notes[i]
            if note[0] == target_pred:
                if set(note[1:]) == target_params_set:
                    goal_node_idx = i
                    break

        # 如果超图中找不到这个目标（可能是纯代数步骤或非几何节点），则跳过
        if goal_node_idx == -1:
            # print(f"Skipping: Goal '{current_goal_str}' not found in Hypergraph nodes.")
            continue

        # 回溯定义依赖
        target_entity_indices = []
        for p_name in target_params:
            for i, note in enumerate(notes):
                # 兼容 Point, Line, Circle
                if note[0] in ["Point", "Line", "Circle"] and note[1] == p_name:
                    target_entity_indices.append(i)
                    break

        # 如果参数找不到定义（极端情况），跳过
        if not target_entity_indices and target_params:
            continue

        basic_indices = set()
        for idx in target_entity_indices:
            basic_indices.add(idx)
            basic_indices.update(nx.descendants(G, idx))

        basic_construction_indices = {
            i for i in basic_indices
            if notes[i][0] in ['Point', 'Line', 'Circle']
        }

        basic_lines = map_indices_to_lines(basic_construction_indices, construction_lines, notes)

        # --- B. 确定“有效构图” (Proof/Effective Construction) ---
        # 直接使用当前 tp_data 中的 current_state，无需再次搜索
        proof_full_lines = set()
        if 'current_state' in tp_data:
            current_facts = tp_data['current_state']
            for fact in current_facts:
                if any(fact.startswith(prefix) for prefix in ['Point', 'Line', 'Circle']):
                    if fact in identifier_to_full_line:
                        proof_full_lines.add(identifier_to_full_line[fact])

        # --- C. 计算辅助构图 (Auxiliary) ---
        auxiliary_lines = proof_full_lines - set(basic_lines)

        # =======================================================
        # 【修改点 2】: 构建数据对象并添加到列表
        # =======================================================
        # 只有当存在有效构图时才保存（或者根据需求保留空辅助线样本作为负样本）
        # 这里我们保存所有能解析的样本

        sample = {
            "input": {
                "basic_construction": basic_lines,
                "goal": current_goal_str
            },
            "output": {
                "auxiliary_construction": list(auxiliary_lines)
            }
        }
        all_training_samples.append(sample)

        # 可选：打印进度
        if index % 5 == 0:
            print(f"Processed {index}/{len(tp_datas)}: Found {len(auxiliary_lines)} aux lines for {current_goal_str}")

    # =======================================================
    # 【修改点 3】: 将所有收集到的数据写入 JSON 文件
    # =======================================================
    print(f"\n处理完成。共生成 {len(all_training_samples)} 条训练数据。")
    print(f"正在写入文件: {output_file_path} ...")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_training_samples, f, indent=2, ensure_ascii=False)

    print("写入成功！")


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

# 定义输入文件路径
hyper_json_path = 'data/hypergraph-1-solve.json'
proof_json_path = 'tp_data/problem_1.json'
output_json_path = 'aux_data/aux_data_1.json'  # 输出文件名

# 执行
generate_all_training_data(hyper_json_path, proof_json_path, constructions, output_json_path)