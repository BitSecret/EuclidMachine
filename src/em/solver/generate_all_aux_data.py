import json
import networkx as nx
import re
import os
import glob


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


# --- 核心处理逻辑 ---

def generate_training_data_for_single_file(hyper_json_path, proof_json_path, construction_lines, output_file_path):
    # 检查依赖文件是否存在
    if not os.path.exists(hyper_json_path):
        print(f"[跳过] Hypergraph文件不存在: {hyper_json_path}")
        return

    print(f"正在读取 Hypergraph: {hyper_json_path} ...")
    with open(hyper_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    notes = data['notes']
    deps = data['dependent_entities']

    # 构建依赖图 (Edge: Child -> Parent)
    G = nx.DiGraph()
    for child_idx, parents in enumerate(deps):
        for p_idx in parents:
            G.add_edge(child_idx, p_idx)

    # 建立构图语句查找表
    identifier_to_full_line = {}
    for line in construction_lines:
        if ':' in line:
            identifier = line.split(':')[0].strip()
        else:
            identifier = line.strip()
        identifier_to_full_line[identifier] = line

    print(f"正在读取 Proof Data: {proof_json_path} ...")
    with open(proof_json_path, 'r', encoding='utf-8') as f:
        tp_datas = json.load(f)

    all_training_samples = []

    # 遍历该文件中的每一个 Goal
    for index, tp_data in enumerate(tp_datas):
        current_goal_str = tp_data.get('goal', '')
        if not current_goal_str: continue

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

        # 如果超图中找不到这个目标，则跳过
        if goal_node_idx == -1:
            continue

        # 回溯定义依赖
        target_entity_indices = []
        for p_name in target_params:
            for i, note in enumerate(notes):
                if note[0] in ["Point", "Line", "Circle"] and note[1] == p_name:
                    target_entity_indices.append(i)
                    break

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
        proof_full_lines = set()
        if 'current_state' in tp_data:
            current_facts = tp_data['current_state']
            for fact in current_facts:
                if any(fact.startswith(prefix) for prefix in ['Point', 'Line', 'Circle']):
                    if fact in identifier_to_full_line:
                        proof_full_lines.add(identifier_to_full_line[fact])

        # --- C. 计算辅助构图 (Auxiliary) ---
        auxiliary_lines = proof_full_lines - set(basic_lines)

        # 保存结果
        sample = {
            #"problem_source": os.path.basename(proof_json_path),  # 记录来源文件
            #"goal_index": index,
            "input": {
                "basic_construction": basic_lines,
                "goal": current_goal_str
            },
            "output": {
                "auxiliary_construction": list(auxiliary_lines)
            }
        }
        all_training_samples.append(sample)

    # 写入 JSON 文件
    if all_training_samples:
        print(f"写入文件: {output_file_path} (共 {len(all_training_samples)} 条数据)")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_training_samples, f, indent=2, ensure_ascii=False)
    else:
        print(f"警报: {proof_json_path} 未生成任何有效数据。")


# --- 批量处理主函数 ---

def batch_process_auxiliary_data():
    # 路径配置
    tp_data_dir = 'tp_data'  # 输入：推理数据目录
    hyper_data_dir = 'data'  # 输入：超图数据目录
    aux_data_dir = 'aux_data'  # 输出：辅助线数据目录
    problem_file = '/home/lengmen/szh/PythonWorkSpace/EuclidMachine/data/test_problem/problem.json'  # 输入：包含 constructions 的总表

    # 0. 预检
    if not os.path.exists(problem_file):
        print(f"错误: 找不到 {problem_file}，无法获取构图语句。")
        return

    if not os.path.exists(aux_data_dir):
        os.makedirs(aux_data_dir)
        print(f"创建输出目录: {aux_data_dir}")

    # 1. 加载所有题目的 Constructions
    print("正在加载 problem.json ...")
    with open(problem_file, 'r', encoding='utf-8') as f:
        problems_data = json.load(f)

    # 2. 获取 tp_data 目录下所有的 .json 文件
    # 排除非 json 文件
    tp_files = [f for f in os.listdir(tp_data_dir) if f.endswith('.json')]

    # 正则表达式：用于匹配 "problem_1.json" 或 "problem_1_multiple.json" 中的数字 ID
    # 解释：problem_ 后面跟数字(\d+)，后面可能跟 _multiple，最后是 .json
    id_pattern = re.compile(r'problem_(\d+)(?:_multiple)?\.json')

    print(f"找到 {len(tp_files)} 个待处理文件。")
    print("-" * 60)

    for filename in sorted(tp_files):
        match = id_pattern.match(filename)
        if not match:
            print(f"[跳过] 无法从文件名解析ID: {filename}")
            continue

        pid = match.group(1)  # 提取出的题目ID，如 "1"

        # 检查 problem.json 中是否有这个 ID
        if pid not in problems_data:
            print(f"[跳过] problem.json 中不存在 ID 为 {pid} 的题目数据。")
            continue

        # 获取该题目的构图语句
        constructions = problems_data[pid].get('constructions', [])

        # 定义路径
        proof_path = os.path.join(tp_data_dir, filename)

        # 假设 hypergraph 文件名格式为 hypergraph-{id}-solve.json
        hyper_path = os.path.join(hyper_data_dir, f"hypergraph-{pid}-solve.json")

        # 定义输出路径，保持和输入文件名类似的对应关系
        # problem_1.json -> aux_data_1.json
        # problem_1_multiple.json -> aux_data_1_multiple.json
        output_filename = filename.replace("problem_", "aux_data_")
        output_path = os.path.join(aux_data_dir, output_filename)

        print(f"正在处理: {filename} (ID: {pid})")

        # 执行生成
        try:
            generate_training_data_for_single_file(
                hyper_path,
                proof_path,
                constructions,
                output_path
            )
        except Exception as e:
            print(f"[错误] 处理 {filename} 时发生异常: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 60)

    print("所有文件批量处理完毕。")


if __name__ == "__main__":
    batch_process_auxiliary_data()