import json


def extract_auxiliary_data(json_data, target_node_index):
    hypergraph = json_data['hypergraph']
    notes = json_data['notes']
    edge_defs = json_data['edges']

    # 1. 构建反向依赖图 (Node -> Producing Edge -> Source Nodes)
    # node_to_source_map: { child_node_id: (edge_index, [parent_node_ids]) }
    node_to_source_map = {}
    for entry in hypergraph:
        sources, edge_idx, targets = entry[0], entry[1], entry[2]
        for t in targets:
            node_to_source_map[t] = (edge_idx, sources)

    # 2. 递归获取有效子图 (Dependency Subgraph)
    effective_nodes = set()
    effective_edges = set()

    def trace_back(node_idx):
        if node_idx in effective_nodes:
            return
        effective_nodes.add(node_idx)

        if node_idx in node_to_source_map:
            edge_idx, parents = node_to_source_map[node_idx]
            effective_edges.add(edge_idx)
            for p in parents:
                trace_back(p)

    # 从目标节点开始回溯
    trace_back(target_node_index)

    # 3. 区分 "初始条件" 和 "辅助构图"
    # 逻辑：我们在 effective_edges 中筛选出属于 "Construction" 类型的边
    # 并且排除掉最初的几个（假设前N个是题目给定，或者根据FreePoint判断）

    auxiliary_actions = []
    initial_context_nodes = set()

    # 简单判定：如果是定理应用(如A80)则跳过，如果是定义(Point/Line)则是构图
    construction_indices = []
    for edge_idx in effective_edges:
        edge_content = edge_defs[edge_idx]
        # 判断是否为构图语句（通过检查关键字或格式）
        # json中构图通常包含 ":"，例如 ['Line', 's', ':', ...]
        if ':' in edge_content or edge_content[0] in ['Point', 'Line', 'Circle']:
            # 进一步过滤：FreePoint通常是初始条件
            if 'FreePoint' not in edge_content:
                construction_indices.append(edge_idx)
            else:
                # 记录初始点，用于State构建
                # (这里简化处理，实际需根据edges找到对应的target nodes)
                pass

    # 按 edge_index 排序，保证构图顺序
    construction_indices.sort()

    results = []

    # 4. 生成 ((State, Goal), Action)
    # 假设 construction_indices 中全是必须的辅助构图
    current_state_edges = set()  # 初始为空，或者包含 FreePoints

    goal_text = str(notes[target_node_index])

    for action_edge_idx in construction_indices:
        action_text = str(edge_defs[action_edge_idx])

        # 构造样本
        sample = {
            "state_hypergraph_edges": list(current_state_edges),  # 当前已有的边
            "goal": goal_text,
            "next_auxiliary_action": action_text
        }
        results.append(sample)

        # 将当前动作加入状态，用于下一个样本
        current_state_edges.add(action_edge_idx)

    return results

# --- 使用示例 ---
# 假设我们想证明最后生成的 LinesParallel (索引 177)
#测试git

if __name__ == '__main__':

    data = json.load(open('data/hypergraph-1-solve.json'))
    training_pairs = extract_auxiliary_data(data, 177)
    print(training_pairs)