"""
目的：负责组织数据并启动强化学习训练过程 。
核心功能：
    加载由 data_processor.py 处理好的数据。
    调用一个离线强化学习框架（可能是外部库）来训练 Neural-based Solver 。
    保存训练好的模型，以供 Neural-based Solver.py 加载和使用。
"""