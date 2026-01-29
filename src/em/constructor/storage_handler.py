import os
import json
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, Any


class StorageHandler:
    """
    JSONL格式的构型存储处理器

    支持按CDL长度分类存储、进程安全写入、自动验证等功能
    """

    def __init__(self, base_dir: str, auto_validate: bool = True, batch_size: int = 5000):
        """
        初始化存储处理器

        Args:
            base_dir: 基础目录
            auto_validate: 是否自动验证构型
            batch_size: 每个JSONL文件包含的构型数量
        """
        self.base_dir = base_dir
        self.auto_validate = auto_validate
        self.batch_size = batch_size
        self.buffers = defaultdict(list)  # {length: [config_dicts]}
        self.file_counters = defaultdict(int)  # {length: current_file_index}
        self.total_added = 0
        self.total_validated = 0
        self.total_failed = 0

    def add(self, generator) -> bool:
        """
        添加一个构型到缓冲区

        Args:
            generator: GeoCfgGenerator实例

        Returns:
            bool: 是否成功添加
        """
        # 1. 验证（如果启用）
        if self.auto_validate:
            start_time = datetime.now()
            try:
                validation_result = generator.validate_and_save()
                validation_time = datetime.now()

                generator.validation_result = validation_result
                generator.validation_time = validation_time
                generator.validation_duration_ms = (validation_time - start_time).total_seconds() * 1000

                # 检查是否所有CDL都通过
                if not validation_result["all_passed"]:
                    self.total_failed += 1
                    return False  # 验证失败，不保存

                self.total_validated += 1
            except Exception as e:
                print(f"验证失败: {e}")
                self.total_failed += 1
                return False

        # 2. 构建配置字典
        config_dict = {
            "constructions": generator.all_cdls,
            "all_constraints": list(generator.all_constraints),
            "entity_ranks": {k: int(v) for k, v in generator.entity_ranks.items()},
            "generation_metadata": {
                "created_at": generator.creation_time.strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3],
                "gdl_file": generator.gdl_file if generator.gdl_file else "unknown",
                "seed": generator.seed,
                "target_entity_num": len(generator.all_cdls),
                "generator_version": "2.0",
            },
        }

        # 添加验证信息
        if generator.validation_result is not None:
            config_dict["validation_metadata"] = {
                "validated": True,
                "validation_time": generator.validation_time.strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3],
                "validation_duration_ms": generator.validation_duration_ms,
                "all_cdls_passed": generator.validation_result["all_passed"],
                "passed_count": generator.validation_result["passed_count"],
                "failed_count": generator.validation_result["failed_count"],
            }

        # 3. 按长度分类
        length = len(config_dict["constructions"])
        self.buffers[length].append(config_dict)

        # 4. 检查是否需要刷新
        if len(self.buffers[length]) >= self.batch_size:
            self.flush(length)

        self.total_added += 1
        return True

    def flush(self, length: Optional[int] = None):
        """
        刷新缓冲区到文件

        Args:
            length: 如果指定，只刷新该长度的缓冲区；否则刷新所有
        """
        lengths_to_flush = [length] if length is not None else list(self.buffers.keys())

        for len_val in lengths_to_flush:
            if not self.buffers[len_val]:
                continue

            # 生成文件名（进程安全）
            filename = self._get_process_safe_filename(len_val)
            filepath = os.path.join(self.base_dir, f"e{len_val}", filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 写入JSONL文件
            with open(filepath, "a", encoding="utf-8") as f:
                for config in self.buffers[len_val]:
                    f.write(json.dumps(config, ensure_ascii=False) + "\n")

            # 清空缓冲区
            self.buffers[len_val] = []
            self.file_counters[len_val] += 1

    def flush_all(self):
        """刷新所有缓冲区"""
        self.flush()

    def _get_process_safe_filename(self, length: int) -> str:
        """
        生成进程安全的文件名

        格式: e{length}_pid{pid}_{timestamp}_{index:03d}.jsonl

        Args:
            length: CDL长度

        Returns:
            str: 文件名
        """
        pid = os.getpid()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        index = self.file_counters[length]
        return f"e{length}_pid{pid}_{timestamp}_{index:03d}.jsonl"

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            dict: 统计信息
        """
        return {
            "total_added": self.total_added,
            "total_validated": self.total_validated,
            "total_failed": self.total_failed,
            "success_rate": (self.total_validated / self.total_added * 100 if self.total_added > 0 else 0),
            "pending_buffers": {f"e{length}": len(buffer) for length, buffer in self.buffers.items()},
        }


# ==================== 工具函数 ====================


def read_jsonl(jsonl_file: str):
    """
    读取JSONL文件，返回构型列表

    Args:
        jsonl_file: JSONL文件路径

    Returns:
        list: 构型字典列表
    """
    configs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                configs.append(json.loads(line))
    return configs


def read_all_configs(base_dir: str, length: int):
    """
    读取指定长度的所有构型

    Args:
        base_dir: 基础目录
        length: CDL长度

    Returns:
        list: 所有构型字典列表
    """
    length_dir = os.path.join(base_dir, f"e{length}")
    configs = []

    if not os.path.exists(length_dir):
        print(f"目录不存在: {length_dir}")
        return configs

    for filename in sorted(os.listdir(length_dir)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(length_dir, filename)
            configs.extend(read_jsonl(filepath))

    return configs


def extract_config(jsonl_file: str, line_number: int, output_file: str):
    """
    从JSONL文件中提取第N个构型，保存为单独的JSON文件

    Args:
        jsonl_file: JSONL文件路径
        line_number: 行号（从0开始）
        output_file: 输出JSON文件路径

    Raises:
        IndexError: 如果行号超出范围
    """
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if line_number >= len(lines):
        raise IndexError(f"文件只有 {len(lines)} 行，请求行号 {line_number}")

    config = json.loads(lines[line_number].strip())

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"构型已提取到: {output_file}")


def get_all_lengths(base_dir: str):
    """
    获取所有可用的CDL长度

    Args:
        base_dir: 基础目录

    Returns:
        list: 可用的长度列表
    """
    lengths = []

    if not os.path.exists(base_dir):
        return lengths

    for dirname in os.listdir(base_dir):
        if dirname.startswith("len_"):
            try:
                length = int(dirname.split("_")[1])
                lengths.append(length)
            except (ValueError, IndexError):
                continue

    return sorted(lengths)


def count_configs_by_length(base_dir: str, length: int) -> int:
    """
    统计指定长度的构型数量

    Args:
        base_dir: 基础目录
        length: CDL长度

    Returns:
        int: 构型数量
    """
    length_dir = os.path.join(base_dir, f"e{length}")

    if not os.path.exists(length_dir):
        return 0

    total_count = 0
    for filename in os.listdir(length_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(length_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                total_count += sum(1 for line in f if line.strip())

    return total_count


def read_jsonl_as_json_list(jsonl_file: str) -> list:
    """
    读取JSONL文件并返回构型列表（JSON格式）

    Args:
        jsonl_file: JSONL文件路径

    Returns:
        list: 构型字典列表
    """
    configs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                config = json.loads(line)
                configs.append(config)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                continue

    return configs


def read_jsonl_as_single_json(jsonl_file: str, line_number: int) -> dict:
    """
    读取JSONL文件中的单个构型并返回（JSON格式）

    Args:
        jsonl_file: JSONL文件路径
        line_number: 行号（从0开始）

    Returns:
        dict: 单个构型字典
    """
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if line_number >= len(lines):
        raise IndexError(f"文件只有 {len(lines)} 行，请求行号 {line_number}")

    line = lines[line_number].strip()
    if not line:
        raise IndexError(f"第 {line_number} 行为空")

    try:
        config = json.loads(line)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"第 {line_number} 行JSON解析失败: {e}")
