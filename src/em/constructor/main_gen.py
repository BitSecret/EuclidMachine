import os
import sys
import random
import multiprocessing

from em.constructor.gc_generator import GeoCfgGenerator
from em.constructor.storage_handler import StorageHandler
from em.formalgeo.tools import load_json
from em.constructor.config_manager import ConfigManager


def worker_task(args):
    """
    子进程执行的函数
    在每个子进程中生成构型

    Args:
        args: (gdl_file, seed_entity_pairs) 元组
              seed_entity_pairs: [(seed1, entity_num1), (seed2, entity_num2), ...]

    Returns:
        list: GeoCfgGenerator 实例列表
    """
    gdl_file, seed_entity_pairs = args

    # 在子进程中加载GDL（避免序列化问题）
    gdl = load_json(gdl_file)

    generators = []
    for seed, entity_num in seed_entity_pairs:
        try:
            generator = GeoCfgGenerator(gdl, seed=seed)
            # 使用指定的实体数量
            generator.generate(target_entity_num=entity_num)
            generators.append(generator)
        except Exception as e:
            print(f"生成失败 (seed={seed}, entity_num={entity_num}): {e}")

    return generators


def batch_generate(per_count=None, min_entities=None, max_entities=None, num_processes=4, gdl_file=None, output_dir=None, config_manager=None):
    """
    批量生成构型

    Args:
        per_count: 每类实体生成的构型数量（如 10 表示每个实体数量生成10个构型）
        min_entities: 最小实体数量
        max_entities: 最大实体数量
        num_processes: 使用的进程数
        gdl_file: GDL文件路径
        output_dir: 输出目录
        config_manager: 配置管理器实例
    """
    # 初始化配置管理器
    if config_manager is None:
        config_manager = ConfigManager()

    # 默认值
    if gdl_file is None:
        gdl_file = "../../../data/gdl/gdl-yc-260126.json"

    if output_dir is None:
        output_dir = "../../../data/cdl_batch_gen"

    # 检查GDL文件是否存在
    if not os.path.exists(gdl_file):
        print(f"错误: GDL文件不存在: {gdl_file}")
        return

    # 构建实体数量目标分布
    target_distribution = {}

    # 优先级 1: 命令行参数
    if per_count is not None and min_entities is not None and max_entities is not None:
        for entity_num in range(min_entities, max_entities + 1):
            target_distribution[entity_num] = per_count
        print(f"使用命令行参数: 每类生成 {per_count} 个构型，实体数量范围 {min_entities}-{max_entities}")
    # 优先级 2: 配置文件
    else:
        target_distribution = config_manager.get_target_distribution()
        if target_distribution:
            print(f"使用配置文件中的 target_distribution: {target_distribution}")
        else:
            print("错误: 未指定实体数量目标分布，请通过命令行参数或配置文件指定")
            return

    # 初始化存储处理器
    storage = StorageHandler(base_dir=output_dir, auto_validate=True, batch_size=5000)

    # 计算总数量
    total_configs = sum(target_distribution.values())

    print(f"开始生成 {total_configs} 个构型...")
    print(f"GDL文件: {gdl_file}")
    print(f"输出目录: {output_dir}")
    print(f"进程数: {num_processes}")
    print(f"自动验证: {storage.auto_validate}")
    print("-" * 50)

    # 准备种子和实体数量对
    seed_entity_pairs = []
    for entity_num, count in target_distribution.items():
        for _ in range(count):
            seed = random.randint(0, 1000000)
            seed_entity_pairs.append((seed, entity_num))

    # 打乱顺序以避免相同实体数量连续生成
    random.shuffle(seed_entity_pairs)

    # 将任务分配给各个进程
    chunk_size = max(1, len(seed_entity_pairs) // num_processes)
    tasks = [(gdl_file, seed_entity_pairs[i : i + chunk_size]) for i in range(0, len(seed_entity_pairs), chunk_size)]

    # 多进程生成
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 获取结果
        for batch_generators in pool.imap_unordered(worker_task, tasks):
            # 主进程负责验证和写入
            for gen in batch_generators:
                # 设置GDL文件名
                gen.gdl_file = os.path.basename(gdl_file)

                # 添加到存储
                success = storage.add(gen)
                if success:
                    print(f"✓ 已保存构型 (seed={gen.seed}, 实体数={gen.target_entity_num}, 长度={len(gen.all_cdls)})")
                else:
                    print(f"✗ 验证失败 (seed={gen.seed})")

    # 刷新所有缓冲区
    storage.flush_all()

    # 打印统计
    stats = storage.get_statistics()
    print("-" * 50)
    print("生成完成！")
    print(f"总添加数: {stats['total_added']}")
    print(f"验证通过: {stats['total_validated']}")
    print(f"验证失败: {stats['total_failed']}")
    print(f"成功率: {stats['success_rate']:.2f}%")
    print(f"待处理缓冲区: {stats['pending_buffers']}")


def simple_run():
    """
    简单示例：生成少量构型（使用配置文件中的 target_distribution）
    """
    print("Simple Example running...")

    # 初始化配置管理器
    config_manager = ConfigManager()

    # 加载GDL
    gdl_file = "../../../data/gdl/gdl-yc-260126.json"
    if not os.path.exists(gdl_file):
        print(f"错误: GDL文件不存在: {gdl_file}")
        return
    else:
        print(f"GDL from: {gdl_file}")

    gdl = load_json(gdl_file)

    # 从配置中读取实体数量目标分布
    target_distribution = config_manager.get_target_distribution()
    if not target_distribution:
        print("错误: 配置文件中未配置 target_distribution，请在 generation_config.json 中配置")
        return

    print(f"使用配置文件中的 target_distribution: {target_distribution}")

    # 初始化存储处理器
    storage = StorageHandler(base_dir="../../../data/cdl_batch_gen", auto_validate=True, batch_size=5)  # 小批量用于测试

    # 生成构型
    success_count = 0
    fail_count = 0

    for entity_num, count in target_distribution.items():
        for i in range(count):
            seed = random.randint(0, 1000000)
            try:
                generator = GeoCfgGenerator(gdl, seed=seed)
                # 使用指定的实体数量
                generator.generate(target_entity_num=entity_num)
                generator.gdl_file = os.path.basename(gdl_file)

                success = storage.add(generator)
                if success:
                    success_count += 1
                    print(f"✓ 构型 {success_count+fail_count}/{sum(target_distribution.values())} (seed={seed}, 实体数={entity_num}, 长度={len(generator.all_cdls)})")
                else:
                    fail_count += 1
                    print(f"✗ 构型 {success_count+fail_count}/{sum(target_distribution.values())} (seed={seed}, 实体数={entity_num}) 验证失败")
            except Exception as e:
                fail_count += 1
                print(f"✗ 构型 {success_count+fail_count}/{sum(target_distribution.values())} (seed={seed}, 实体数={entity_num}) 生成失败: {e}")

    # 刷新所有缓冲区
    storage.flush_all()

    # 打印统计
    print("-" * 50)
    print(f"生成完成！成功: {success_count}, 失败: {fail_count}")
    print(f"验证失败: {stats['total_failed']}")
    print(f"成功率: {stats['success_rate']:.2f}%")


def read_jsonl_as_json(jsonl_file, output_file=None):
    """
    读取JSONL文件并输出为JSON格式

    Args:
        jsonl_file: JSONL文件路径
        output_file: 输出JSON文件路径（可选）
    """
    from em.constructor.storage_handler import read_jsonl_as_json_list
    import json

    print(f"正在读取JSONL文件: {jsonl_file}")
    configs = read_jsonl_as_json_list(jsonl_file)

    print(f"成功读取 {len(configs)} 个构型")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(configs, f, indent=2, ensure_ascii=False)
        print(f"构型已保存到: {output_file}")
    else:
        # 直接打印前几个构型作为示例
        print("\n前3个构型预览:")
        for i, config in enumerate(configs[:3]):
            print(f"\n构型 {i+1}:")
            print(f"  种子: {config.get('generation_metadata', {}).get('seed', 'N/A')}")
            print(f"  长度: {len(config.get('constructions', []))}")
            print(f"  构造: {config.get('constructions', [])[:2]}...")


def read_jsonl_single_as_json(jsonl_file, line_number, output_file=None):
    """
    读取JSONL文件中的单个构型并输出为JSON格式

    Args:
        jsonl_file: JSONL文件路径
        line_number: 行号（从0开始）
        output_file: 输出JSON文件路径（可选）
    """
    from em.constructor.storage_handler import read_jsonl_as_single_json
    import json

    print(f"正在读取JSONL文件第 {line_number} 行: {jsonl_file}")
    config = read_jsonl_as_single_json(jsonl_file, line_number)

    print(f"成功读取构型:")
    print(f"  种子: {config.get('generation_metadata', {}).get('seed', 'N/A')}")
    print(f"  长度: {len(config.get('constructions', []))}")
    print(f"  构造: {config.get('constructions', [])}")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"构型已保存到: {output_file}")


def parse_command_line_args():
    """
    解析命令行参数并返回操作类型和参数字典

    Returns:
        tuple: (operation_type, params_dict)
    """
    if len(sys.argv) < 2:
        return "simple", {}

    operation = sys.argv[1]

    # 处理帮助参数
    if operation in ["-h", "--help", "help"]:
        print("=" * 70)
        print("几何构型生成系统 - 命令行工具")
        print("=" * 70)
        print("")
        print("用法:")
        print("  python main_gen.py <操作> [参数...]")
        print("")
        print("可用操作:")
        print("  simple              运行简单示例（使用配置文件中的 target_distribution）")
        print("  batch [每类数量] [最小实体数] [最大实体数] [进程数]  批量生成构型")
        print("  test [每类数量] [最小实体数] [最大实体数] [进程数]   测试生成模式")
        print("  read <jsonl文件> [输出json文件]  读取JSONL并输出为JSON格式")
        print("  read-single <jsonl文件> <行号> [输出json文件]  读取单个构型")
        print("  help, --help       显示此帮助信息")
        print("")
        print("配置说明:")
        print("  方式1: 通过命令行参数指定（优先级更高）")
        print("    python main_gen.py batch 10 6 15 2")
        print("      → 每类实体生成10个构型，实体数量范围6-15，使用2个进程")
        print("      → 总计生成: 10类 × 10个 = 100个构型")
        print("")
        print("  方式2: 通过 generation_config.json 配置")
        print("    entity_generation.target_distribution:")
        print("      {\"6\": 10, \"7\": 10, \"8\": 10, ...}")
        print("      → 实体数量为6的生成10个，7的生成10个，等等")
        print("")
        print("示例:")
        print("  python main_gen.py simple")
        print("      → 运行简单示例，使用配置文件中的 target_distribution")
        print("")
        print("  python main_gen.py batch 10 6 15 2")
        print("      → 每类生成10个，范围6-15，使用2个进程，总计100个")
        print("")
        print("  python main_gen.py batch 5 8 12 4")
        print("      → 每类生成5个，范围8-12，使用4个进程，总计25个")
        print("")
        print("  python main_gen.py read data/cdl_batch_gen/e2/e2_*.jsonl")
        print("      → 读取JSONL文件并显示前3个构型预览")
        print("")
        print("  python main_gen.py read-single data/cdl_batch_gen/e2/e2_*.jsonl 0")
        print("      → 读取JSONL文件中的第一个构型")
        print("")
        print("配置文件位置:")
        print("  src/em/constructor/generation_config.json")
        print("")
        print("更多信息:")
        print("  参见 IFLOW.md 文档")
        print("=" * 70)
        return None, None

    if operation == "simple":
        return "simple", {}
    elif operation == "batch":
        # 新格式: batch [per_count] [min_entities] [max_entities] [num_processes]
        # 例如: batch 10 6 15 2  → 每类10个，范围6-15，2个进程
        per_count = int(sys.argv[2]) if len(sys.argv) > 2 else None
        min_entities = int(sys.argv[3]) if len(sys.argv) > 3 else None
        max_entities = int(sys.argv[4]) if len(sys.argv) > 4 else None
        num_processes = int(sys.argv[5]) if len(sys.argv) > 5 else None
        return "batch", {"per_count": per_count, "min_entities": min_entities, "max_entities": max_entities, "num_processes": num_processes}
    elif operation == "test":
        # 新格式: test [per_count] [min_entities] [max_entities] [num_processes]
        # 例如: test 5 6 10 2  → 每类5个，范围6-10，2个进程
        per_count = int(sys.argv[2]) if len(sys.argv) > 2 else None
        min_entities = int(sys.argv[3]) if len(sys.argv) > 3 else None
        max_entities = int(sys.argv[4]) if len(sys.argv) > 4 else None
        num_processes = int(sys.argv[5]) if len(sys.argv) > 5 else None
        return "test", {"per_count": per_count, "min_entities": min_entities, "max_entities": max_entities, "num_processes": num_processes}
    elif operation == "read":
        if len(sys.argv) < 3:
            print("用法:")
            print("  python main_gen.py read <jsonl文件> [输出json文件]")
            print("例如:")
            print("  python main_gen.py read data/cdl_batch_gen/e2/e2_*.jsonl")
            print("  python main_gen.py read data/cdl_batch_gen/e2/e2_*.jsonl configs.json")
            return None, None
        jsonl_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        return "read", {"jsonl_file": jsonl_file, "output_file": output_file}
    elif operation == "read-single":
        if len(sys.argv) < 4:
            print("用法:")
            print("  python main_gen.py read-single <jsonl文件> <行号> [输出json文件]")
            print("例如:")
            print("  python main_gen.py read-single data/cdl_batch_gen/e2/e2_*.jsonl 0")
            print("  python main_gen.py read-single data/cdl_batch_gen/e2/e2_*.jsonl 0 config.json")
            return None, None
        jsonl_file = sys.argv[2]
        line_number = int(sys.argv[3])
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        return "read-single", {"jsonl_file": jsonl_file, "line_number": line_number, "output_file": output_file}
    else:
        print("用法:")
        print("  python main_gen.py simple          # 运行简单示例（10个构型）")
        print("  python main_gen.py batch [数量] [进程数]  # 批量生成")
        print("  python main_gen.py test [数量] [进程数]   # 测试生成")
        print("  python main_gen.py read <jsonl文件> [输出json文件]  # 读取JSONL并输出为JSON格式")
        print("  python main_gen.py read-single <jsonl文件> <行号> [输出json文件]  # 读取单个构型")
        print("")
        print("实体数量范围配置:")
        print("  通过 generation_config.json 文件配置实体数量范围")
        print("  entity_generation.constraints.min_total_entities: 最小实体数")
        print("  entity_generation.constraints.max_total_entities: 最大实体数")
        print("  每个构型的实体数量将在配置范围内随机选择，类型自由组合")
        print("")
        print("例如:")
        print("  python main_gen.py simple       # 运行简单示例（10个构型）")
        print("  python main_gen.py batch 100 2  # 生成100个构型，使用2个进程")
        print("  python main_gen.py test 50 2    # 测试50个构型，使用2个进程")
        print("  python main_gen.py read data/cdl_batch_gen/e2/e2_*.jsonl")
        print("  python main_gen.py read-single data/cdl_batch_gen/e2/e2_*.jsonl 0")
        return None, None


def execute_operation(operation_type, params):
    """
    根据操作类型执行相应的操作

    Args:
        operation_type: 操作类型
        params: 参数字典
    """
    if operation_type == "simple":
        simple_run()
    elif operation_type == "batch":
        config_manager = ConfigManager()
        per_count = params.get("per_count")
        min_entities = params.get("min_entities")
        max_entities = params.get("max_entities")
        num_processes = params.get("num_processes")

        # 添加 config_manager 参数
        batch_generate(per_count=per_count, min_entities=min_entities, max_entities=max_entities,
                      num_processes=num_processes, config_manager=config_manager)
    elif operation_type == "test":
        config_manager = ConfigManager()
        per_count = params.get("per_count")
        min_entities = params.get("min_entities")
        max_entities = params.get("max_entities")
        num_processes = params.get("num_processes")

        # 添加 config_manager 参数
        batch_generate(per_count=per_count, min_entities=min_entities, max_entities=max_entities,
                      num_processes=num_processes, config_manager=config_manager)
    elif operation_type == "read":
        jsonl_file = params["jsonl_file"]
        output_file = params.get("output_file")
        read_jsonl_as_json(jsonl_file, output_file)
    elif operation_type == "read-single":
        jsonl_file = params["jsonl_file"]
        line_number = params["line_number"]
        output_file = params.get("output_file")
        read_jsonl_single_as_json(jsonl_file, line_number, output_file)


if __name__ == "__main__":
    # 解析命令行参数
    operation_type, params = parse_command_line_args()

    if operation_type is not None:
        try:
            execute_operation(operation_type, params)
        except KeyboardInterrupt:
            print("\n操作被用户中断!")
        except Exception as e:
            print(f"执行操作时发生错误: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("无效的命令行参数")
