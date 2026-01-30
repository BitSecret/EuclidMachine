import os
import json
import logging
import logging.config


from pprint import pprint

from em.formalgeo.tools import load_json, parse_gdl
from em.constructor.gc_generator import GeoCfgGenerator as gcgen

# from em.formalgeo.configuration import GeometricConfiguration
# from constructor.gcgenerator import GeoConfigurationGenerator


def setup_logging(log_config_path):
    """
    设置日志配置并返回logger实例

    Args:
        log_config_path (str): 日志配置文件路径

    Returns:
        logging.Logger: 配置好的logger实例
    """
    if os.path.exists(log_config_path):
        config_dict = load_json(log_config_path)
        logging.config.dictConfig(config_dict)
        logger = logging.getLogger(__name__)
        logger.info(f"Logging using Config File: {log_config_path}")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()],  # 输出到控制台
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging using Basic Config")

    return logger


def test_gcgen(gdl, gc_save_dir, p_gdl_save_path, logger: logging.Logger):
    """
    测试几何构型生成器

    Args:
        gdl (dict): GDL配置
        gc_save_path (str): 生成构型的保存路径
        p_gdl_save_path (str): 解析GDL保存路径
        logger (logging.Logger): 日志记录器
    """
    logger.info("GeoConfig Generation Test")
    logger.info(f"Generated CDL will save to: [Dir] {gc_save_dir}")
    logger.info(f"Parsed GDL will save as: [File] {p_gdl_save_path}")

    gcg = gcgen(gdl)
    # Save parsed_gdl
    # gcg.save_parsed_gdl_to(p_gdl_save_path)

    # flow control
    generate = True
    validate = False
    # generation config
    gen_num = 1
    target_entity_num = 5
    constraint_per_ent = 4

    if generate:
        for i in range(gen_num):
            pprint(gcg.generate(target_entity_num, constraint_per_ent))
            # gcg.check_constructions()
            # gcg.save_cdls_to(gc_save_path)
        logger.info("Generation test complete!")
    else:
        logger.info("No Generation test!")

    if validate:
        gcg.batch_val(gc_save_dir)
    else:
        logger.info("No Validation test!")


if __name__ == "__main__":
    # gdl, gc(example), problem_id
    gdl_file = "../../../data/gdl/gdl-yc-260126.json"
    output_parsed_gdl_file = "../../../data/outputs/parsed_gdl.json"
    parsed_gdl_file = "../../../data/gdl/parsed_gdl_new.json"

    gc_file = "../../../data/gdl/gc-yuchang.json"

    logging_cfg_file = "./logging_config.json"

    # file save path
    gen_gc_save_dir = "../../../data/cdl_batch_gen/"

    # 先将gdl解析保存为更易用的json格式 同步可能的gdl修改
    # save_readable_parsed_gdl(
    #     parsed_gdl=parse_gdl(load_json(gdl_file_path)),
    #     filename=parsed_gdl_file_path
    # )

    # 加载gdl
    # parsed_gdl = load_json(parsed_gdl_file_path)
    gdl_load = load_json(gdl_file)

    # 设置日志配置并获取logger
    logger = setup_logging(logging_cfg_file)

    test_gcgen(gdl_load, gen_gc_save_dir, parsed_gdl_file, logger)
