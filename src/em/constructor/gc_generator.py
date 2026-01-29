import os
import time
import logging
import itertools

from enum import Enum
from random import Random
from sympy import symbols, Matrix
from pprint import pprint
from datetime import datetime
from multiprocessing import pool

from em.formalgeo.tools import entity_letters, save_json, parse_gdl, load_json
from em.formalgeo.tools import _replace_expr, _parse_geometric_fact
from em.formalgeo.configuration import GeometricConfiguration as geocfg
from em.constructor.jacobian_verifier import JacobianMatrixVerifier as jac_verifier, JacobianResult


# setup gcgenerator logger
logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """约束类型枚举"""

    FREE = "free"  # Free约束，无代数表达式
    EQUATION = "equation"  # 只含等式
    INEQUALITY = "inequality"  # 只含不等式
    MIXED = "mixed"  # 混合约束


def form_fact(geo_predicate: str, paras_list: list):
    """
    Transform geometric fact from logic form to string

    Args:
        predicate (str): 几何谓词名, e.g. "Point", "PointOnLine", etc.
        paras_list (list): 谓词的参数列表, e.g. ['A', 'l']

    Returns:
        result (str): 完整的几何事实句子, e.g. "PointOnLine(A,l)"
    """
    return f"{geo_predicate}({','.join(paras_list)})"


class GeoCfgGenerator:
    def __init__(self, gdl, seed=None):
        """
        Geometric Configuration Generator, 几何构型生成器

        Args:
            gdl (dict): 尚未解析的GDL
            seed (int, optional): 随机种子，如果为 None 则使用时间戳生成
            [Removed] parsed_gdl (dict): 解析后的GDL
        """
        # 创建独立的 Random 实例控制生成时使用的种子
        if seed is None:
            seed = time.time_ns() ^ (id(self) & 0xFFFF)  # 结合时间戳和对象ID生成种子
            self.seed = seed
        else:
            self.seed = seed
        self.random_gcg = Random(seed)

        # 几何实体的参数空间 (实际上所有实体共享参数空间)
        self.letters_point = entity_letters
        self.letters_line = entity_letters
        self.letters_circle = entity_letters

        # 已构造的几何实体(参数), e.g. ["A", "a", ...]
        self.points = []  # (list): 当前已经构造的点(参数)
        self.lines = []
        self.circles = []
        # 几何实体的 sympy.symbol, e.g. {'A': ['A.x', 'A.y']}
        self.sym_of_ent = {}

        # GDL all based on parse_gdl()
        self.parsed_gdl = parse_gdl(gdl)  # (dict): 解析后的GDL

        self.gdl_constraints = self._get_basic_constraints()  # 过滤得到 basic constraints
        # --- REMOVED ---
        # self.gdl = gdl  # (dict): 尚未解析的GDL
        # self.gdl_constraints = gdl["Relations"]
        # self.parsed_gdl_constraints = self.parsed_gdl["Relations"]

        # 当前构型状态下已经解锁的约束
        self.active_constraints = {"Point": [], "Line": [], "Circle": []}
        # 待解锁的约束
        self.pending_constraints = {"Point": [], "Line": [], "Circle": []}

        # 约束相关属性
        self.all_constraints = set()  # 已生成的约束 避免生成完全等价的constraints
        self.all_cdls = []  # (list) of (str), 生成几何构型的所有CDL句子

        # TODO: !! 约束选择控制
        self.constrained_ratio_target = 0.8  # 整体构型的约束比例
        self.free_entity_limit = 6  # 能够生成的free几何实体个数
        self.free_constraint_num = 0
        self.eq_constraint_num = 0
        self.ineq_constraint_num = 0

        # 实体自由度映射
        self.entity_dof_map = {"Point": 2, "Line": 2, "Circle": 3}
        # 实体的 Jacobian 秩 {entity_para: rank}
        self.entity_ranks = {}
        # 约束类型缓存（预计算）{constraint_name: ConstraintType}
        self.constraint_type_cache = {}

        # 几何构型 用于阶段性验证CDLs
        self.gc = geocfg(self.parsed_gdl)

        # 为高效获取约束预创建索引
        self._build_constraint_index()

        # 元数据属性
        self.gdl_file = None  # 使用的GDL文件名（需在外部设置）
        self.creation_time = datetime.now()  # 生成时间

        # 验证相关属性
        self.validation_result = None  # 验证结果 {cdl_index: bool}
        self.validation_time = None  # 验证时间
        self.validation_duration_ms = None  # 验证耗时（毫秒）

        logger.debug("[Inited] Geo Cfg Generator!")

    """↑------- Geo Cfg Generator Initialization ------↑"""
    """↓--------- Current GC Generation Status --------↓"""

    @property
    def points_num(self):
        """已构造的点的个数"""
        return len(self.points)

    @property
    def lines_num(self):
        """已构造的线的个数"""
        return len(self.lines)

    @property
    def circles_num(self):
        """已构造的圆的个数"""
        return len(self.circles)

    @property
    def all_entities(self):
        """所有已构造的几何实体的参数列表"""
        return self.points + self.lines + self.circles

    # @property
    # def all_entities_num(self):
    #     """所有已构造的几何实体的总数"""
    #     return len(self.all_entities)

    @property
    def all_entity_dof(self):
        """所有已构造的几何实体的最大自由度 Degree of Freedom"""
        points_dof = self.points_num * 2
        lines_dof = self.lines_num * 2
        circles_dof = self.circles_num * 3

        return sum(points_dof, lines_dof, circles_dof)

    # @property
    # def all_constraints_num(self):
    #     """所有已选择约束的个数"""
    #     return self.eq_constraint_num + self.ineq_constraint_num + self.free_constraint_num

    @property
    def curr_constraint_ratio(self):
        """当前整体构型的约束程度, 基于各实体的Jacobian秩"""
        if not self.all_entities:
            # 没有任何实体 限制程度为0
            return 0.0

        total_ratio = 0.0
        for entity_para in self.all_entities:
            geo_ent_type = self._get_entity_type(entity_para)
            dof = self.entity_dof_map[geo_ent_type]
            rank = self.entity_ranks.get(entity_para, 0)
            total_ratio += rank / dof

        return total_ratio / len(self.all_entities)

    def _get_entity_type(self, entity_para: str) -> str:
        """
        根据实体参数获取实体类型

        Args:
            entity_para: 实体参数，e.g. "A", "a", "Γ"

        Returns:
            str: 实体类型，"Point", "Line", 或 "Circle"
        """
        if entity_para in self.points:
            return "Point"
        elif entity_para in self.lines:
            return "Line"
        elif entity_para in self.circles:
            return "Circle"
        else:
            raise ValueError(f"Unknown entity parameter: {entity_para}")

    def _update_entity_rank(self, entity_para: str, rank: int):
        """
        更新实体的 Jacobian 秩

        Args:
            entity_para: 实体参数，e.g. "A"
            rank: 当前 Jacobian 秩
        """
        self.entity_ranks[entity_para] = rank

        # 检查是否为 Free 实体（rank == dof）
        geo_ent_type = self._get_entity_type(entity_para)
        dof = self.entity_dof_map[geo_ent_type]

        # 更新 Free 计数器
        if rank == dof:
            # 检查之前是否已经是 Free 状态
            if entity_para not in self.entity_ranks or self.entity_ranks.get(entity_para, 0) < dof:
                self.free_constraint_num += 1
                logger.info(f"Entity {geo_ent_type}({entity_para}) set as Free entity (rank={rank}, dof={dof})")

    def _test_gc_gen_tree_structure(self):
        """
        GeoCfg Generation Tree structure.

        Tree level:
            GeoCfg Gen tree is a list of geo entities with their relations(constraints),
            their positions in the list(tree) imply their generation orders.

        Geo Entity level:
            Every Geo entity is stored as a dict:
                key: a completed string of geo entity.
                value: its chosen relations(constraints), stored as dict.

        Constraint level:
            Every geo entity has a chosen constraints list,
            constraints' positions in the list imply their generation orders.

            Every constraint has:
                key: relation(constraint) name.
                value:
                    "chosen_parameters" (list): chosen parameter combination for the constraint.
                    "candidate_paras" (list): all correct (no duplicate) parameter combinations.

        Parameters level:
            chosen_parameters: only choose one, so it's a list of letters.
            candidate_paras: may exist many combinations, so it's a list of list of letters.

        Returns:
            gc_gen_tree (list):
            [
                {...},
                {
                    "Line(a)":
                    [
                        {
                            "PointOnLine":
                            {
                                "chosen_parameters": ["B", "a"],
                                "candidate_paras": [["C", "a"]]
                            }
                        },
                        {
                            "PointOnLine":
                            {
                                "chosen_parameters": ["B", "a"],
                                "candidate_paras": [
                                    ["H", "a"],
                                    ["L", "a"]
                                ]
                            }
                        }
                    ]
                },
                {...}
            ]
        """
        # it should be self.gc_gen_tree
        logger.debug("test gc gen tree structure!")

    @property
    def status(self):
        """
        返回生成构型当前的状态

        Returns:
            cfg_status (dict): 包含当前生成状态的几何实体和CDL句子
        """
        # Debug
        logger.debug("GeoCfgGenerator status:")
        logger.debug(f"Points:{self.points}")
        logger.debug(f"Lines:{self.lines}")
        logger.debug(f"Circles:{self.circles}\n")

        logger.debug("All CDLs:")
        for index, c in enumerate(self.all_cdls):
            logger.debug(f"{index+1}. {c}")

        # TODO: Directly return the gc_gen_tree?

        return {
            "Points": self.points,
            "Lines": self.lines,
            "Circles": self.circles,
            "All_Constraints": self.all_constraints,
            "All_CDLs": self.all_cdls,
        }

    def clear_status(self):
        """清空几何构型生成器的生成状态"""
        # 重置 Random seed
        seed = time.time_ns() ^ (id(self) & 0xFFFF)  # 结合时间戳和对象ID生成种子
        self.random_gcg = Random(seed)

        self.points = []
        self.lines = []
        self.circles = []
        self.sym_of_ent = {}

        # 重置已解锁的约束
        self.active_constraints = {"Point": [], "Line": [], "Circle": []}
        # 重置待解锁的约束
        self.pending_constraints = {"Point": [], "Line": [], "Circle": []}
        # 重置生成的CDL
        self.all_constraints.clear()  # 清空约束集合
        self.all_cdls = []

        # 重置约束状态
        self.entity_ranks.clear()
        self.free_constraint_num = 0
        self.eq_constraint_num = 0
        self.ineq_constraint_num = 0

        # 重置几何构型
        self.gc = geocfg(self.parsed_gdl)

        # 重新创建约束索引
        self._build_constraint_index()

        logger.warning("GeoCfgGenerator status Cleared!")

    """↑--------- Current GC Generation Status --------↑"""
    """↓------- Geo Entity Parameter Generation -------↓"""

    def random_point_para(self, k=1):
        """随机生成k个点的参数字母"""
        return self.random_gcg.sample(self.letters_point, k)

    def random_line_para(self, k=1):
        """随机生成k个线的参数字母"""
        return self.random_gcg.sample(self.letters_line, k)

    def random_circle_para(self, k=1):
        """随机生成k个圆的参数字母"""
        return self.random_gcg.sample(self.letters_circle, k)

    """↑------- Geo Entity Parameter Generation -------↑"""
    """↓------------ Geo Entity Generation ------------↓"""

    def generate_one_geo_entity(self):
        """
        随机生成一个几何元素实体,

        更新"激活"的约束列表: self.active_constraints,

        更新实体对应的参数字典: self.sym_of_ent.

        Returns:
            geo_entity (str): 生成的几何实体及其参数,
            e.g. "Point(A)", "Line(a)", "Circle(Γ)"
        """
        # 1. Choose Geometric Entity Predicate from [Point, Line, Circle]
        # Geo Entity predicate is also its Type
        # geo_ent_predicate = self.random_gcg.choice(['Point', 'Line', 'Circle'])

        # utilize parsed_gdl
        base_entity_list = list(self.parsed_gdl["Entities"].keys())
        logger.debug(f"base Entity in parsed_gdl: {base_entity_list}")
        geo_ent_type = self.random_gcg.choice(base_entity_list)

        # TODO: generate more than 1 entity

        # 2. Set the parameter for the chosen entity
        geo_ent_para = self._gen_para_for_entity(geo_ent_type)

        # 3. When a new geo entity is generated, update active constraints
        self._update_active_constraints()
        logger.debug(f"current active constraints:\n{self.active_constraints}")

        self._set_symbols(geo_ent_type, geo_ent_para)
        # structure return as completed string, e.g. "Point(A)"
        geo_entity = form_fact(geo_ent_type, geo_ent_para)

        logger.info(f"+ Generated Geo Entity: {geo_entity}")

        return geo_entity

    def gen_geo_entity_of_type(self, geo_ent_type: str):
        """
        生成一个指定类型的几何实体

        Args:
            geo_entity_type (str): 指定几何实体的类型, e.g. Point, Line, Circle

        Returns:
            geo_entity (str): 完整的几何实体, e.g. "Point(A)", "Line(a)", "Circle(Γ)"
        """
        base_entity_list = list(self.parsed_gdl["Entities"].keys())
        if geo_ent_type not in base_entity_list:
            raise ValueError(f"Unknown Geo Entity type: {geo_ent_type}!")

        # 生成指定类型的几何实体的参数
        geo_ent_para = self._gen_para_for_entity(geo_ent_type)
        # 设置其对应的 sympy 符号
        self._set_symbols(geo_ent_type, geo_ent_para)
        geo_entity = form_fact(geo_ent_type, geo_ent_para)

        return geo_entity

    def _gen_para_for_entity(self, geo_entity_type: str):
        """
        根据几何实体谓词名 生成其对应的参数 避免生成已存在的参数

        Args:
            geo_entity_type (str): 几何实体的类型, e.g. "Point", "Line", "Circle"

        Returns:
            ent_para (str): 几何实体的参数, e.g. "A", "a", "α"
        """
        retry_count = 0
        # Avoid infinite loop in case of points pool exhaustion
        max_retries = len(entity_letters)

        # Max try num Error
        try_error = f"Max retry count reached while generating {geo_entity_type}!"

        # Point
        if geo_entity_type == "Point":
            # Generate a new point not already in self.points
            # e.g. A (For Point(A))
            p_para = self.random_point_para(k=1)[0]
            # 因为几何实体的参数是共享的，因此需要检查总体的参数空间
            while p_para in self.all_entities:
                # Re-generate if Point already exists
                p_para = self.random_point_para(k=1)[0]

                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(try_error)

            # Add the new point to self.points
            self.points.append(p_para)
            return p_para

        # Line
        elif geo_entity_type == "Line":
            # Generate a new line not already in self.lines
            # e.g. a (For Line(a))
            l_para = self.random_line_para(k=1)[0]
            while l_para in self.all_entities:
                # Re-generate if Line already exists
                l_para = self.random_line_para(k=1)[0]

                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(try_error)

            # Add the new line to self.lines
            self.lines.append(l_para)
            return l_para

        # Circle
        elif geo_entity_type == "Circle":
            # Generate a new circle not already in self.circles
            # e.g. α (For Circle(α))
            c_para = self.random_circle_para(k=1)[0]
            while c_para in self.all_entities:
                c_para = self.random_circle_para(k=1)[0]

                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(try_error)

            self.circles.append(c_para)
            return c_para

        else:
            raise ValueError(f"Unknown entity type: {geo_entity_type}")

    def _set_symbols(self, geo_ent_type: str, geo_ent_para: str):
        """
        设置几何实体 geo_entity 的 SymPy 符号

        Args:
            geo_ent_type (str): 几何实体类型
            geo_ent_para (str): 几何实体参数
        """
        # symbols of current geo parameter
        curr_entity_symbols = []
        found_symbol = False
        for attr_name, attr_info in self.parsed_gdl["Attributions"].items():
            # 检查测度属性的 ee_check 是否与当前实体类型匹配

            # e.g. 'geometric_constraints': ( ('Circle', ('O',) ), )
            logger.debug(attr_info["geometric_constraints"][0])

            if (
                len(attr_name) == 1
                and "geometric_constraints" in attr_info
                and len(attr_info["geometric_constraints"]) == 1
                and attr_info["geometric_constraints"][0][0] == geo_ent_type
            ):
                found_symbol = True
                logger.debug(f"Valid Attribution: {attr_name}")

                # 获取符号并创建当前实体的 sympy 符号
                curr_entity_symbols.append(symbols(f"{geo_ent_para}.{attr_name}"))
                logger.debug(f"curr ent symbols: {curr_entity_symbols}")

        # 找到了需要设置的符号
        if found_symbol:
            # set sympy symbol of geo_entity
            self.sym_of_ent[geo_ent_para] = curr_entity_symbols
            logger.debug(f"{geo_ent_para} set symbol: {curr_entity_symbols}")
        else:
            # 没有找到任何匹配的测度 及其符号
            error_info = f"No Measures for entity {geo_ent_type}!"
            logger.error(error_info)
            raise ValueError(error_info)

    def _parse_geo_entity(self, geo_entity: str):
        """Parse geo entity into logic form"""
        # format check already implemented in tools._parse_geometric_fact()
        geo_ent_type, geo_ent_para = _parse_geometric_fact(geo_entity)

        # original geo_ent_para is a list, e.g. ['A']
        geo_ent_para = geo_ent_para[0]

        return geo_ent_type, geo_ent_para

    def _check_format_geo_entity(self, geo_entity: str):
        """Check if the given geo_entity is valid"""
        geo_ent_type, geo_ent_para = self._parse_geo_entity(geo_entity)

        # check entity type
        valid_geo_ent_type = self.parsed_gdl["Entities"].keys()
        if geo_ent_type not in valid_geo_ent_type:
            error_info = f"Unknown geo entity type: {geo_ent_type}"
            logger.error(error_info)
            raise ValueError(error_info)

        # check parameter length
        if len(geo_ent_para) != 1:
            error_info = f"Temporarily Wrong entity parameter: {geo_ent_para}"
            logger.error(error_info)
            raise ValueError(error_info)

        # parameter validation (existed)
        curr_entity_list = getattr(self, f"{geo_ent_type.lower()}s")
        logger.debug(f"Constructed {geo_ent_type} Entity list: {curr_entity_list}")
        logger.debug(f"current geo ent para: {geo_ent_para}")
        if geo_ent_para not in set(curr_entity_list):
            error_info = f"Unconstructed entity parameter: {geo_ent_para}"
            logger.error(error_info)
            raise ValueError(error_info)

        return True

    def _get_current_entities(self) -> dict:
        """获取当前已构造的所有几何类型的实体(参数)列表"""
        curr_ent_paras = {}

        for ent_type in ["Points", "Lines", "Circles"]:
            curr_ent_paras[ent_type] = list(getattr(self, ent_type.lower()))

        return curr_ent_paras

    """↑------------ Geo Entity Generation ------------↑"""
    """↓------- Geometry Constraints Generation -------↓"""

    def _get_basic_constraints(self):
        """
        获取 self.parsed_gdl["Relations"] 中 basic relations 作为约束

        同时排除一些 basic constraints 中不需要的约束

        Returns:
            base_constraints (dict): parsed_gdl 中初步筛选后的 constraints, e.g.:
            {
                '...',
                'EqualDistance':
                    {
                        'algebraic_forms':
                        (
                            ('Eq', (-A.x + B.x)**2 + (-A.y + B.y)**2 - (-C.x + D.x)**2 - (-C.y + D.y)**2),
                            '&',
                            ('Ueq', (-A.x + B.x)**2 + (-A.y + B.y)**2)
                        ),
                        'geometric_constraints':
                        (
                            ('Point', ('A',)),
                            ('Point', ('B',)),
                            ('Point', ('C',)),
                            ('Point', ('D',))
                        ),
                        'paras': ('A', 'B', 'C', 'D')
                    },

                'SegmentEqualSegment':
                    {
                        'type': 'basic',
                        'paras': ('A', 'B', 'C', 'D')
                        'ee_checks': ('Point',
                                      'Point',
                                      'Point',
                                      'Point'),
                        'constraints': (('Eq',
                                        (-A.x + B.x)**2 + (-A.y + B.y)**2 - (-C.x + D.x)**2 - (-C.y + D.y)**2),),
                        'extends': (('Equation', AB.dpp - CD.dpp),),
                        'multiple_forms': ( (1, 0, 2, 3),
                                            (0, 1, 3, 2),
                                            (1, 0, 3, 2),
                                            (2, 3, 0, 1),
                                            (2, 3, 1, 0),
                                            (3, 2, 1, 0),
                                            (3, 2, 0, 1)),
                        'implicit_entities': { 'Circle': (),
                                               'Line': (),
                                               'Point': ()},
                        'implicit_extends': (),
                    },
                '...',
            }
        """
        # 需要通过约束名排除的约束集合
        exclude_constraints = {
            "SamePoint",
            "SameLine",
            "SameCircle",
            "AngleBisector",
            "Triangle",
            "CongruentTriangle",
        }

        basic_constraints = {}

        for constraint_name, constraint_info in self.parsed_gdl["Relations"].items():
            # REMOVED - Only basic type constraints (Relations) with exclusion
            if constraint_name not in exclude_constraints:
                basic_constraints[constraint_name] = constraint_info
                logger.debug(f"{constraint_name} added to base constraint")
                # pprint(basic_constraints[constraint_name])

        logger.debug(f"basic constraints length = {len(basic_constraints.keys())}")

        return basic_constraints

    def _count_eq_ineq_in_tree(self, expr_tree: tuple) -> tuple[int, int]:
        """
        递归统计表达式树中的等式和不等式数量

        Args:
            expr_tree: 代数表达式树

        Returns:
            (eq_count, ineq_count): 等式数量和不等式数量
        """
        expr_len = len(expr_tree)

        if expr_len == 0:
            return 0, 0
        elif expr_len == 1:
            return 0, 0
        elif expr_len == 2:
            op = expr_tree[0]
            if op == "Eq":
                return 1, 0
            elif op == "!":
                return 0, 0
            else:
                return 0, 1
        elif expr_len == 3:
            connector = expr_tree[1]
            if connector in ["&", "|"]:
                left_eq, left_ineq = self._count_eq_ineq_in_tree(expr_tree[0])
                right_eq, right_ineq = self._count_eq_ineq_in_tree(expr_tree[2])
                return left_eq + right_eq, left_ineq + right_ineq

        return 0, 0

    def _classify_constraint_type(self, constraint_name: str) -> ConstraintType:
        """
        分类约束类型

        Args:
            constraint_name: 约束名称

        Returns:
            ConstraintType: 约束类型
        """
        algebraic_forms = self.gdl_constraints[constraint_name]["algebraic_forms"]

        if len(algebraic_forms) == 0:
            return ConstraintType.FREE

        # 递归分析表达式树
        eq_count, ineq_count = self._count_eq_ineq_in_tree(algebraic_forms)

        if eq_count > 0 and ineq_count > 0:
            return ConstraintType.MIXED
        elif eq_count > 0:
            return ConstraintType.EQUATION
        else:
            return ConstraintType.INEQUALITY

    def _build_constraint_index(self):
        """
        根据self.gdl_constraints["ee_checks"]中所需的实体类型和数量，预计算并构建约束索引

        将所有约束根据"触发类型"计算"剩余需求", 并存入 pending_constraints
        """
        # 预计算约束类型
        for constraint_name, constraint_data in self.gdl_constraints.items():
            self.constraint_type_cache[constraint_name] = self._classify_constraint_type(constraint_name)
        logger.info(f"Precomputed constraint types for {len(self.constraint_type_cache)} constraints")

        # 遍历所有约束 获取约束名和约束相关内容
        for constraint_name, constraint_info in self.gdl_constraints.items():
            if not self._check_constraint_info(constraint_name):
                break

            # 获取约束的"ee_checks"
            constraint_geo_constraints = constraint_info.get("geometric_constraints", [])
            if not constraint_geo_constraints:
                logger.error(f"No geometric_constraints for {constraint_name}")
                continue

            # 1. 统计该约束的总需求
            total_reqs = {"Point": 0, "Line": 0, "Circle": 0}
            involved_types = set()  # 记录当前约束涉及到的几何实体集合
            for ee_check in constraint_geo_constraints:
                # e.g. ee_check = "("Point", ('A',))"
                entity_type = ee_check[0]
                # 如果是需要的实体类型
                if entity_type in total_reqs:
                    total_reqs[entity_type] += 1
                    involved_types.add(entity_type)

            # 2. 为每种触发类型计算"剩余需求" (Pool Requirements)
            for trigger_ent_type in involved_types:
                # 复制一份总需求
                pool_reqs = total_reqs.copy()

                # 不需要-1 因为触发实体和当前实体都会被用于填充约束的参数
                # pool_reqs[trigger_ent_type] -= 1

                # 构造约束项
                constraint_item = {
                    "c_name": constraint_name,
                    "pool_reqs": pool_reqs,  # 只有满足这些池中数量，该约束才解锁
                    "raw_geometric_constraints": constraint_geo_constraints,
                }

                # 存入待解锁队列
                self.pending_constraints[trigger_ent_type].append(constraint_item)

        # 3. 初始状态下可能有部分约束直接满足（例如只需0个实体的约束，虽然少见）
        # 执行一次空更新以初始化 active_constraints 列表
        self._update_active_constraints()

        # Debug Point triggers
        # logger.debug("Point triggers:", len(self.constraints_by_trigger["Point"]), "constraints")

    def _update_active_constraints(self):
        """
        初始状态以及每次生成新几何实体时，更新当前构型状态下可用的约束列表:
        1. 遍历 self.pending_constraints
        2. 将满足当前实体数量条件的约束移入 self.active_constraints
        随着过程进行，pending_constraints 列表会越来越短，检查速度越来越快。
        """
        # 获取当前池中实体的数量
        curr_gcgen_states = {
            "Point": self.points_num,
            "Line": self.lines_num,
            "Circle": self.circles_num,
        }

        # 遍历三种类型的待解锁队列
        for trigger_type in ["Point", "Line", "Circle"]:
            # 记录遍历之后仍然尚未解锁的constraint
            still_pending = []

            # 检查每一个待定约束
            for constraint in self.pending_constraints[trigger_type]:
                reqs = constraint["pool_reqs"]

                # 检查约束需要的几何实体组合数量是否满足
                is_satisfied = True
                for ent_type, count_needed in reqs.items():
                    # 任意一个不满足要求 就不能解锁
                    if curr_gcgen_states[ent_type] < count_needed:
                        is_satisfied = False
                        break

                if is_satisfied:
                    # 条件满足 解锁! 加入 active 列表
                    self.active_constraints[trigger_type].append(constraint["c_name"])
                    logger.debug(f"{constraint['c_name']} Unlocked for {trigger_type}!")
                else:
                    # 不满足，保留在 pending 中
                    still_pending.append(constraint)

            # 更新 pending 列表（移除已解锁的约束）
            self.pending_constraints[trigger_type] = still_pending

    def _remove_ineq_constraints(self, geo_entity: str, base_avail_constraints: list) -> list:
        """
        移除只含不等式的约束

        Args:
            geo_entity (str): 几何实体
            base_avail_constraints (list): 基础可用约束列表

        Returns:
            filtered_constraint (list): 过滤只包含不等式约束后的约束列表
        """
        self._check_format_geo_entity(geo_entity)

        result_avail_constraints = []
        removed_count = 0

        for constraint_name in base_avail_constraints:
            constraint_type = self.constraint_type_cache.get(constraint_name, None)
            if constraint_type == None:
                logger.error(f"No type for {constraint_name}!")

            if constraint_type != ConstraintType.INEQUALITY:
                result_avail_constraints.append(constraint_name)
            else:
                removed_count += 1
                logger.debug(f"Removed Inequality-only constraint: {constraint_name}")

        if removed_count > 0:
            logger.info(f"Removed {removed_count} Inequality-only constraints for {geo_entity}")

        return result_avail_constraints

    def _remove_free_constraint(self, geo_entity: str, base_avail_constraints: list) -> list:
        """
        移除Free约束

        Args:
            geo_entity (str): 几何实体
            base_avail_constraints (list): 基础可用约束列表

        Returns:
            filtered_constraint (list): 过滤自由约束后的约束列表
        """
        self._check_format_geo_entity(geo_entity)

        geo_ent_type, geo_ent_para = _parse_geometric_fact(geo_entity)
        free_constraint = "Free" + geo_ent_type

        if free_constraint in base_avail_constraints:
            logger.info(f"Removed constraint {free_constraint}!")

        # Remove Free type constraint
        result_avail_constraints = [c for c in base_avail_constraints if c != free_constraint]

        return result_avail_constraints

    def _update_avail_constraints(self, geo_entity: str) -> list:
        """
        根据当前的生成状态, 更新当前的可用约束列表

        Args:
            geo_entity: 几何实体
            curr_avail_constraints: 当前可用的约束
            curr_constraints: 当前已经成功生成的约束

        Returns:
            list: 更新后的可用约束列表
        """
        self._check_format_geo_entity(geo_entity)

        # 提取当前实体的类型和参数 当前类型为 trigger type
        geo_ent_type, geo_ent_para = self._parse_geo_entity(geo_entity)

        # 直接从预计算的 self.active_constraints 中获取该类型所有的可用约束
        # 这些约束已经通过 self._update_active_constraints() 验证过，满足池中实体数量要求
        # 并且通过 self._build_constraint_indices() 保证了必然包含该实体类型
        result_avail_constraints = self.active_constraints[geo_ent_type].copy()

        # Filter [Free] constraint
        if self.free_constraint_num >= self.free_entity_limit:
            result_avail_constraints = self._remove_free_constraint(geo_entity, result_avail_constraints)

        # Filter [Ineq-only] constraint
        if self.curr_constraint_ratio < self.constrained_ratio_target:
            result_avail_constraints = self._remove_ineq_constraints(geo_entity, result_avail_constraints)

        return result_avail_constraints

    def _check_constraint_info(self, constraint_name: str):
        """
        检查约束的属性是否齐全,

        attributions needed:
        {
            "geometric_constraints",
            "algebraic_forms",
            "paras",
        }
        """
        constraint_info = self.gdl_constraints.get(constraint_name, {})
        info_keys_reqr = {
            "geometric_constraints",
            "algebraic_forms",
            "paras",
        }
        attr_info = set(constraint_info.keys())

        if not info_keys_reqr.issubset(attr_info):
            missing_keys = info_keys_reqr - attr_info
            error_msg = (
                f"Constraint '{constraint_name}' is missing required attributes:\n"
                f"{missing_keys}\n"
                "Please Check GDL file!\n"
            )
            raise ValueError(error_msg)

        return True

    def _get_all_para_combs(self, geo_ent_type: str, geo_ent_para: str, constraint_name: str):
        """
        根据当前的几何实体和选中的约束，随机尝试该约束的参数，返回尽可能多的组合(笛卡尔积)以及对应的参数映射

        Args:
            geo_ent_type (str): 几何实体类型, e.g. "Point"
            geo_ent_para (list): 几何实体的参数, e.g. 'A'
            constraint (str): 约束名称, e.g. "PointOnLine", "PointInAngle"

        Returns:
            constraint_paras (str): 约束的参数字符串, e.g. "A,B,C"
            dft_to_curr_lttr (dict): 默认参数->当前参数的映射关系, e.g. {"A": "a"}
        """
        # 约束合法性检测
        if not self._check_constraint_info(constraint_name):
            return None

        # 获取当前被约束几何实体的类型和参数
        # Extract real geo_ent_para
        # geo_ent_para = geo_ent_para[0]

        constraint_info = self.gdl_constraints.get(constraint_name)
        # 获取约束的 geometric_constraints (参数类型与位置对应关系)
        constraint_geo_constraints = constraint_info.get("geometric_constraints", [])

        # 先通过几何实体类型 找出当前几何实体在该约束参数中的所有可能位置
        possible_pos = []
        for index, entity_with_paras in enumerate(constraint_geo_constraints):
            # e.g. entity_with_paras  = ('Point', ('A', ))
            entity_type = entity_with_paras[0]
            if entity_type == geo_ent_type:
                possible_pos.append(index)

        if not possible_pos:
            # 检查是否存在合法位置
            logger.error(f"No {geo_ent_type} needed for {constraint_name}!")
            return None

        # 获取约束的默认参数
        # e.g. PointOnLine["paras"] -> ('A', 'l')
        default_paras = constraint_info.get("paras", ())  # tuple for immutable
        if not default_paras:
            logger.error(f"No default parameters for {constraint_name}!")
            return None

        # TODO: 构造当前可用实体池
        curr_entities = self._get_current_entities()
        # 去除当前实体 避免重复选择当前实体 (Change in future)
        if geo_ent_para in curr_entities[f"{geo_ent_type}s"]:
            curr_entities[f"{geo_ent_type}s"].remove(geo_ent_para)

        # 所有可能的参数组合 若存在合法参数 会加入(默认字符->实际字符)的映射
        para_combs_with_mapping = []

        # 遍历当前实体所有可能的位置
        for selected_pos in possible_pos:
            # 为当前选择的位置创建参数占位符列表
            constraint_params_template = [None] * len(constraint_geo_constraints)
            constraint_params_template[selected_pos] = geo_ent_para

            # 构建除了当前实体位置外的其他位置列表
            other_positions = []
            for i in range(len(constraint_geo_constraints)):
                if i != selected_pos:
                    other_positions.append(i)

            # 收集每个位置可选的实体列表
            position_options = []
            for pos in other_positions:
                entity_with_paras = constraint_geo_constraints[pos]
                ent_type_key = f"{entity_with_paras[0]}s"
                position_options.append(curr_entities[ent_type_key])

            # 如果有任何位置没有可选实体 组合发生错误 跳过
            if any(len(options) == 0 for options in position_options):
                print(f"No available entities for constraint {constraint_name}!")
                continue

            # 使用笛卡尔积生成所有组合
            for combination in itertools.product(*position_options):
                # 创建新的参数列表副本
                constraint_params = constraint_params_template.copy()

                # 建立参数映射关系 先构建当前实体的映射
                dft_to_curr_lttr = {default_paras[selected_pos]: geo_ent_para}

                # 再填充其他位置的参数和映射
                for i, pos in enumerate(other_positions):
                    param_value = combination[i]
                    constraint_params[pos] = param_value
                    dft_to_curr_lttr[default_paras[pos]] = param_value

                # 添加到结果字典列表
                para_combs_with_mapping.append({"parameters": constraint_params, "mapping": dft_to_curr_lttr})

        if para_combs_with_mapping:
            # Debug valid para combinations
            logger.debug("++ All Possible para combs and mapping:")
            logger.debug(para_combs_with_mapping)
            # pprint(para_combs_with_mapping)
        else:
            logger.error(f"No Valid para comb for {constraint_name}!")

        return para_combs_with_mapping

    def _and_or_not(self):
        """
        生成约束之间的链接关系, e.g. ['&', '|', '~']

        支持'or'和'not'时记得修改

        Returns:
            delimiter (str): and/or/not, e.g. '&', '|', '~'
        """
        # 随机选择
        # d_str = random.choice['&', '|', '~']

        # d_str = self.get_avail_delimeter() maybe

        # return d_str
        return "&"

    def _form_cdl(self, geo_entity: str, constraints_list: list):
        """
        将几何实体与其约束列表组成完整的CDL语句。

        Args:
            geo_entity (str): 几何实体, e.g. "Point(A)"
            constraints_list (list): 约束连接形成的字符串, e.g. "PointOnLine(A,l)", "PointOnLine(A,b)"

        Returns:
            cdl (str): 一句CDL语句
        """
        constraints_str = self._and_or_not().join(constraints_list)
        return f"{geo_entity}:{constraints_str}"

    def _extract_entity_params(self, expression, geo_entity_para: str) -> list:
        """
        从方程表达式中提取涉及当前实体的参数

        Args:
            expression: 方程表达式（SymPy表达式）
            geo_entity_para (str): 当前实体参数, e.g. 'A'

        Returns:
            list: 方程中涉及当前实体的参数列表, e.g. ['A.x', 'A.y']
        """
        entity_params = []

        # 遍历表达式中的所有自由符号
        curr_ent_sym = self.sym_of_ent[geo_entity_para]
        logger.info(f"Current entity sym: {curr_ent_sym}")

        for sym in expression.free_symbols:
            sym_str = str(sym)  # e.g., "A.x", "l.k", "O.r"
            sym_entity = sym_str.split(".")[0]  # e.g., "A", "l", "O"

            # 检查是否涉及当前实体
            if sym_entity == geo_entity_para:
                entity_params.append(sym_str)

        return entity_params

    def _analyze_constraint_for_entity(self, constraint_name: str, geo_entity_para: str) -> ConstraintType:
        """
        分析约束对当前实体的约束类型（动态分类）

        Args:
            constraint_name (str): 约束名称
            geo_entity_para (str): 当前实体参数, e.g. 'A'

        Returns:
            ConstraintType: 对当前实体的约束类型
        """
        logger.info(f"Analyzing Constraint {constraint_name}")
        algebraic_forms = self.gdl_constraints[constraint_name]["algebraic_forms"]
        logger.debug(f"algb expression:\n\t{algebraic_forms}")
        logger.debug(f"length: {len(algebraic_forms)}")

        if constraint_name.startswith("Free"):
            return ConstraintType.FREE

        # 递归分析表达式树，统计等式和不等式数量
        eq_count, ineq_count = self._count_eq_ineq_in_tree(algebraic_forms)

        if eq_count == 0 and ineq_count > 0:
            # 没有等式，只有不等式
            return ConstraintType.INEQUALITY

        # 有等式，需要分析等式是否约束当前实体
        has_entity_constraint = False
        has_other_entity_constraint = False

        # 递归检查表达式树中的等式
        def check_expr_tree(expr_tree):
            nonlocal has_entity_constraint, has_other_entity_constraint

            expr_len = len(expr_tree)

            if expr_len == 0:
                return None
            elif expr_len == 1:
                error_msg = f"Impossible expression length: {expr_len}!"
                logger.error(error_msg)
                logger.error(f"expression: {expr_tree}")
                return None
            elif expr_len == 2:
                operator = expr_tree[0]
                logger.info(f"Operator: {operator}")
                if operator == "Eq":
                    # 这是一个等式，检查是否涉及当前实体
                    expression = expr_tree[1]
                    dft_to_curr_lttr = {self.gdl_constraints[constraint_name]["paras"][0]: geo_entity_para}
                    logger.info(f"Expression before: {expression}")
                    logger.info(f"Mapping: {dft_to_curr_lttr}")
                    expression = _replace_expr(expression, dft_to_curr_lttr)
                    logger.info(f"Expression replaced: {expression}")

                    entity_params = self._extract_entity_params(expression, geo_entity_para)

                    if entity_params:
                        # 等式涉及当前实体
                        has_entity_constraint = True

                        # 检查是否还涉及其他实体
                        for sym in expression.free_symbols:
                            sym_str = str(sym)
                            sym_entity = sym_str.split(".")[0]
                            logger.info(f"symbol of entity {geo_entity_para} is {sym_entity}")
                            if sym_entity != geo_entity_para:
                                has_other_entity_constraint = True
                                break
                elif operator == "!":
                    # NOT 操作，递归处理子表达式
                    check_expr_tree(expr_tree[1])
            elif expr_len == 3:
                connector = expr_tree[1]
                if connector in ["&", "|"]:
                    # AND 或 OR 逻辑，递归处理两边
                    check_expr_tree(expr_tree[0])
                    check_expr_tree(expr_tree[2])

        check_expr_tree(algebraic_forms)

        if not has_entity_constraint:
            # 等式不涉及当前实体
            return ConstraintType.INEQUALITY
        elif not has_other_entity_constraint:
            # 等式只涉及当前实体
            return ConstraintType.EQUATION
        else:
            # 等式同时涉及当前实体和其他实体
            if ineq_count > 0:
                return ConstraintType.MIXED
            else:
                # 只有等式，但等式同时涉及多个实体
                return ConstraintType.EQUATION

    def _is_duplicate(self, constraint: str):
        """
        检查给定的约束是否重复, 给出 Debug Info.

        Args:
            constraint (str): 带有参数的约束 用于检测重复

        Returns:
            bool: 是否重复
        """
        if constraint in self.all_constraints:
            logger.info(f"[Duplicate constraint] {constraint}!\n")
            return True
        else:
            return False

    def _update_exclude_index(
        self,
        original_exclude_comb_index: set,
        rest_para_combs_with_mapping: list,
        curr_ent_para: str,
        curr_comb_index: int,
    ):
        """
        更新需要被跳过的参数组合的下标集合

        Args:
            original_exclude_comb_index (set): 原先的排除下标集合
            rest_para_combs_with_mapping (list): 接下来需要遍历的参数组合列表
            curr_ent_para (str): 当前的实体参数
            curr_comb_index (int): 当前组合的(起始)下标

        Returns:
            new_exclude_index (set): 更新后的排除集合
        """
        logger.info(f"Original exclude index: {original_exclude_comb_index}")
        new_exclude_index = original_exclude_comb_index
        logger.info(f"First para combs: {rest_para_combs_with_mapping[0]}")

        # 当前参数的下标
        curr_ent_index = rest_para_combs_with_mapping[0]["parameters"].index(curr_ent_para)
        logger.info(f"Current Entity index: {curr_ent_index}")

        for comb_index, combination in enumerate(rest_para_combs_with_mapping):
            if comb_index == 0:
                # 触发错误的参数组合肯定要排除 且不会再被重复遍历 因此直接跳过
                continue

            # 参数列表
            params = combination["parameters"]
            if params[curr_ent_index] == curr_ent_para:
                new_exclude_index.add(comb_index + curr_comb_index)
                logger.info(f"Delete para combs [{params}]")

        return new_exclude_index

    def _add_equations_from_tree(self, expr_tree: tuple, dft_to_curr: dict) -> tuple[int, int]:
        """
        递归遍历代数表达式树，添加方程到 Jacobian 验证器

        Args:
            expr_tree: 代数表达式树
            dft_to_curr: 默认参数到当前参数的映射

        Returns:
            (eq_count, ineq_count): 方程数量和不等式数量
        """
        expr_len = len(expr_tree)

        # 长度为 0：Free 约束，没有代数表达式
        if expr_len == 0:
            return 0, 0

        # 长度为 1：错误情况
        if expr_len == 1:
            logger.error(f"Invalid expression tree length: {expr_tree}")
            return 0, 0

        # 长度为 2：叶子节点或 NOT 操作
        if expr_len == 2:
            op = expr_tree[0]

            if op == "!":
                # NOT 操作：不添加任何方程
                return 0, 0
            else:
                # 叶子节点：添加到 Jacobian 验证器
                expression = _replace_expr(expr_tree[1], dft_to_curr)
                if op == "Eq":
                    logger.info(f"[Real Add expr] expr: {expression}")
                    self.jacob_verifier.add_equation(expression)
                    return 1, 0
                else:
                    return 0, 1

        # 长度为 3：AND 或 OR 逻辑
        if expr_len == 3:
            connector = expr_tree[1]
            if connector == "&" or connector == "|":
                # 递归处理两边，累加方程和不等式数量
                left_eq, left_ineq = self._add_equations_from_tree(expr_tree[0], dft_to_curr)
                right_eq, right_ineq = self._add_equations_from_tree(expr_tree[2], dft_to_curr)
                return left_eq + right_eq, left_ineq + right_ineq
            else:
                logger.error(f"Unknown connector: {connector}")
                return 0, 0

        logger.error(f"Invalid expression tree length: {expr_len}")
        return 0, 0

    def _bfs_validate_expr_tree(
        self,
        expr_tree: tuple,
        dft_to_curr_lttr: dict,
        params: list,
        constraint_name: str,
        geo_entity: str,
        curr_constraints: list,
        comb_index: int,
        eq_exclude_combs_index: set,
        para_combs_with_mapping: list,
        geo_ent_para: str,
    ) -> tuple[bool, set]:
        """
        递归遍历代数表达式树，验证每个表达式

        Args:
            expr_tree (tuple): 代数表达式树，e.g. (('Eq', expr), '&', (('G', expr), '&', ('G', expr)))
            dft_to_curr_lttr (dict): 默认参数到当前参数的映射
            params (str): 当前参数组合
            constraint_name (str): 约束名称
            geo_entity (str): 几何实体
            curr_constraints (list): 当前约束列表
            comb_index (int): 当前参数组合索引
            eq_exclude_combs_index (set): 需要排除的参数组合索引集合
            para_combs_with_mapping (list): 所有参数组合
            geo_ent_para (str): 几何实体参数

        Returns:
            tuple ((bool, set)):
            - success: 是否验证成功（False 表示参数组合失败，需要尝试下一个）
            - updated_exclude_set: 更新后的排除索引集合
        """
        expr_len = len(expr_tree)

        # 长度为 0：Free 约束，没有代数表达式
        if expr_len == 0:
            return True, eq_exclude_combs_index

        # 长度为 1：错误情况
        if expr_len == 1:
            logger.error(f"Invalid expression (length=1): {expr_tree}")
            return False, eq_exclude_combs_index

        # FIXME: Validation not right

        # 长度为 2：叶子节点或 NOT 操作
        if expr_len == 2:
            op = expr_tree[0]

            # NOT 操作：('!', expr)
            if op == "!":
                sub_expr = expr_tree[1]
                sub_success, sub_exclude = self._bfs_validate_expr_tree(
                    sub_expr,
                    dft_to_curr_lttr,
                    params,
                    constraint_name,
                    geo_entity,
                    curr_constraints,
                    comb_index,
                    eq_exclude_combs_index,
                    para_combs_with_mapping,
                    geo_ent_para,
                )
                return not sub_success, sub_exclude

            # 叶子节点：('Eq', expr) 或 ('G', expr) 等
            else:
                algebra_relation = op
                expression = expr_tree[1]
                expression = _replace_expr(expression, dft_to_curr_lttr)

                # 先复制一份之前已经生成的约束列表
                curr_constraints_list = curr_constraints.copy()
                # 根据选中的约束和参数构造cdl用于可能的验证
                now_constraint = form_fact(constraint_name, params)
                curr_constraints_list.append(now_constraint)

                # 形成待验证的CDL语句
                curr_cdl = self._form_cdl(geo_entity, curr_constraints_list)
                logger.info("+++ [Trying] current CDL:")
                logger.info(f"+   {curr_cdl}\n")

                if algebra_relation == "Eq":
                    # 是方程，先验证 Jacobian Matrix
                    if self.jacob_verifier.is_solved():
                        # 实体已经被完全约束，再添加方程无意义
                        logger.error(f"{geo_entity} completely Constrained! No more Equations!\n")
                        return False, eq_exclude_combs_index

                    # 实体并没有被完全约束，那么尝试添加方程并求解
                    jacobian_result = self.jacob_verifier.try_new_equation(expression)

                    if jacobian_result == JacobianResult.VALID:
                        # 若 Jacobian 秩增加，可以实际添加方程并验证
                        eq_solved = False
                        try:
                            eq_solved = self.gc.construct(curr_cdl, False)
                        except Exception as e:
                            # 有异常，说明求解失败
                            updated_exclude = self._update_exclude_index(
                                eq_exclude_combs_index,
                                para_combs_with_mapping[comb_index:],
                                geo_ent_para,
                                comb_index,
                            )
                            logger.error(f"Newer exclude: {updated_exclude}")
                            logger.error(f"when solving {expression}\nGot Error: {e}\n")
                            return False, updated_exclude

                        if eq_solved:
                            # 可以真正求解
                            logger.info(f"[Constructed] Eq: {expression}")
                            return True, eq_exclude_combs_index
                        else:
                            # 尝试其他位置的参数
                            updated_exclude = self._update_exclude_index(
                                eq_exclude_combs_index,
                                para_combs_with_mapping[comb_index:],
                                geo_ent_para,
                                comb_index,
                            )
                            logger.error(f"Newer exclude: {updated_exclude}")
                            logger.error(f"[Wrong] para comb index: {comb_index}\n")
                            return False, updated_exclude

                    elif jacobian_result == JacobianResult.REDUNDANT:
                        # 秩没有增加，方程冗余，排除当前参数组合
                        updated_exclude = self._update_exclude_index(
                            eq_exclude_combs_index,
                            para_combs_with_mapping[comb_index:],
                            geo_ent_para,
                            comb_index,
                        )
                        logger.error(f"Newer exclude: {updated_exclude}")
                        logger.info(f"for Eq: {expression}")
                        logger.error(f"[Redundant] para combs: {params}, Index: {comb_index}\n")
                        return False, updated_exclude
                else:
                    # 是不等式，直接求解验证
                    logger.info(f"Expression type: {algebra_relation}")
                    ineq_solved = False
                    try:
                        ineq_solved = self.gc.construct(curr_cdl, False)
                    except Exception as e:
                        ineq_solved = False
                        logger.error(f"when solving Ineq: {expression}\ngot error:{e}")

                    if ineq_solved:
                        logger.info(f"[Yes] Ineq: {expression} Constructed")
                        return True, eq_exclude_combs_index
                    else:
                        logger.error(f"[Ineq Not Solved]: {expression}")
                        logger.error(f"[Wrong] Para Combs: {params}")
                        return False, eq_exclude_combs_index

        # 长度为 3：内部节点，格式为 (left, '&', right) 或 (left, '|', right)
        if expr_len == 3:
            left_expr = expr_tree[0]
            connector = expr_tree[1]
            right_expr = expr_tree[2]

            if connector == "&":
                # AND 逻辑：所有子表达式都必须成功
                left_success, left_exclude = self._bfs_validate_expr_tree(
                    left_expr,
                    dft_to_curr_lttr,
                    params,
                    constraint_name,
                    geo_entity,
                    curr_constraints,
                    comb_index,
                    eq_exclude_combs_index,
                    para_combs_with_mapping,
                    geo_ent_para,
                )
                if not left_success:
                    return False, left_exclude

                right_success, right_exclude = self._bfs_validate_expr_tree(
                    right_expr,
                    dft_to_curr_lttr,
                    params,
                    constraint_name,
                    geo_entity,
                    curr_constraints,
                    comb_index,
                    left_exclude,
                    para_combs_with_mapping,
                    geo_ent_para,
                )
                return right_success, right_exclude

            elif connector == "|":
                # OR 逻辑：任一子表达式成功即可
                left_success, left_exclude = self._bfs_validate_expr_tree(
                    left_expr,
                    dft_to_curr_lttr,
                    params,
                    constraint_name,
                    geo_entity,
                    curr_constraints,
                    comb_index,
                    eq_exclude_combs_index,
                    para_combs_with_mapping,
                    geo_ent_para,
                )
                if left_success:
                    return True, left_exclude

                right_success, right_exclude = self._bfs_validate_expr_tree(
                    right_expr,
                    dft_to_curr_lttr,
                    params,
                    constraint_name,
                    geo_entity,
                    curr_constraints,
                    comb_index,
                    eq_exclude_combs_index,
                    para_combs_with_mapping,
                    geo_ent_para,
                )
                return right_success, right_exclude

            else:
                logger.error(f"Unknown connector: {connector}")
                return False, eq_exclude_combs_index

        # 其他长度：错误情况
        logger.error(f"Expression {expr_tree} invalid length: {expr_len}")
        return False, eq_exclude_combs_index

    def _try_construct(
        self,
        geo_entity: str,
        constraint_name: str,
        para_combs_with_mapping: list,
        curr_constraints: list,
    ):
        """
        基于选中的几何实体 geo_entity 和选中的约束 constraint 尝试填充约束的参数组合。

        对于等式方程：先用 Jacob Matrix 快速验证。若为有效组合：再用 self.gc.construct() 验证；反之：跳过该参数组合。

        对于不等式：直接使用 self.gc.construct() 计算可解性，可解：通过；不可解：跳过参数组合。

        Args:
            geo_entity (str): 几何实体, e.g. "Point(A)", "Line(l)"
            constraint (str): 选中的几何实体, e.g. "PointOnLine"
            para_combs_with_mapping (list): 带有映射的参数组合列表, e.g.
                [
                    {
                        "parameters": ["..."]
                        "mapping": ["..."]
                    },
                    {
                        "parameters": ["..."]
                        "mapping": ["..."]
                    },
                    {...}
                ]
            curr_constraints (list): 当前成功生成的约束, e.g.
                ["PointOnLine(A,l)", "PointOnLine(A, b)", "..."]

        Returns:
            tuple (bool, list):
                - find_correct_params (bool): 是否存在能够成功求解的参数组合
                - correct_p_combs_index (list): 能够正确构造的约束的参数的index
        """
        # 获取当前被约束几何实体的类型和参数
        geo_ent_type, geo_ent_para = self._parse_geo_entity(geo_entity)

        found_correct_params = False  # 可能的参数中是否存在最终construct()成功的参数组合
        correct_para_combs_index = []  # 记录所有成功通过验证的参数组合的index

        # TODO: Ready to remove, already implemented in self.gen_constraints_for_geo_ent()
        # Free type constraint
        if constraint_name.startswith("Free"):
            if len(curr_constraints) != 0:
                # Free不是第一个约束了 不应该添加该约束
                logger.error(f"{constraint_name} on Constrained Entity!")
                logger.error(f"Current constraints:\n    {curr_constraints}\n")
                return False, correct_para_combs_index
            else:
                # Free是第一个约束 判断自由度情况
                if self.jacob_verifier.current_rank == 0:
                    # 若自由度为0 则直接设为满秩 自由度应该肯定是0
                    self.jacob_verifier.current_rank = self.jacob_verifier.num_vars
                    logger.info(f"{geo_entity} freedom fixed!")
                    logger.info(f"{constraint_name} constrained {geo_entity}")
                    # 返回添加成功 以及使用的参数 (一般参数只应该是当前的实体)
                    return True, [0]
                else:
                    # 自由度不为0 无法添加自由约束
                    return False, correct_para_combs_index

        # 获取选中约束的具体代数约束
        algebra_expr_tree = self.gdl_constraints[constraint_name]["algebraic_forms"]

        # TODO: 多线程尝试所有可能的参数
        # 遍历所有可能的参数
        for comb_index, combination in enumerate(para_combs_with_mapping):
            # 固定住参数 再去验证表达式
            logger.info(f"+++ Try [{comb_index+1}] paras combination:")

            # 使用当前参数和映射 替换得到实际的方程
            params = combination["parameters"]
            dft_to_curr_lttr = combination["mapping"]

            # 参数组合尝试失败(方程)需要排除的参数下标集合
            eq_exclude_combs_index = set()
            if comb_index in eq_exclude_combs_index:
                continue

            logger.info(f"\033[35m{constraint_name}\033[0m algebra expression:")

            # BFS 遍历表达式树
            # FIXME: parameters correction
            success, eq_exclude_combs_index = self._bfs_validate_expr_tree(
                algebra_expr_tree,
                dft_to_curr_lttr,
                params,
                constraint_name,
                geo_entity,
                curr_constraints,
                comb_index,
                eq_exclude_combs_index,
                para_combs_with_mapping,
                geo_ent_para,
            )

            # 如果验证失败，跳过当前参数组合
            if not success:
                logger.error("[❌] Algebraic expressions not satisfied!")
                continue
            else:
                logger.info("[✅] All algebraic expressions satisfied!")
                found_correct_params = True
                correct_para_combs_index.append(comb_index)
                logger.info(f"[✅] para comb index: {comb_index}\n")

        return found_correct_params, correct_para_combs_index

    def gen_constraints_for_geo_ent(self, geo_entity: str, c_num_limit=3):
        """
        为几何实体生成不超过 c_num_limit 个约束

        Args:
            geo_entity (str): 几何实体, e.g. "Point(A)", "Line(l)"
            c_num_limit (int): 每个几何实体的最大约束数量

        Returns:
            generated_constraints (list): 生成的约束列表
            e.g. ["PointOnLine(A,l)", "PointInAngle(A,l,m)", ...]
        """
        self._check_format_geo_entity(geo_entity)

        # 提取几何实体的类型和参数
        # e.g. extracted geo_ent_para = 'A'
        geo_ent_type, geo_ent_para = self._parse_geo_entity(geo_entity)

        # 根据当前约束状态更新可用约束列表
        avail_constraints = self._update_avail_constraints(geo_entity)

        if not avail_constraints:
            logger.error(f"No available constraints for {geo_entity}!")
            return []

        # 确定生成约束的个数上限 若随机数大于当前可用约束数 取可用约束的个数
        random_limit = self.random_gcg.randint(1, c_num_limit)
        constraints_num_limit = min(len(avail_constraints), random_limit)

        # 尝试次数限制
        try_count = 1
        max_try = 30

        # 当前成功生成的约束列表 带参数
        curr_constraints = []

        # 将当前几何实体的符号设为jacobian验证代数约束的目标符号
        curr_free_symbols = self.sym_of_ent[geo_ent_para]
        self.jacob_verifier = jac_verifier(curr_free_symbols)

        # 初始化实体秩（默认为0）
        self.entity_ranks[geo_ent_para] = 0

        while len(curr_constraints) < constraints_num_limit and try_count <= max_try and len(avail_constraints) > 0:
            # 随机选择约束，完全依赖Jacobian验证阶段进行检查
            constraint_chosen = self.random_gcg.choice(avail_constraints)

            print(f"[Try round {try_count}] constraint: {constraint_chosen}")

            # 生成该约束所有可能的参数组合
            para_combs_with_mapping = self._get_all_para_combs(geo_ent_type, geo_ent_para, constraint_chosen)

            # 成功生成了参数组合
            if para_combs_with_mapping:
                # 单独处理 Free 约束情况
                if constraint_chosen.startswith("Free"):
                    if len(curr_constraints) > 0:
                        # 已有其他约束 不能再加Free约束 所以同时删除所有Free约束
                        # 应该只有该实体类型对应的Free constraint
                        avail_constraints.remove(constraint_chosen)
                        try_count += 1
                        # 再尝试其他可用约束
                        continue
                    else:
                        # 可以添加Free约束 直接添加
                        # 正确参数一般只有自身
                        paras = para_combs_with_mapping[0]["parameters"]
                        correct_constraint = form_fact(constraint_chosen, paras)
                        if self._is_duplicate(correct_constraint):
                            # 生成了重复的自由几何实体 有问题
                            logger.error(f"Duplicate {correct_constraint}!\n")
                            avail_constraints.remove(constraint_chosen)
                            try_count += 1
                            # 删除后 尝试其他约束
                            continue
                        else:
                            curr_constraints.append(correct_constraint)
                            self.all_constraints.add(correct_constraint)

                            # 设置为 Free 实体（rank = dof）
                            dof = self.entity_dof_map[geo_ent_type]
                            self._update_entity_rank(geo_ent_para, dof)
                            # 添加Free约束后不能添加其他约束 直接break
                            break

                # 用实际方程检验参数组合是否有解
                find_correct_params, correct_para_comb_index = self._try_construct(
                    geo_entity,
                    constraint_chosen,
                    para_combs_with_mapping,
                    curr_constraints,
                )

                # 发现了有效的参数组合
                if find_correct_params:
                    # TODO: 多线程尝试所有正确的参数组合

                    # TODO: find_no_duplicate_para()

                    no_duplicate_idx = []  # 收集有效且不重复的下标
                    for index in correct_para_comb_index:
                        # 获取正确索引的参数和映射
                        paras = para_combs_with_mapping[index]["parameters"]
                        # 构造完整约束事实(句子)
                        correct_constraint = form_fact(constraint_chosen, paras)

                        # 检查是否重复约束
                        if self._is_duplicate(correct_constraint):
                            print(f"- [Duplicate] para combs index: [{index}]")
                            # 参数组合导致重复约束 检查下一个正确参数的index
                            continue
                        else:
                            no_duplicate_idx.append(index)
                            print("+++ Correct Constraint generated:")
                            print(f"+   {correct_constraint}")

                    logger.info(f"[{len(no_duplicate_idx)}] Correct index for {constraint_chosen}")

                    if len(no_duplicate_idx) == 0:
                        # 没有不重复的正确方程 说明没有有效的参数组合
                        logger.error(f"Constraint '{constraint_chosen}' for {geo_entity}!")
                        logger.error("[All Duplicate] para combs!")
                        # 移除该约束 没有无重复的正确的参数组合
                        avail_constraints.remove(constraint_chosen)

                    else:
                        # 当前只取第一个不重复的参数
                        correct_index = no_duplicate_idx[0]
                        paras = para_combs_with_mapping[correct_index]["parameters"]
                        correct_constraint = form_fact(constraint_chosen, paras)

                        # 获取选中约束的代数方程
                        algebra_constraints = self.gdl_constraints[constraint_chosen]["algebraic_forms"]
                        dft_to_curr = para_combs_with_mapping[correct_index]["mapping"]

                        # 使用递归函数添加方程
                        eq_count, ineq_count = self._add_equations_from_tree(algebra_constraints, dft_to_curr)
                        self.eq_constraint_num += eq_count
                        self.ineq_constraint_num += ineq_count

                        # 最终添加
                        curr_constraints.append(correct_constraint)
                        self.all_constraints.add(correct_constraint)

                        # 更新实体约束状态
                        self._update_entity_rank(geo_ent_para, self.jacob_verifier.current_rank)
                        dof = self.entity_dof_map[geo_ent_type]
                        ratio = self.jacob_verifier.current_rank / dof
                        logger.info(f"{geo_entity} jacobian rank: {self.jacob_verifier.current_rank}")
                        logger.info(f"dof={dof}, ratio={ratio:.2f}")

                        # 将约束的multiple_forms也添加进列表 避免添加参数不同但等价的约束
                        # TODO: self._add_multiple_forms(correct_constraint, para_comb)
                        # No multiple_forms for now

                # 没有有效的参数组合
                else:
                    logger.error(f"- [FAIL] constraint {constraint_chosen} for {geo_entity}!")
                    logger.error("---- All para combs exhausted!")
                    # 穷尽参数组合 无法得到正确的参数 应该防止再次抽取到该约束 将其移除
                    avail_constraints.remove(constraint_chosen)

            else:
                # 没有任何参数组合 说明约束不合适
                logger.error(f"{constraint_chosen} for {geo_entity:}")
                logger.error(f"--- NO Suitable Parameter!\n")
                # 防止再次抽取到该约束 将其移除
                avail_constraints.remove(constraint_chosen)
                # 尝试别的约束
                continue

            try_count += 1

        # Break reason check
        if len(curr_constraints) < constraints_num_limit:
            print(f"--- [FAIL] to Generate enough constraints: {geo_entity}")

        if try_count > max_try:
            print(f"--- MAX try count reached for {geo_entity}!")

        if len(avail_constraints) <= 0:
            print(f"--- NO Available constraints for {geo_entity}!")

        return curr_constraints

    """↑------- Geometry Constraints Generation -------↑"""
    """↓--------------- Main Interface ----------------↓"""

    def generate(self, target_entity_num=5, max_cons_per_ent=3):
        """
        生成一个完整的几何构型

        Args:
            cdl_nums (int): 需要生成的CDL句子数量，也是几何实体的数量
            max_constraints_per_entity (int): 每个几何实体的最大约束数量

        Returns:
            self.status (dict): 构型生成的状态(结果): 几何实体和CDL句子
        """
        # 类 GeometricConfiguration 中设置了固定的随机种子
        # 在初始化时实现了类专用的Random实例
        self.clear_status()  # 多次生成时 需要先清空先前的构型状态

        # 随机生成几何实体(构图语句CDL的个数即几何实体的个数)
        for i in range(target_entity_num):
            # 随机生成一个几何实体
            geo_entity = self.generate_one_geo_entity()
            # 为当前几何实体生成约束
            constraints = self.gen_constraints_for_geo_ent(geo_entity, max_cons_per_ent)

            # 通过 geo_entity 及其约束 constraints 形成完整的一句CDL语句
            # e.g. Point(A):PointOnLine(A, l)&PointOnLine(A, n)
            if constraints:
                # 使用&符号连接所有约束 若链接符号改动 记得修改
                # 组合成完整的CDL语句
                # cdl_sentence = f"{geo_entity}:{constraints_str}"
                cdl_sentence = self._form_cdl(geo_entity, constraints)
                print(f"[{i+1}] CDL Completed:\n{cdl_sentence}\n")

                # 确定了单句 CDL 直接实际构造图形
                try:
                    self.gc.construct(cdl_sentence, True)
                except Exception as e:
                    logger.error(f"[CDL] {cdl_sentence}\n")
                    logger.error(f"construct(): {e}")
                    # 终止生成 如果需要保存 把已经生成的保存下来
                    # if save_cdl:
                    #     self.save_cdls_to()
                    break

                # 将完整的构图语句添加进 self.all_cdls
                self.all_cdls.append(cdl_sentence)

            else:
                # 约束生成失败 只添加当前几何实体
                self.all_cdls.append(geo_entity)
                logger.error(f"No Constraints for {geo_entity}!")
                # Retry geo entity of other types

        return self.status

    def validate_and_save(self, cdls=None, save_to_file=None):
        """
        验证 CDL 并存为 JSONL 文件, 默认在完整生成完后进行验证和存储, 因此使用 self.all_cdls

        若通过参数提供cdl列表, 则使用参数中的cdl列表进行验证

        Args:
            cdls: 需要验证的CDL列表
            save_to_file: 验证成功后保存的文件路径(可选)

        Returns:
            dict: {
                "all_passed": bool,
                "passed_count": int,
                "failed_count": int,
                "total_count": int,
                "errors": list  # 失败的CDL和错误信息
            }
        """
        # 使用提供的CDL列表或默认使用self.all_cdls
        cdls_to_validate = cdls if cdls is not None else self.all_cdls

        result = {
            "all_passed": False,
            "passed_count": 0,
            "failed_count": 0,
            "total_count": len(cdls_to_validate),
            "errors": [],
        }

        if not cdls_to_validate:
            logger.warning("No CDLs to validate!")
            return result

        # 初始化gc类
        gc = geocfg(self.parsed_gdl)

        # 遍历CDL列表，调用gc.construct()验证
        for i, cdl in enumerate(cdls_to_validate):
            try:
                construct_result = gc.construct(cdl)
                if construct_result:
                    # 验证成功
                    msg = f"PASSED CDL {i + 1}/{len(cdls_to_validate)}: {cdl}"
                    logger.info(msg)
                    result["passed_count"] += 1
                else:
                    # 验证失败
                    msg = f"FAILED CDL {i + 1}/{len(cdls_to_validate)}: {cdl}"
                    logger.error(msg)
                    result["failed_count"] += 1
                    result["errors"].append({"index": i, "cdl": cdl, "error": "Construction failed"})
            except Exception as e:
                # 验证异常
                msg = f"ERROR CDL {i + 1}/{len(cdls_to_validate)}: {cdl} - Exception: {str(e)}"
                logger.error(msg)
                result["failed_count"] += 1
                result["errors"].append({"index": i, "cdl": cdl, "error": str(e)})

        # 判断是否全部通过
        result["all_passed"] = result["failed_count"] == 0

        # 记录总结信息
        if result["all_passed"]:
            summary_msg = f"All CDLs PASSED! Total: {result['total_count']}"
            logger.info(summary_msg)
        else:
            summary_msg = f"Validation completed with errors. Total: {result['total_count']}, Passed: {result['passed_count']}, Failed: {result['failed_count']}"
            logger.warning(summary_msg)

        # 如果全部通过且指定了保存路径，保存到文件
        if result["all_passed"] and save_to_file:
            try:
                # 确保输出目录存在
                output_dir = os.path.dirname(save_to_file) if os.path.dirname(save_to_file) else "."
                os.makedirs(output_dir, exist_ok=True)

                # 生成时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                # 检查文件是否已存在，如果存在则添加序号
                final_file_path = save_to_file
                counter = 1
                while os.path.exists(final_file_path):
                    base_name = os.path.splitext(os.path.basename(save_to_file))[0]
                    extension = os.path.splitext(save_to_file)[1] or ".json"
                    new_filename = f"{base_name}_{timestamp}_{counter:03d}{extension}"
                    final_file_path = os.path.join(output_dir, new_filename)
                    counter += 1

                # 构建JSON数据
                result_json = {
                    "problem_id": 1,
                    "annotation": f"QC_GCG_{datetime.now().strftime('%Y-%m')}",
                    "source": "Geometric Configuration Generator",
                    "problem_text_cn": "",
                    "problem_text_en": "",
                    "problem_img": "",
                    "seed": getattr(self, "seed", 0),
                    "constructions": cdls_to_validate,
                    "text_cdl": [],
                    "image_cdl": [],
                    "goal_cdl": "",
                    "problem_answer": "",
                    "theorem_seqs": [],
                    "metadata": {
                        "created_at": timestamp,
                        "total_cdls": len(cdls_to_validate),
                    },
                }

                # 保存文件
                save_json(result_json, final_file_path)
                logger.info(f"{len(cdls_to_validate)} CDLs saved to: {final_file_path}")

                result["saved_to"] = final_file_path
            except Exception as e:
                logger.error(f"Failed to save CDLs to file: {e}")
                result["save_error"] = str(e)

        return result

    def batch_val(self, problems_dir, log_file=None):
        """
        批量验证目录中的所有问题

        Args:
            problems_dir (str): 存放生成构型的目录
            log_file (str): 日志文件, Defaults to None.

        Returns:
            dict: 验证结果统计
        """
        # 如果没有提供日志文件，则创建一个默认的日志文件名
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(problems_dir, f"batch_val_{timestamp}.log")

        # 设置日志记录器
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,  # 覆盖之前的配置
        )
        logger = logging.getLogger(__name__)

        stats = {"total": 0, "passed": 0, "failed": 0, "errors": []}

        # 获取所有问题文件
        try:
            problem_files = [f for f in os.listdir(problems_dir) if f.endswith(".json")]

        except Exception as e:
            logger.error(f"Failed to list directory {problems_dir}: {str(e)}")
            return stats

        logger.info(f"Starting batch validation of {len(problem_files)} problems from {problems_dir}")

        for problem_file in problem_files:
            stats["total"] += 1
            problem_id = os.path.splitext(problem_file)[0]
            problem_path = os.path.join(problems_dir, problem_file)

            try:
                # 加载问题数据
                problem_data = load_json(problem_path)
                # 获取构型数据
                constructions = problem_data.get("constructions", [])

                # 验证构型
                logger.info(f"Validating problem {problem_id} from {problem_path}")

                # 使用统一的验证方法
                result = self.validate_and_save(cdls=constructions)

                # 更新统计信息
                if result["all_passed"]:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                stats["failed"] += 1
                error_info = {"problem_id": problem_id, "error": str(e)}
                stats["errors"].append(error_info)
                logger.error(f"Error loading {problem_id}: {str(e)}")

        # 输出统计信息
        summary = (
            f"Batch validation completed. Total: {stats['total']}, Passed: {stats['passed']}, Failed: {stats['failed']}"
        )
        logger.info(summary)
        logger.info(f"Validation logs saved to: {log_file}")

        return stats

    """↑--------------- Main Interface ----------------↑"""
    """↓--------------- Other Interface ---------------↓"""

    def save_parsed_gdl_to(self, file_path):
        """
        将解析过后的 GDL 存入 file_path 中

        Args:
            file_path (str): parsed_gdl 的存储路径, e.g. /path/to/your/parsed_gdl
        """
        if not self.parsed_gdl:
            print("Error: NO parsed_gdl to save!\n")
            return None

        # 确保输出目录存在
        output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
        os.makedirs(output_dir, exist_ok=True)

        save_json(self.parsed_gdl, file_path)

    def save_cdls_to(self, file_path):
        """
        将当前生成的 CDLs 保存为 JSON 文件，文件名避免冲突

        Args:
            file_path (str): 目标文件夹路径

        Returns:
            final_file_path (str): 最终保存的文件路径
        """
        if not self.all_cdls:
            print("Error: No CDLs to save!\n")
            return file_path

        # 确保输出目录存在
        output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else "."
        os.makedirs(output_dir, exist_ok=True)
        print(output_dir)

        # 获取基础文件名和扩展名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        extension = os.path.splitext(file_path)[1] or ".json"

        # 生成时间戳和唯一标识
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒

        # 检查文件是否已存在，如果存在则添加序号
        final_file_path = file_path
        counter = 1

        while os.path.exists(final_file_path):
            # 构建新文件名：原文件名_时间戳_序号
            new_filename = f"{base_name}_{timestamp}_{counter:03d}{extension}"
            final_file_path = os.path.join(output_dir, new_filename)
            counter += 1

        # 直接构建JSON数据
        result_json = {
            "problem_id": 1,
            "annotation": f"QC_GCG_{datetime.now().strftime('%Y-%m')}",
            "source": "Geometric Configuration Generator",
            "problem_text_cn": "",
            "problem_text_en": "",
            "problem_img": "",
            "seed": getattr(self, "seed", 0),
            "constructions": self.all_cdls,
            "text_cdl": [],
            "image_cdl": [],
            "goal_cdl": "",
            "problem_answer": "",
            "theorem_seqs": [],
            "metadata": {
                "created_at": timestamp,
                "total_cdls": len(self.all_cdls),
            },
        }

        # 保存文件
        save_json(result_json, final_file_path)

        print(f"{len(self.all_cdls)} CDLs save to: {final_file_path}\n")
        return final_file_path

    def _count_constraints_by_type(self) -> dict:
        """
        统计各类型约束的数量

        Returns:
            dict: {constraint_type: count}
        """
        constraint_counts = {}
        for constraint in self.all_constraints:
            # 根据约束的实际结构提取类型
            if isinstance(constraint, dict):
                c_type = constraint.get("type", "unknown")
            elif isinstance(constraint, str):
                # 如果约束是字符串形式，尝试提取类型
                c_type = constraint.split("(")[0] if "(" in constraint else "unknown"
            else:
                c_type = "unknown"
            constraint_counts[c_type] = constraint_counts.get(c_type, 0) + 1
        return constraint_counts

    def _calculate_total_dof(self) -> int:
        """
        计算总自由度

        Returns:
            int: 总自由度（点*2 + 线*2 + 圆*3）
        """
        points_dof = len(self.points) * 2
        lines_dof = len(self.lines) * 2
        circles_dof = len(self.circles) * 3
        return points_dof + lines_dof + circles_dof

    def _calculate_constrained_dof(self) -> int:
        """
        计算被约束的自由度

        Returns:
            int: 被约束的自由度（简化计算：约束数量）
        """
        # 简化计算：每个约束平均约束1个自由度
        # 更精确的计算需要分析每个约束实际约束的自由度数
        return len(self.all_constraints)

    """↑--------------- Other Interface  ----------------↑"""
