import random
import sympy as sp
import numpy as np
from enum import Enum


class JacobianResult(Enum):
    """Jacobian验证结果枚举"""

    VALID = "valid"  # 有效：秩增加，方程约束了当前实体
    REDUNDANT = "redundant"  # 冗余：秩没增加，方程对当前实体无新约束


class JacobianMatrixVerifier:
    def __init__(self, target_variables):
        """
        初始化针对特定几何实体的求解器。

        Args:
            target_variables (list): 需要求解的自由变元，例如 [x, y] 或 [u, v, r]
        """
        self.variables = target_variables
        self.num_vars = len(target_variables)

        # 已被采纳的有效方程列表
        self.accepted_equations = []
        # 当前已获得的矩阵秩 (初始为0)
        self.current_rank = 0
        # 缓存一个随机数映射表，保证同一个求解步骤中，
        # 同一个变量（包括已知量和未知量）总是对应同一个随机数，
        # 避免因随机数跳变导致的计算抖动。
        self._random_cache = {}

    def _get_random_subs(self, equations):
        """
        生成随机代入字典。
        将方程涉及的所有符号（包括未知变量和已知参数）都映射为随机浮点数。
        """
        # 收集当前所有方程中的所有符号
        all_symbols = set()
        for eq in equations:
            all_symbols.update(eq.free_symbols)

        # check free symbols
        print(f"+++ In Eq: {equations}")
        print(f"    Free symbols: {all_symbols}")
        print()

        # 为尚未赋值的符号生成随机数
        subs = {}
        for sym in all_symbols:
            if sym not in self._random_cache:
                # 范围选宽一点，避开 0 和常见整数
                self._random_cache[sym] = random.uniform(-50.0, 50.0)
            subs[sym] = self._random_cache[sym]

        # Show symbol-value mapping
        print("++++ Symbol-Value random mapping:")
        column = 1
        for symbol, random_value in subs.items():
            end_mark = ""
            # Change line when column hit 5
            if column == 5:
                column = 1
                end_mark = "\n"
            else:
                end_mark = "; "
            print(f" {symbol} = {random_value:.5f}", end=end_mark)
            column += 1

        print()

        return subs

    def _calculate_rank(self, eq_list):
        """
        核心方法：计算给定方程组针对目标变量的雅可比矩阵的数值秩。
        """
        if not eq_list:
            return 0

        # 1. 构建符号雅可比矩阵 (Rows=方程数, Cols=变量数)
        # 注意：这里只对 target_variables 求导
        J_sym = sp.Matrix(eq_list).jacobian(self.variables)

        # 2. 获取随机数值映射
        subs_map = self._get_random_subs(eq_list)

        # 3. 代入数值
        # try-except 是为了防止极罕见的除零错误
        try:
            J_num = J_sym.subs(subs_map)
            # 使用 numpy 转换求解秩通常更快，但 SymPy 自带的也够用
            # 这里强制转换为 float 类型矩阵以触发数值算法
            J_num = sp.matrix2numpy(J_num, dtype=float)

            # 使用 numpy 的秩计算 (需要 import numpy)
            return np.linalg.matrix_rank(J_num)

        except Exception as e:
            print("[Error] when computing Rank, may be singular point!")
            print(f"[Message] {e}\n")
            return 0

    def try_new_equation(self, new_equation) -> JacobianResult:
        """
        尝试 new_equation 是否实际增加了目标实体的秩

        Args:
            new_equation (Equation): 尝试添加的等式表达式

        Returns:
            JacobianResult: 验证结果
                - VALID: 有效，秩增加了
                - REDUNDANT: 冗余，秩没增加，方程对当前实体无新约束
        """
        if not self.variables:
            # 不存在目标变量，无法验证
            print("[Redundant & Error] No target variables")
            print(f"Current variables: {self.variables}")
            print(f"Eq: {new_equation}\n")
            return JacobianResult.REDUNDANT

        # 统一通过 Jacobian 秩验证，不管方程是否包含其他实体
        candidate_eq_list = self.accepted_equations + [new_equation]
        new_rank = self._calculate_rank(candidate_eq_list)

        print(f"[Try Add Equation] expr: {new_equation}")
        # 判断秩的变化情况
        if new_rank > self.current_rank:
            # 秩增加，方程约束了目标实体
            # 检查是否已经完全可解
            is_solved = new_rank == self.num_vars

            if is_solved:
                print(f"[Solved] Entity var: {self.variables}")
            else:
                print(f"[Not Solved] Entity var: {self.variables}")

            print(f"[Current Rank] {new_rank} / {self.num_vars}")
            print()
            return JacobianResult.VALID
        else:
            # 秩不增加，方程冗余
            print(f"[Redundant] Rank remain: {self.current_rank}")
            print()
            return JacobianResult.REDUNDANT

    def add_equation(self, new_correct_eq):
        """默认添加的方程 是经过检验的正确方程"""
        # 以防万一 再次测试该方程
        jacobian_result = self.try_new_equation(new_correct_eq)

        if jacobian_result == JacobianResult.VALID:
            # 方程有效，添加并更新秩
            self.accepted_equations.append(new_correct_eq)
            new_rank = self._calculate_rank(self.accepted_equations)
            self.current_rank = new_rank
            print(f"[Rank Updated] {new_rank}/{self.num_vars} (rank/freedom)")
            print()
        elif jacobian_result == JacobianResult.REDUNDANT:
            # 方程冗余，不添加
            print(f"[Not Added - REDUNDANT] {new_correct_eq}!")
            print()
        else:
            print(f"[Wrong] Unknown Jacobian Result!")

    def remove_current_equation(self):
        """移除最新验证的方程 并更新秩"""
        if self.accepted_equations:
            rm_eq = self.accepted_equations.pop()
            # 去除方程后 重新计算秩
            new_rank = self._calculate_rank(self.accepted_equations)
            self.current_rank = new_rank

            print(f"[REMOVED Eq]:{rm_eq}")
            print(f"[Rank Status]: {self.current_rank}/{self.num_vars}")
            print()

        else:
            print("[Error] NO Equation to Remove")
            print()

    def is_solved(self):
        """判断当前实体是否已完全确定"""
        return self.current_rank == self.num_vars

    def get_dof(self):
        """获取剩余自由度"""
        return self.num_vars - self.current_rank

    def _test_jacobian_matrix(self, target_free_symbols, equations):
        """设定目标变元和方程组, 测试Jacobian Matrix"""
        matrix = sp.Matrix(equations)
        Jacobian = matrix.jacobian(target_free_symbols)
        print("+++ Target free symbols:")
        print(target_free_symbols)
        print("+++ Matrix:")
        print(matrix)
        print()

        print("+++ Jacobian, Free Symbols:")
        print(Jacobian.free_symbols)
        print()

        print("+++ Jacob Rank:")
        print(Jacobian.rank())
        print()
