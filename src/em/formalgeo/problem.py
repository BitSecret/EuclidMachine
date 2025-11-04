from em.formalgeo.parser import parse_gdl


class Problem:
    """The <Problem> class represents the construction and reasoning process of a geometric configuration.

    Attributions:
        self.parsed_gdl (dict): Parsed Geometry Definition Language (GDL). You can save it to json to view the specific
        format. The function em.formalgeo.parser.parse_gdl demonstrates the detailed parsing process of GDL.

        self.facts (list): fact_id -> (predicate, instance, premise_ids, entity_ids, operation_id). Store all known
        facts of the current problem. The index of element is its fact_id. Specifically:
            1.predicate (str): The type of fact, such as 'Line', 'Parallel', 'Equation', etc.
            2.instance (tuple): The instance of fact, categorized into geometric relations and algebraic relations,
            such as ('A', 'B', 'C'), 'MeasureOfAngle(lk)-1', etc.
            3.premise_ids (tuple): The premise of the current fact.
            4.entity_ids (tuple): The dependent entities of the current fact.
            5.operation_id (int): The operation that yielded the current fact.
        examples: [('Point', ('A',), (,), (,), 0),  # free entity
                   ('Line', ('l',), (,), (3, 6), 3),  # constructed entity
                   ('PointOnLine', ('A', 'l'), (,), (3, 6), 3),  # initial geometric relations
                   ('Perpendicular', ('l', 'm'), (7, 8, 9), (2, 3), 6),  # inferred geometric relations
                   ('Equation', 'MeasureOfAngle(lk)-1', (10, 11), (1, 4), 8)]  # algebraic relations

        self.id (dict): (predicate, instance) -> fact_id. Mapping from fact to fact_id.
        examples: {('Point', ('A',)): 0, ('PointOnLine', ('A', 'l')): 5, ('Equation', 'MeasureOfAngle(lk)-1'): 12}

        self.groups (list): operation_id -> (fact_id). A tuple of fact_id that share the same operation_id. The
        index of element is its operation_id.
        examples: [(0, 1, 2), (3, 4), (5, 6, 7, 8)]

        self.ids_of_predicate (dict): predicate -> [fact_id]. The fact_id of facts with the same predicate.
        examples: {'Point': [0, 1, 3, 4], 'PointOnLine': [5, 7]}

        self.instances_of_predicate (dict): predicate -> [instance]. The instance of facts with the same predicate.
        examples: {'Point': ['A', 'B', 'C', 'D'], 'PointOnLine': [('A', l), ('B', l)]}

        self.operations (list): operation_id -> operation. Mapping from operation_id to operation. The index of
        element is its operation_id.
        examples: ['Point(A): PointOnLine(A, l)', 'perpendicular_judgment_angle(l,k)']

        self.constructions (list): operation_id -> (constraints, dependent_syms, target_syms, solved_values). Store
        the constraints obtained from parsing each construction statement. The index of element is its operation_id.
        specifically:
            1.constraints (list): The disjunctive normal form of constraints, where each element is a conjunction
            with the form of {'eq': [], 'l': [], 'leq': [], 'g': [], 'geq': [], 'ueq': []}.
            2.dependent_syms (tuple): Tuple of symbols related to the dependency entities of the current entity,
            which are replaced with values before solving.
            3.target_syms (tuple): Tuple of symbols related to the current entity, which need to be solved.
            4.solved_values (list): List of solved values, which stores multiple values obtained from solving, where
            the values correspond one-to-one with target_syms.
        examples: [([{'eq': [A.y-l.k*A.x-l.b], 'l': [], 'leq': [], 'g': [], 'geq': [], 'ueq': []}],
                   (l.k, l.b), (A.x, A.y), [(0, 0), (1, 1), ..., (999, 999)]),  # infinite solutions
                   ([{'eq': [A.y-l.k*A.x-l.b, B.y-l.k*B.x-l.b], 'l': [], 'leq': [], 'g': [], 'geq': [], 'ueq': []}],
                   (A.x, A.y, B.x, B.y), (l.k, l.b), [(1, 0)])]  # finite solutions

        self.sym_of_attr (dict): attr -> sym. The mapping from attribute names to attribute symbols.
        examples: {'A.x': A.x, 'l.k': l.k,  # parameter of entity
                   'LengthOfSegment(AB)': LengthOfSegment(AB),  # attribution of entity
                   'LengthOfSegment(BA)': LengthOfSegment(AB)}  # attribution of entity

        self.attr_of_sym (dict): sym -> [attr]. The mapping from attribute symbols to attribute names. An attribute
        symbol may have multiple attribute names.
        examples: {A.x: ['A.x'], l.k: ['l.k'],  # parameter of entity
                   LengthOfSegment(AB): ['LengthOfSegment(AB)', 'LengthOfSegment(BA)']}  # attribution of entity

        self.value_of_sym (dict): sym -> value. The value of a symbol.
        examples: {A.x: 0.25846578, l.k: 0.89568755, LengthOfSegment(AB): 5, MeasureOfAngle(lk): 90}

        self.equations (list): [[simplified_equation, original_equation_fact_id, dependent_equation_fact_id]]. Store
        the simplified equation, along with the fact_id of its original equation and the dependent equations used
        in the simplification process. Taking the equation a + b - c = 0 as an example, if the fact_id of a + b - c = 0
        and c - 2 = 0 are 1 and 2, respectively, then the element (a + b - 2, 1, [2]) would be added to self.equations.
        specifically:
            1.simplified_equation (equation): The simplified equation.
            2.original_equation_fact_id (int): The fact_id of its original equation.
            3.dependent_equation_fact_id (list): The dependent equations used in the simplification process. Note that
            all dependent equations are solved values. Whenever a symbol sym is successfully solved to obtain a value,
            an algebraic relation sym - value = 0 is constructed and added to the self.facts. Simultaneously, For each
            element in self.equations, all sym in the  simplified_equation are replaced with its value, and the fact_id
            of the algebraic relation sym - value = 0 is added to dependent_equation_fact_id. If the simplified_equation
            contains no unsolved symbols, remove current element from self.equations.
        example: [[a + b - 2, 1, [2]], [d + e, 2, []]]

    Methods:
        self.construct(entity): Construct a geometric entity (point, line, circle). If successfully construct the
        entity, return True; otherwise, return False.

        self.apply(theorem): Apply a theorem. If applying the theorem adds new conditions, return True; otherwise,
        return False.
    """

    def __init__(self, parsed_gdl):
        """Problem conditions, goal, and solving message.

        Args:
            parsed_gdl (dict): Parsed Geometry Definition Language (GDL).
        """
        self.parsed_gdl = parsed_gdl

        self.facts = []  # fact_id -> (predicate, instance, premise_ids, entity_ids, operation_id)
        self.id = {}  # (predicate, instance) -> fact_id
        self.groups = []  # operation_id -> (fact_id)
        self.ids_of_predicate = {}  # predicate -> [fact_id]
        self.instances_of_predicate = {}  # predicate -> [instance]
        self.operations = []  # operation_id -> operation
        self.times = []  # operation_id -> time

        self.constructions = []  # operation_id -> (constraints, dependent_syms, target_syms, solved_values)

        self.sym_of_attr = {}  # attr -> sym
        self.attr_of_sym = {}  # sym -> [attr]
        self.value_of_sym = {}  # sym -> value
        self.equations = []  # [[simplified_equation, original_equation_fact_id, dependent_equation_fact_ids]]

    def construct(self, entity):
        """执行复杂的多步计算。
        1.格式合法性检查和实体存在性检查（每个约束至少有当前实体，并且涉及到的其他实体必须已经存在），通过 self._parse(entity)完成。 如果
        通过，返回解析好的
        2.运行same_entity检查，若本轮构建的实体实际上已经在facts中，则将其替换为facts中的实体，并对几何关系去重，对约束关系调整
        3.添加fact

        Args:
            entity (str): 字符串列表，描述...

        Returns:
            result (bool): If successfully construct the entity, return True; otherwise, return False.

        Raises:
            TypeError: 如果参数类型错误
            RuntimeError: 如果计算过程中出现异常

        Example:
            'theorem_name'
        """

        pass

    # 暂时不关心赋予属性具体值的操作
    # def assign(self, attr, value):
    #     """After the shape is constructed, specific values can be assigned to the attribute of entity.
    #     构图过程，会添加 1.实体 2.几何关系 2.代数关系（比例关系，不包含具体的属性值）
    #     属性赋予过程，会添加 1.代数关系（具体的属性值，并且要求与构图过程的比例关系不冲突）
    #     定理应用过程，会添加 1.几何关系（实体的真实值如点坐标，带入几何关系的代数约束，要成立）
    #                      2.代数关系（实体的真实值如点坐标，带入代数关系要成立）
    #
    #     self.assign(attr, value): After the shape is constructed, specific values can be assigned to the attribute
    #         of entity. If the assignment is successful, return True; otherwise, return False.
    #
    #     Args:
    #         attr (str): 字符串列表，描述...
    #         value (float): 字符串列表，描述...
    #
    #     Returns:
    #         result (bool): If successfully construct the entity, return True; otherwise, return False.
    #
    #     Raises:
    #         TypeError: 如果参数类型错误
    #         RuntimeError: 如果计算过程中出现异常
    #
    #     Example:
    #         >>> self.apply("theorem_name")
    #         'theorem_name'
    #     """

    def apply(self, theorem):
        """执行复杂的多步计算。
        这里是详细的功能描述，可以跨越多行。
        第二段通常描述实现细节或算法原理。
        3.通过construction部分的构建或solver部分的推理，得到新的fact

        Args:
            theorem (str): 字符串列表，描述...

        Returns:
            result (bool): If applying the theorem adds new conditions, return True; otherwise, return False.

        Raises:
            TypeError: 如果参数类型错误
            RuntimeError: 如果计算过程中出现异常

        Example:
            >>> self.apply("theorem_name")
            'theorem_name'
        """
        pass

    def _parse_entity(self, entity):
        """Parse a geometric construction statement into constraints, dependent entities, and the geometric and
        algebraic relations to be added.

        Args:
            entity (str): Predicate definition.

        Returns:
            self.predicate_definition (dict): Parsed predicate definition.

            self.theorem_definition (dict): Parsed theorem definition. Save dict to json to view the specific format.

            self.facts (list): fact_id -> (predicate, instance, premise_ids, entity_ids, operation_id). Store all known
            facts of the current problem. The index of element is its fact_id. Specifically:
                1.predicate (str): The type of fact, such as 'Line', 'Parallel', 'Equation', etc.
                2.instance (tuple): The instance of fact, categorized into geometric relations and algebraic relations,
                such as ('A', 'B', 'C'), 'MeasureOfAngle(lk)-1', etc.
                3.premise_ids (tuple): The premise of the current fact.
                4.entity_ids (tuple): The dependent entities of the current fact.
                5.operation_id (int): The operation that yielded the current fact.
            examples: [('Point', ('A',), (,), (,), 0),  # free entity
                       ('Line', ('l',), (,), (3, 6), 3),  # constructed entity
                       ('PointOnLine', ('A', 'l'), (,), (3, 6), 3),  # initial geometric relations
                       ('Perpendicular', ('l', 'm'), (7, 8, 9), (2, 3), 6),  # inferred geometric relations
                       ('Equation', 'MeasureOfAngle(lk)-1', (10, 11), (1, 4), 8)]  # algebraic relations

            self.id (dict): (predicate, instance) -> fact_id. Mapping from fact to fact_id.
            examples: {('Point', ('A',)): 0, ('PointOnLine', ('A', 'l')): 5, ('Equation', 'MeasureOfAngle(lk)-1'): 12}

            self.groups (list): operation_id -> (fact_id). A tuple of fact_id that share the same operation_id. The
            index of element is its operation_id.
            examples: [(0, 1, 2), (3, 4), (5, 6, 7, 8)]

            self.ids_of_predicate (dict): predicate -> [fact_id]. The fact_id of facts with the same predicate.
            examples: {'Point': [0, 1, 3, 4], 'PointOnLine': [5, 7]}

            self.instances_of_predicate (dict): predicate -> [instance]. The instance of facts with the same predicate.
            examples: {'Point': ['A', 'B', 'C', 'D'], 'PointOnLine': [('A', l), ('B', l)]}

            self.operations (list): operation_id -> operation. Mapping from operation_id to operation. The index of
            element is its operation_id.
            examples: ['Point(A): PointOnLine(A, l)', 'perpendicular_judgment_angle(l,k)']

            self.constructions (list): operation_id -> (constraints, dependent_syms, target_syms, solved_values). Store
            the constraints obtained from parsing each construction statement. The index of element is its operation_id.
            specifically:
                1.constraints (list): The disjunctive normal form of constraints, where each element is a conjunction
                with the form of {'eq': [], 'l': [], 'leq': [], 'g': [], 'geq': [], 'ueq': []}.
                2.dependent_syms (tuple): Tuple of symbols related to the dependency entities of the current entity,
                which are replaced with values before solving.
                3.target_syms (tuple): Tuple of symbols related to the current entity, which need to be solved.
                4.solved_values (list): List of solved values, which stores multiple values obtained from solving, where
                the values correspond one-to-one with target_syms.
            examples: [([{'eq': [A.y-l.k*A.x-l.b], 'l': [], 'leq': [], 'g': [], 'geq': [], 'ueq': []}],
                       (l.k, l.b), (A.x, A.y), [(0, 0), (1, 1), ..., (999, 999)]),  # infinite solutions
                       ([{'eq': [A.y-l.k*A.x-l.b, B.y-l.k*B.x-l.b], 'l': [], 'leq': [], 'g': [], 'geq': [], 'ueq': []}],
                       (A.x, A.y, B.x, B.y), (l.k, l.b), [(1, 0)])]  # finite solutions

            self.sym_of_attr (dict): attr -> sym. The mapping from attribute names to attribute symbols.
            examples: {'A.x': A.x, 'l.k': l.k,
                       'LengthOfSegment(AB)': LengthOfSegment(AB), 'LengthOfSegment(BA)': LengthOfSegment(AB)}

            self.attr_of_sym (dict): sym -> [attr]. The mapping from attribute symbols to attribute names. A attribute
            symbol may have multiple attribute names.
            examples: {A.x: ['A.x'], l.k: ['l.k'],  # parameter of entity
                       LengthOfSegment(AB): ['LengthOfSegment(AB)', 'LengthOfSegment(BA)']}  # attribution of entity

            self.value_of_sym (dict): sym -> value. The value of a symbol.
            examples: {A.x: 0.25846578, l.k: 0.89568755, LengthOfSegment(AB): 5, MeasureOfAngle(lk): 90}

            self.equations (list): [[simplified_equation, original_equation_fact_id, dependent_equation_fact_id]]. Store
            the simplified equation, along with the fact_id of its original equation and the dependent equations used
            in the simplification process. Taking the equation a + b - c = 0 as an example, if the fact_id of
            a + b - c = 0 and c - 2 = 0 are 1 and 2, respectively, then the element (a + b - 2, 1, [2]) would be added
            to self.equations. Note that all dependent equations are solved values. Whenever a symbol sym is
            successfully solved to obtain a value, an algebraic relation sym - value = 0 is constructed and added to the
            self.facts. Simultaneously, For each element in self.equations, all sym in the  simplified_equation are
            replaced with its value, and the fact_id of the algebraic relation sym - value = 0 is added to
            dependent_equation_fact_id. If the simplified_equation contains no unsolved symbols, remove current element
            from self.equations.
            example: [[a + b - 2, 1, [2]], [d + e, 2, []]]

        Raises:
            ValueError: pass.

        Returns:
            self.construct(entity): Construct a geometric entity (point, line, circle). If successfully construct the
            entity, return True; otherwise, return False.

            self.apply(theorem): Apply a theorem. If applying the theorem adds new conditions, return True; otherwise,
            return False.
        """
        pass
