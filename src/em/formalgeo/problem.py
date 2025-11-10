from em.formalgeo.tools import letters_p, letters_l, letters_c
from em.formalgeo.tools import parse_predicate, parse_algebra, replace_paras, replace_expr
from sympy import symbols, nonlinsolve, tan, pi, nsimplify, sqrt
import random


def _get_sym_from_entities(entities):
    syms = []  # symbols of parameter
    for predicate, instance in entities:
        if predicate == 'Point':
            syms.append(symbols(f"{instance}.x"))
            syms.append(symbols(f"{instance}.y"))
        elif predicate == 'Line':
            syms.append(symbols(f"{instance}.k"))
            syms.append(symbols(f"{instance}.b"))
        else:
            syms.append(symbols(f"{instance}.cx"))
            syms.append(symbols(f"{instance}.cy"))
            syms.append(symbols(f"{instance}.r"))
    return syms


def _get_dependent_entities(predicate, instance, parsed_gdl):
    dependent_entities = []  # (predicate, instance)
    if predicate == 'Equation':
        for sym in instance.free_symbols:
            entities, sym = str(sym).split('.')
            measure = parsed_gdl['Measure'][parsed_gdl['sym_to_measure'][sym]]
            for predicate, instance in zip(measure['ee_check'], list(entities)):
                dependent_entities.append((predicate, instance))
    else:
        for i in range(len(instance)):
            dependent_entities.append((parsed_gdl['Relations'][predicate]['ee_check'][i], instance[i]))
    return list(set(dependent_entities))


def _satisfy_constraints(sym_to_value, constraints):
    """Satisfy Inequalities"""
    for g in constraints['G']:
        if g.subs(sym_to_value).evalf(chop=True) <= 0:
            return False
    for geq in constraints['Geq']:
        if geq.subs(sym_to_value).evalf(chop=True) < 0:
            return False
    for l in constraints['L']:
        if l.subs(sym_to_value).evalf(chop=True) >= 0:
            return False
    for leq in constraints['Leq']:
        if leq.subs(sym_to_value).evalf(chop=True) > 0:
            return False
    for ueq in constraints['Ueq']:
        if ueq.subs(sym_to_value).evalf(chop=True) == 0:
            return False
    return True


def _get_free_symbols(constraint_value):
    free_symbols = set()
    for item in constraint_value:
        free_symbols = free_symbols | item.free_symbols
    return list(free_symbols)


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
        examples: [('Point', 'A', (,), (,), 0),  # free entity
                   ('Line', 'l', (,), (3, 6), 3),  # constructed entity
                   ('PointOnLine', ('A', 'l'), (3,), (3, 6), 3),  # initial geometric relations
                   ('Perpendicular', ('l', 'm'), (7, 8, 9), (2, 3), 6),  # inferred geometric relations
                   ('Equation', 'lk.ma-1', (10, 11), (1, 4), 8)]  # algebraic relations

        self.id (dict): (predicate, instance) -> fact_id. Mapping from fact to fact_id.
        examples: {('Point', ('A',)): 0, ('PointOnLine', ('A', 'l')): 5, ('Equation', 'MeasureOfAngle(lk)-1'): 12}

        self.groups (list): operation_id -> [fact_id]. A tuple of fact_id that share the same operation_id. The
        index of element is its operation_id.
        examples: [[0, 1, 2], [3, 4], [5, 6, 7, 8]]

        self.ids_of_predicate (dict): predicate -> [fact_id]. The fact_id of facts with the same predicate.
        examples: {'Point': [0, 1, 3, 4], 'PointOnLine': [5, 7]}

        self.instances_of_predicate (dict): predicate -> [instance]. The instance of facts with the same predicate.
        examples: {'Point': ['A', 'B', 'C', 'D'], 'PointOnLine': [('A', l), ('B', l)]}

        self.operations (list): operation_id -> operation. Mapping from operation_id to operation. The index of
        element is its operation_id.
        examples: ['Point(A): PointOnLine(A, l)', 'perpendicular_judgment_angle(l,k)']

        self.constructions (list): operation_id -> (constraints, target_entities, dependent_entities, solved_values).
        Store the constraints obtained from parsing each construction statement. The index of element is its
        operation_id. specifically:
            1.constraints (list): The disjunctive normal form of constraints, where each element is a conjunction
            with the form of {'eq': [], 'l': [], 'leq': [], 'g': [], 'geq': [], 'ueq': []}.
            2.target_entities (dict): Tuple of symbols related to the current entity, which need to be solved.
            3.dependent_entities (dict): Tuple of symbols related to the dependency entities of the current entity,
            which are replaced with values before solving.
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

    def __init__(self, parsed_gdl, random_seed=0, tolerance=1e-3, max_samples=1, max_epoch=1000, rate=1.2):
        """Problem conditions, goal, and solving message.

        Args:
            parsed_gdl (dict): Parsed Geometry Definition Language (GDL).
        """
        self.parsed_gdl = parsed_gdl
        self.letters = {'Point': list(letters_p),
                        'Line': list(letters_l),
                        'Circle': list(letters_c)}  # available letters
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.tolerance = tolerance  # 随机采样近似分数时的容忍误差
        self.max_samples = max_samples  # 随机样本的最大数量
        self.max_epoch = max_epoch  # 随机采样的最大次数
        self.range = {'x_max': 1, 'x_min': -1, 'y_max': 1, 'y_min': -1}  # 点坐标的范围
        self.rate = rate  # 随机采样时的范围扩大率

        self.facts = []  # fact_id -> (predicate, instance, premise_ids, entity_ids, operation_id)
        self.id = {}  # (predicate, instance) -> fact_id
        self.groups = []  # operation_id -> [fact_id]
        self.ids_of_predicate = {'Point': [], 'Line': [], 'Circle': [], 'Equation': []}  # predicate -> [fact_id]
        self.instances_of_predicate = {'Point': [], 'Line': [], 'Circle': [], 'Equation': []}  # predicate -> [instance]
        for relation in self.parsed_gdl['relations']:
            self.ids_of_predicate[relation] = []
            self.instances_of_predicate[relation] = []
        self.operations = []  # operation_id -> operation

        self.constructions = []  # operation_id -> (t_entity, i_entities, d_entities, constraints, solved_values)

        self.value_of_sym = {}  # sym -> value
        self.equations = []  # [[simplified_equation, original_equation_fact_id, dependent_equation_fact_ids]]

    def _has(self, predicate, instance):
        return instance in self.instances_of_predicate[predicate]

    def _add(self, predicate, instance, premise_ids, entity_ids, operation_id):
        if self._has(predicate, instance):
            return False

        fact_id = len(self.facts)
        self.facts.append((predicate, instance, tuple(set(premise_ids)), tuple(set(entity_ids)), operation_id))
        self.id[(predicate, instance)] = fact_id
        self.groups[operation_id].append(fact_id)
        self.ids_of_predicate[predicate].append(fact_id)
        self.instances_of_predicate[predicate].append(instance)

        if predicate == 'Equation':
            return True

        replace = dict(zip(self.parsed_gdl['Relations'][predicate]['paras'], instance))

        operation_id = len(self.operations)
        self.operations.append('auto_extend')
        self.groups[operation_id] = []

        for predicate, instance in self.parsed_gdl['Relations'][predicate]['extend']:
            if predicate == 'Equation':
                instance = replace_expr(instance, replace)
            else:
                instance = tuple(replace_paras(instance, replace))
            entity_ids = tuple(self._get_entity_ids(predicate, instance))
            self._add(predicate, instance, (fact_id,), entity_ids, operation_id)

        return True

    def construct(self, entity):
        """Construct a new point, line, or circle.
        1.Parse the geometric construction statement, extract the target entity, implicit entities, and dependent
        entities, perform entity existence checks, format validity checks, and linear construction checks, and add the
        relations that do not contain implicit entities to the added_facts.

        2.Combine all constraints together, and update the implicit entities during the merging process.

        3.Solve the constraints to obtain the values of all unknown variables. For results involving random variables,
        generate random sampled values. Variables related to implicit entities, although solved for, will be ignored.
        Only the values of the variables related to the target entities are ultimately returned. This aligns with the
        first step of discarding relations that contain implicit entities. If an entity is implicit, we are not
        concerned with the implicit entity itself or its associated relations. The purpose of implicit entities is
        solely to transmit constraint relationships between the target entities and the dependent entities.

        4.Add all content generated by the current geometric construction statement to the database. This includes the
        current action, newly generated entities and their parameter values, and the generated relations, etc. Note that
        we also record the current constraints and their other solved values. This is primarily to serve the rollback
        functionality in future versions and is not used in the current version.

        Args:
            entity (str): Constructed entities and their constraints. The constraints can be either constraints defined
            in GDL or algebraic constraints. The constraints need to be connected by '&'. Each constraint must include
            the entity currently being constructed, and the remaining entities must all be known entities. Furthermore,
            the temporary forms of entities are also permitted for use here.
            examples: 'Point(A):FreePoint(A)'  # free entity
                      'Line(l):PointOnLine(A,l)&PointOnLine(B,l)'  # constraints defined in GDL
                      'Point(C):Eq(Sub(DPP(C.x,C.y,A.x,A.y),Mul(DPP(C.x,C.y,B.x,B.y),2)))'  # algebraic constraints
                      'Point(D):PointOnLine(D,AC)'  # temporary forms of Line

        Returns:
            result (bool): If successfully construct the entity, return True; otherwise, return False.
        """
        letters = {'Point': list(self.letters['Point']),
                   'Line': list(self.letters['Line']),
                   'Circle': list(self.letters['Circle'])}  # available letters

        t_entity, i_entities, d_entities, parsed_constraints, added_facts = self._parse_entity(entity, letters)
        i_entities, constraints = self._merge_constraints(i_entities, parsed_constraints, letters)
        solved_values = self._solve_constraints(t_entity, i_entities, d_entities, constraints)

        if len(solved_values) == 0:  # No solved entity
            return False

        # add entity to self.operations
        operation_id = len(self.operations)
        self.operations.append(entity)
        self.groups[operation_id] = []

        premise_ids = []
        for predicate in d_entities:
            for instance in d_entities[predicate]:
                premise_ids.append(self.id[(predicate, (instance,))])
        premise_ids = tuple(premise_ids)

        # add target_entities to self.facts
        self._add(t_entity[0], (t_entity[1],), premise_ids, premise_ids, operation_id)

        # set entity's parameter to solved value
        solved_value = solved_values.pop(0)
        syms = _get_sym_from_entities([t_entity])
        for i in range(len(solved_values)):
            self.value_of_sym[syms[i]] = solved_value[i]
        if t_entity[0] == 'Point':
            if self.value_of_sym[syms[0]] > self.range['x_max']:
                self.range['x_max'] = self.value_of_sym[syms[0]]
            if self.value_of_sym[syms[0]] < self.range['x_min']:
                self.range['x_min'] = self.value_of_sym[syms[0]]
            if self.value_of_sym[syms[1]] > self.range['y_max']:
                self.range['y_max'] = self.value_of_sym[syms[1]]
            if self.value_of_sym[syms[1]] < self.range['y_min']:
                self.range['y_min'] = self.value_of_sym[syms[1]]

        # add relation to self.facts
        for predicate, instance in added_facts:
            entity_ids = tuple(self._get_entity_ids(predicate, instance))
            self._add(predicate, instance, premise_ids, entity_ids, operation_id)

        # add (t_entity, i_entities, d_entities, constraints, solved_values) to self.constructions
        self.constructions.append((t_entity, i_entities, d_entities, constraints, tuple(solved_values)))

        # update self.letters
        self.letters[t_entity[0]].remove(t_entity[1])

        return True

    def _parse_entity(self, entity, letters):
        target_entity, constraints = entity.split(':')
        predicate, paras = parse_predicate(entity)
        target_entity = (predicate, paras[0])
        implicit_entities = []
        dependent_entities = []
        parsed_constraints = []
        added_facts = []

        if target_entity[0] not in ['Point', 'Line', 'Circle']:
            e_msg = f"Incorrect entity type: '{target_entity[0]}'. Expected: 'Point', 'Line' and 'Circle'."
            raise Exception(e_msg)
        if target_entity[1] in self.instances_of_predicate[target_entity[0]]:
            e_msg = f"Entity '{target_entity[0]}({target_entity[1]})' already exits. "
            raise Exception(e_msg)
        letters[target_entity[0]].remove(target_entity[1])

        if len(constraints) == 0:
            return (target_entity[0], target_entity[1]), implicit_entities, dependent_entities, parsed_constraints

        for constraint in constraints.split('&'):
            if (constraint.startswith('Eq(') or constraint.startswith('G(')  # algebraic constraint
                    or constraint.startswith('Geq(') or constraint.startswith('L(')
                    or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):
                algebra_relation, expr = parse_algebra(constraint)  # 约束解析

                has_target_entity = False  # 参数实体存在性检查、线性构图检查
                for predicate, instance in _get_dependent_entities('Equation', expr, self.parsed_gdl):
                    if instance == target_entity[1] and predicate == target_entity[0]:
                        has_target_entity = True
                    elif instance in letters[predicate]:
                        e_msg = (f"Incorrect algebraic constraint: '{constraint}'. "
                                 f"Dependent entity '{predicate}({instance})' not exists.")
                        raise Exception(e_msg)
                    else:
                        dependent_entities.append((predicate, instance))
                if not has_target_entity:
                    e_msg = (f"Incorrect algebraic constraint: '{constraint}'. "
                             f"Target entity '{target_entity[1]}' not in algebraic constraints.")
                    raise Exception(e_msg)

                parsed_constraints.append((algebra_relation, expr))  # 通过检查，添加到parsed_constraints
            else:  # predefined constraint
                constraint_name, constraint_paras = parse_predicate(constraint)
                if constraint_name not in self.parsed_gdl['Relations']:  # 约束是否存在
                    e_msg = f"Unknown constraint: '{constraint}."
                    raise Exception(e_msg)
                if len(constraint_paras) != len(self.parsed_gdl['Relations'][constraint_name]["paras"]):  # 参数数量是否正确
                    e_msg = (f"Incorrect number of paras: '{constraint}'. "
                             f"Expected: {len(self.parsed_gdl['Relations'][constraint_name]['paras'])}, "
                             f"Actual: {len(constraint_paras)}.")
                    raise Exception(e_msg)

                has_target_entity = False
                has_implicit_entity = False
                for i in range(len(constraint_paras)):
                    predicate = self.parsed_gdl['Relations'][constraint_name]["ee_check"][i]
                    if len(constraint_paras[i]) == 1:
                        if constraint_paras[i] == target_entity[1] and target_entity[0] == predicate:  # 是要构建的实体
                            has_target_entity = True
                        elif constraint_paras[i] not in self.instances_of_predicate[predicate]:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity '{predicate}({constraint_paras[i]})' not exists.")
                            raise Exception(e_msg)
                        else:
                            dependent_entities.append((predicate, constraint_paras[i]))
                    elif len(constraint_paras[i]) == 2 and predicate == 'Line':  # Line(AB)
                        has_implicit_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] in constraint_paras[i]:
                            has_target_entity = True
                        if constraint_paras[i][0] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][0]})' not exists.")
                            raise Exception(e_msg)
                        if constraint_paras[i][1] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][0]})' not exists.")
                            raise Exception(e_msg)

                        dependent_entities.append(('Point', constraint_paras[i][0]))
                        dependent_entities.append(('Point', constraint_paras[i][1]))

                        temporary_entity = letters['Line'].pop(0)  # 添加临时实体
                        implicit_entities.append(('Line', temporary_entity))
                        parsed_constraints.append(('PointOnLine', (constraint_paras[i][0], temporary_entity)))
                        parsed_constraints.append(('PointOnLine', (constraint_paras[i][1], temporary_entity)))
                        constraint_paras[i] = temporary_entity
                    elif len(constraint_paras[i]) == 3 and predicate == 'Line':  # Line(A;l)
                        has_implicit_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] == constraint_paras[i][0]:
                            has_target_entity = True
                        if target_entity[0] == 'Line' and target_entity[1] == constraint_paras[i][2]:
                            has_target_entity = True

                        if constraint_paras[i][0] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][0]})' not exists.")
                            raise Exception(e_msg)
                        if constraint_paras[i][2] not in self.instances_of_predicate['Line']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Line({constraint_paras[i][2]})' not exists.")
                            raise Exception(e_msg)

                        dependent_entities.append(('Point', constraint_paras[i][0]))
                        dependent_entities.append(('Line', constraint_paras[i][2]))

                        temporary_entity = letters['Line'].pop(0)  # 添加临时实体
                        implicit_entities.append(('Line', temporary_entity))
                        parsed_constraints.append(('PointOnLine', (constraint_paras[i][0], temporary_entity)))
                        parsed_constraints.append(('Parallel', (constraint_paras[i][2], temporary_entity)))
                        constraint_paras[i] = temporary_entity
                    elif len(constraint_paras[i]) == 3 and predicate == 'Circle':  # Circle(ABC)
                        has_implicit_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] == constraint_paras[i][0]:
                            has_target_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] == constraint_paras[i][1]:
                            has_target_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] == constraint_paras[i][2]:
                            has_target_entity = True

                        if constraint_paras[i][0] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][0]})' not exists.")
                            raise Exception(e_msg)
                        if constraint_paras[i][1] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][1]})' not exists.")
                            raise Exception(e_msg)
                        if constraint_paras[i][2] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][2]})' not exists.")
                            raise Exception(e_msg)

                        dependent_entities.append(('Point', constraint_paras[i][0]))
                        dependent_entities.append(('Point', constraint_paras[i][1]))
                        dependent_entities.append(('Point', constraint_paras[i][2]))

                        temporary_entity = letters['Circle'].pop(0)
                        implicit_entities.append(('Circle', temporary_entity))
                        parsed_constraints.append(('PointOnCircle', (constraint_paras[i][0], temporary_entity)))
                        parsed_constraints.append(('PointOnCircle', (constraint_paras[i][1], temporary_entity)))
                        parsed_constraints.append(('PointOnCircle', (constraint_paras[i][2], temporary_entity)))
                        constraint_paras[i] = temporary_entity
                    elif len(constraint_paras[i]) == 4 and predicate == 'Circle':  # Circle(O;AB)
                        has_implicit_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] == constraint_paras[i][0]:
                            has_target_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] == constraint_paras[i][2]:
                            has_target_entity = True
                        if target_entity[0] == 'Point' and target_entity[1] == constraint_paras[i][3]:
                            has_target_entity = True

                        if constraint_paras[i][0] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][0]})' not exists.")
                            raise Exception(e_msg)
                        if constraint_paras[i][2] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][2]})' not exists.")
                            raise Exception(e_msg)
                        if constraint_paras[i][3] not in self.instances_of_predicate['Point']:
                            e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                                     f"Dependent entity 'Point({constraint_paras[i][3]})' not exists.")
                            raise Exception(e_msg)

                        dependent_entities.append(('Point', constraint_paras[i][0]))
                        dependent_entities.append(('Point', constraint_paras[i][2]))
                        dependent_entities.append(('Point', constraint_paras[i][3]))

                        temporary_entity = letters['Circle'].pop(0)
                        implicit_entities.append(('Circle', temporary_entity))
                        parsed_constraints.append(('PointIsCircleCenter', (constraint_paras[i][0], temporary_entity)))
                        parsed_constraints.append(('PointOnCircle', (constraint_paras[i][2], temporary_entity)))
                        parsed_constraints.append(('PointOnCircle', (constraint_paras[i][3], temporary_entity)))
                        constraint_paras[i] = temporary_entity
                    else:
                        e_msg = f"Incorrect temporary form '{constraint_paras[i]}' for '{predicate}'."
                        raise Exception(e_msg)

                if not has_target_entity:
                    e_msg = (f"Incorrect geometric constraint: '{constraint}'. "
                             f"Target entity '{target_entity[1]}' not in geometric constraints.")
                    raise Exception(e_msg)
                parsed_constraints.append((constraint_name, tuple(constraint_paras)))

                if not has_implicit_entity:
                    added_facts.append((constraint_name, tuple(constraint_paras)))

        return target_entity, implicit_entities, dependent_entities, parsed_constraints, added_facts

    def _merge_constraints(self, implicit_entities, parsed_constraints, letters):
        constraints = {'Eq': [], 'G': [], 'Geq': [], 'L': [], 'Leq': [], 'Ueq': []}

        for predicate, instance in parsed_constraints:
            if type(instance) is not tuple:
                constraints[predicate].append(instance)
            else:
                relation = self.parsed_gdl['Relations'][predicate]
                replace = dict(zip(relation['paras'], instance))
                for implicit_predicate in relation['implicit_entity']:
                    for implicit_instance in relation['implicit_entity'][implicit_predicate]:
                        if implicit_instance in letters[implicit_predicate]:
                            implicit_entities.append((implicit_predicate, implicit_instance))
                            letters[implicit_predicate].remove(implicit_instance)
                        else:
                            replaced_instance = letters[implicit_predicate].pop(0)
                            replace[implicit_instance] = replaced_instance
                            implicit_entities.append((implicit_predicate, replaced_instance))
                for constraint_type in relation['constraints']:
                    for constraint in relation['constraints'][constraint_type]:
                        constraint = replace_expr(constraint, replace)
                        constraints[constraint_type].append(constraint)

        return implicit_entities, constraints

    def _solve_constraints(self, target_entity, implicit_entities, dependent_entities, constraints):
        target_syms = _get_sym_from_entities([target_entity])
        implicit_syms = _get_sym_from_entities(implicit_entities)
        solved_values = []  # list of values, such as [[1, 0.5], [1.5, 0.5]]
        constraint_values = []  # list of constraint values, contains symbols, such as [[y, y - 1], [x, 0.5]]

        sym_to_value = {}  # replace dependent entity's parameter
        for sym in _get_sym_from_entities(dependent_entities):
            sym_to_value[sym] = self.value_of_sym[sym]
        replaced_constraints = {'Eq': [], 'G': [], 'Geq': [], 'L': [], 'Leq': [], 'Ueq': []}
        for constraint_type in constraints:
            for constraint in constraints[constraint_type]:
                replaced_constraints[constraint_type].append(constraint.subs(sym_to_value))

        if len(replaced_constraints['Eq']) == 0:  # free point
            constraint_values.append(tuple(target_syms + implicit_syms))
        else:
            for solved_value in list(nonlinsolve(replaced_constraints['Eq'], target_syms + implicit_syms)):
                if len(_get_free_symbols(solved_value)) == 0:
                    if _satisfy_constraints(solved_value, replaced_constraints):
                        solved_values.append(tuple(list(solved_value)[0:len(target_syms)]))
                else:
                    constraint_values.append(solved_value)

        if len(constraint_values) == 0:
            return solved_values

        epoch = 0
        while len(solved_values) < self.max_samples and epoch < self.max_epoch:
            constraint_value = constraint_values[epoch % len(constraint_values)]
            solved_value = self._random_value(target_syms + implicit_syms, constraint_value)
            if _satisfy_constraints(solved_value, replaced_constraints):
                solved_values.append(tuple(list(solved_value)[0:len(target_syms)]))
            epoch += 1

        return solved_values

    def _random_value(self, syms, constraint_value):
        random_values = {}
        for i in range(len(syms)):  # save k for sampling b
            if len(constraint_value[i].free_symbols) == 0:
                random_values[syms[i]] = constraint_value[i]

        unsolved_syms = _get_free_symbols(constraint_value)

        for sym in unsolved_syms:  # sample k first, because the value of k is used when sampling b
            if str(sym).split('.')[1] != 'k':
                continue
            random_k = tan(random.uniform(-89, 89) * pi / 180)
            random_k = nsimplify(random_k, tolerance=self.tolerance)
            random_values[sym] = random_k

        for sym in unsolved_syms:
            if str(sym).split('.')[1] in ['x', 'cx']:
                random_x = random.uniform(float(self.range['x_max'] * self.rate),
                                          float(self.range['x_min'] * self.rate))
                random_x = nsimplify(random_x, tolerance=self.tolerance)
                random_values[sym] = random_x
            elif str(sym).split('.')[1] in ['y', 'cy']:
                random_y = random.uniform(float(self.range['y_max'] * self.rate),
                                          float(self.range['y_min'] * self.rate))
                random_y = nsimplify(random_y, tolerance=self.tolerance)
                random_values[sym] = random_y
            elif str(sym).split('.')[1] == 'r':
                max_distance = float(sqrt((self.range['y_max'] - self.range['y_min']) ** 2 +
                                          (self.range['x_max'] - self.range['x_min']) ** 2))
                random_r = random.uniform(0, max_distance)
                random_r = nsimplify(random_r, tolerance=self.tolerance)
                random_values[sym] = random_r
            elif str(sym).split('.')[1] == 'b':
                k_value = random_values[symbols(str(sym).split('.')[0] + '.k')]
                b_range = [float(self.range['y_max'] * self.rate - k_value * self.range['x_min'] * self.rate),
                           float(self.range['y_min'] * self.rate - k_value * self.range['x_min'] * self.rate),
                           float(self.range['y_max'] * self.rate - k_value * self.range['x_max'] * self.rate),
                           float(self.range['y_min'] * self.rate - k_value * self.range['x_max'] * self.rate)]
                random_b = random.uniform(min(b_range), max(b_range))
                random_b = nsimplify(random_b, tolerance=self.tolerance)
                random_values[sym] = random_b
            else:  # str(sym).split('.')[1] == 'k'
                continue

        solved_value = [item.subs(random_values) for item in constraint_value]

        return tuple(solved_value)

    def _get_entity_ids(self, predicate, instance):
        entity_ids = []
        for d_predicate, d_instance in _get_dependent_entities(predicate, instance, self.parsed_gdl):
            entity_ids.append(self.id[(d_predicate, (d_instance,))])
        return entity_ids

    def apply(self, theorem):
        """Apply a theorem.
        1.这里是详细的功能描述，可以跨越多行。
        2.第二段通常描述实现细节或算法原理。
        3.solver部分的推理，得到新的fact

        Args:
            theorem (str): Theorem to be applied. Two forms: a parameterized form and a parameter-free form.
            examples: 'adjacent_complementary_angle(l,k)'  # parameterized form
                      'adjacent_complementary_angle'  # parameter-free form

        Returns:
            result (bool): If applying the theorem adds new conditions, return True; otherwise, return False.
        """
        add_new_fact = False

        if '(' in theorem:  # parameterized form
            theorem_name, theorem_paras = parse_predicate(theorem)
            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
            replace = dict(zip(theorem_gdl['paras'], theorem_paras))

            # EE check
            passed, premise_ids = self._pass_ee_check(theorem_gdl, replace)
            if not passed:
                return False

            # AC check
            passed = self._pass_ac_check(theorem_gdl, replace)
            if not passed:
                return False

            # geometric premise
            passed, geometric_premise_ids = self._pass_geometric_premise(theorem_gdl, replace)
            if not passed:
                return False
            premise_ids += geometric_premise_ids

            # algebraic premise
            passed, algebraic_premise_ids = self._pass_algebraic_premise(theorem_gdl, replace)
            if not passed:
                return False
            premise_ids += algebraic_premise_ids

            # add theorem to self.operations
            operation_id = len(self.operations)
            self.operations.append(theorem)
            self.groups[operation_id] = []

            # add geometric conclusion
            for predicate, instance in theorem_gdl['geometric_conclusion']:
                instance = tuple(replace_paras(instance, replace))
                entity_ids = tuple(self._get_entity_ids(predicate, instance))
                add_new_fact = self._add(predicate, instance, premise_ids, entity_ids, operation_id) or add_new_fact

            # add algebraic conclusion
            for expr in theorem_gdl['algebraic_conclusion']:
                expr = tuple(replace_expr(expr, replace))
                entity_ids = tuple(self._get_entity_ids('Equation', expr))
                add_new_fact = self._add('Equation', expr, premise_ids, entity_ids, operation_id) or add_new_fact
        else:  # parameter-free form
            theorem_name = theorem
            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]

            # geometric predicate logic
            gpl = list(theorem_gdl['geometric_premise'])
            for i in range(len(theorem_gdl['paras'])):
                gpl.append((theorem_gdl['ee_check'][i], [theorem_gdl['paras'][i]]))

            # run geometric predicate logic
            results = self._run_gpl(gpl, theorem_gdl['paras'])

            # ac check
            checked_results = []
            for _, instance in results:
                pass
            results = checked_results

            # check algebraic premise
            checked_results = []
            for _, instance in results:
                pass
            results = checked_results

            # 添加到结论

        return add_new_fact

    def _pass_ee_check(self, theorem_gdl, replace):
        premise_ids = []
        for i in range(len(theorem_gdl['paras'])):
            predicate = theorem_gdl['ee_check'][i]
            instance = (replace[theorem_gdl['paras'][i]],)
            if not self._has(predicate, instance):
                return False, None
            premise_ids.append(self.id[(predicate, instance)])
        return True, premise_ids

    def _pass_ac_check(self, theorem_gdl, replace):
        constraints = {'Eq': [], 'G': [], 'Geq': [], 'L': [], 'Leq': [], 'Ueq': []}
        for algebraic_type in theorem_gdl['ac_check']:
            for constraint in theorem_gdl['ac_check'][algebraic_type]:
                constraints[algebraic_type].append(replace_expr(constraint, replace))
        return _satisfy_constraints(self.value_of_sym, constraints)

    def _pass_geometric_premise(self, theorem_gdl, replace):
        premise_ids = []
        for predicate, instance in theorem_gdl['geometric_premise']:
            instance = replace_paras(instance, replace)
            if not self._has(predicate, tuple(instance)):
                return False, None
            premise_ids.append(self.id[(predicate, tuple(instance))])
        return True, premise_ids

    def _pass_algebraic_premise(self, theorem_gdl, replace):
        constraints = []
        premise_ids = []
        for constraint in theorem_gdl['algebraic_premise']:
            constraints.append(replace_expr(constraint, replace))
        satisfied, algebraic_premise_ids = self._satisfy_algebraic_constraints(constraints)  # 定义在此函数即可
        if not satisfied:
            return False, None
        return True, premise_ids

    def _run_gpl(self, gpl, paras):
        result = []  # ([premise_ids], instance)
        return result
