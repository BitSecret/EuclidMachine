from em.formalgeo.tools import entity_letters, satisfy_algebra, satisfy_inequalities
from em.formalgeo.tools import parse_fact, parse_algebra, replace_paras, replace_expr, parse_disjunctive
from sympy import symbols, nonlinsolve, tan, pi, FiniteSet, EmptySet
from func_timeout import func_timeout, FunctionTimedOut
import random


class GeometricConfiguration:

    def __init__(self, parsed_gdl, random_seed=0, max_samples=1, max_epoch=1000, rate=1.2, timeout=2):
        """The <Configuration> class represents the construction and reasoning process of a geometric configuration.

        Args:
            parsed_gdl (dict): Parsed Geometry Definition Language (GDL).

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
        self.parsed_gdl = parsed_gdl
        self.letters = list(entity_letters)  # available entity letters
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.max_samples = max_samples  # Maximum number of random samples
        self.max_epoch = max_epoch  # Maximum number of random sampling iterations
        self.range = {'x_max': 1, 'x_min': -1, 'y_max': 1, 'y_min': -1}  # Coordinate range for points
        self.rate = rate  # Range expansion ratio during random sampling
        self.timeout = timeout  # Maximum tolerance time for solving algebraic premises

        self.facts = []  # fact_id -> (predicate, instance, premise_ids, entity_ids, operation_id)
        self.id = {}  # (predicate, instance) -> fact_id
        self.ids_of_predicate = {'Point': [], 'Line': [], 'Circle': [], 'Equation': []}  # predicate -> [fact_id]
        self.instances_of_predicate = {'Point': [], 'Line': [], 'Circle': [], 'Equation': []}  # predicate -> [instance]
        for relation in self.parsed_gdl['Relations']:
            self.ids_of_predicate[relation] = []
            self.instances_of_predicate[relation] = []
        self.operations = []  # operation_id -> operation
        self.groups = []  # operation_id -> [fact_id]
        self.entity_map = {}  # entity -> entity_type

        self.value_of_para_sym = {}  # sym -> value
        self.constructions = {}  # operation_id -> (t_entity, i_entities, d_entities, constraints, solved_values)

        self.value_of_attr_sym = {}  # sym -> value (solved sym)
        self.equations = []  # equation_id -> [simplified_equation, fact_id, dependent_equation_fact_ids]
        self.attr_sym_to_equations = {}  # sym -> [equation_id] (unsolved sym)

    def _add_fact(self, predicate, instance, premise_ids, entity_ids, operation_id):
        if predicate == 'Equation':
            return self._add_equation(predicate, instance, premise_ids, entity_ids, operation_id)
        elif predicate in {'Point', 'Line', 'Circle'}:
            return self._add_entity(predicate, instance, premise_ids, entity_ids, operation_id)
        elif predicate in {'SamePoint', 'SameLine', 'SameCircle'}:
            return self._add_same(predicate, instance, premise_ids, entity_ids, operation_id)
        else:
            return self._add_relation(predicate, instance, premise_ids, entity_ids, operation_id)

    def _add_entity(self, predicate, instance, premise_ids, entity_ids, operation_id):
        if (predicate, instance) in self.id:
            return False

        self._add_one(predicate, instance, premise_ids, entity_ids, operation_id)

        self.entity_map[instance[0]] = predicate

        return True

    def _add_relation(self, predicate, instance, premise_ids, entity_ids, operation_id):
        if (predicate, instance) in self.id:
            return False

        fact_id = self._add_one(predicate, instance, premise_ids, entity_ids, operation_id)

        operation_id = self._add_operation('multiple_forms')
        for indexes in self.parsed_gdl['Relations'][predicate]['multiple_forms']:
            multiple_form = tuple([instance[i] for i in indexes])
            self._add_fact(predicate, multiple_form, [fact_id], entity_ids, operation_id)

        replace = dict(zip(self.parsed_gdl['Relations'][predicate]['paras'], instance))
        operation_id = self._add_operation('auto_extend')
        for predicate, instance in self.parsed_gdl['Relations'][predicate]['extends']:
            if predicate == 'Equation':
                instance = replace_expr(instance, replace)
            else:
                instance = tuple(replace_paras(instance, replace))
            entity_ids = tuple(self._get_entity_ids(predicate, instance))
            self._add_fact(predicate, instance, (fact_id,), entity_ids, operation_id)

        return True

    def _add_equation(self, predicate, instance, premise_ids, entity_ids, operation_id):
        if len(instance.free_symbols) == 0:
            return False
        if str(instance)[0] == '-':
            instance = - instance
        if (predicate, instance) in self.id:
            return False

        fact_id = self._add_one(predicate, instance, premise_ids, entity_ids, operation_id)

        if self.operations[operation_id] in ['solve_eq', 'same_entity_extend']:
            return True

        sym_to_value = {}  # add equation to self.equations
        dependent_equation_fact_ids = []
        for sym in instance.free_symbols:
            if sym not in self.value_of_attr_sym:
                continue
            sym_to_value[sym] = self.value_of_attr_sym[sym]
            dependent_equation_fact_ids.append(self.id[('Equation', sym - self.value_of_attr_sym[sym])])
        free_symbols = instance.free_symbols - set(sym_to_value)
        if len(free_symbols) == 0:
            return True

        free_symbols = sorted(list(free_symbols), key=str)  # sorting ensures reproducibility
        instance = instance.subs(sym_to_value)
        equation_id = len(self.equations)
        self.equations.append([instance, fact_id, dependent_equation_fact_ids])

        for sym in free_symbols:
            if sym not in self.attr_sym_to_equations:
                self.attr_sym_to_equations[sym] = [equation_id]
            else:
                self.attr_sym_to_equations[sym].append(equation_id)

        return True

    def _add_same(self, predicate, instance, premise_ids, entity_ids, operation_id):
        """Add SamePoint, SameLine, and SameCircle, and remove redundant facts.
        For two equivalent entities A and B, first sort them. Then remove the entity that comes later in the sorted
        order (entity B). All facts related to entity B will be removed from ids_of_predicate and instances_of_predicate
        (but not deleted from the facts and id). Subsequently, the removed relations are replaced with relations to
        entity A and added to the facts.
        When using theorems with parameters, facts related to B (deleted entity) are derived; when using the
        parameterless form of theorems, no related facts are derived.
        The process involves three steps: First, replace B with A in all relations. Second, update symbols in
        self.attr_sym_to_equations, replacing any B with A. Finally, apply the same substitution to
        self.value_of_attr_sym and add A's value.
        """
        if instance[0] == instance[1]:
            return False

        instance = sorted([instance, (instance[1], instance[0])])[0]

        if (predicate, instance) in self.id:
            return False

        fact_id = self._add_one(predicate, instance, premise_ids, entity_ids, operation_id)

        A, B = instance

        for predicate in self.instances_of_predicate:  # replace geometric relation
            if predicate in {'SamePoint', 'SameLine', 'SameCircle', 'Equation'}:
                continue
            for i in range(len(self.instances_of_predicate[predicate]))[::-1]:
                if B in self.instances_of_predicate[predicate][i]:  # replace B with A
                    instance = tuple([e if e != B else A for e in self.instances_of_predicate[predicate][i]])
                    premise_ids = [fact_id, self.ids_of_predicate[predicate][i]]
                    entity_ids = self._get_entity_ids(predicate, instance)
                    operation_id = self._add_operation('same_entity_extend')
                    self._add_fact(predicate, instance, premise_ids, entity_ids, operation_id)  # add fact about A
                    self.instances_of_predicate[predicate].pop(i)  # delete fact about B
                    self.ids_of_predicate[predicate].pop(i)

        self.value_of_attr_sym = {}  # sym -> value (solved sym)
        self.equations = []  # equation_id -> [simplified_equation, fact_id, dependent_equation_fact_ids]
        self.attr_sym_to_equations = {}  # sym -> [equation_id] (unsolved sym)

        for b_sym in self.attr_sym_to_equations:  # replace algebraic relation
            entities, attr = str(b_sym).split('.')
            if B not in entities:
                continue
            a_sym = symbols(entities.replace(B, A) + '.' + attr)
            if a_sym in self.value_of_attr_sym:  # b_sym unknown, a_sym known: set b_sym to the same value as a_sym
                operation_id = self._add_operation('same_entity_extend')
                premise_ids = [fact_id, self.id[('Equation', a_sym - self.value_of_attr_sym[a_sym])]]
                self._set_value_of_attr_sym(b_sym, self.value_of_attr_sym[a_sym], premise_ids, operation_id)
            else:  # b_sym unknown, a_sym unknown: replace b_sym with a_sym
                for i in range(len(self.attr_sym_to_equations[b_sym])):
                    equation_id = self.attr_sym_to_equations[b_sym][i]
                    self.equations[equation_id][0] = self.equations[equation_id][0].subs({b_sym: a_sym})
                    self.equations[equation_id][2].append(fact_id)
                    # symbols may be merged, such as: AC.dpp-BC.dpp
                    if a_sym not in self.equations[equation_id][0].free_symbols:
                        self.attr_sym_to_equations[b_sym].pop(i)
                self.attr_sym_to_equations[a_sym] = self.attr_sym_to_equations[b_sym]
                self.attr_sym_to_equations.pop(b_sym)

        for b_sym in self.value_of_attr_sym:
            entities, attr = str(b_sym).split('.')
            if B not in entities:
                continue

            a_sym = symbols(entities.replace(B, A) + '.' + attr)
            if a_sym in self.value_of_attr_sym:  # b_sym known, a_sym known: skip
                continue

            operation_id = self._add_operation('same_entity_extend')  # b_sym known, a_sym unknown: set a_sym's value
            premise_ids = [fact_id, self.id[('Equation', b_sym - self.value_of_attr_sym[b_sym])]]
            self._set_value_of_attr_sym(a_sym, self.value_of_attr_sym[b_sym], premise_ids, operation_id)

        return True

    def _add_one(self, predicate, instance, premise_ids, entity_ids, operation_id):
        fact_id = len(self.facts)
        premise_ids = tuple(sorted(list(set(premise_ids))))
        entity_ids = tuple(sorted(list(set(entity_ids))))
        self.facts.append((predicate, instance, premise_ids, entity_ids, operation_id))
        self.id[(predicate, instance)] = fact_id
        self.groups[operation_id].append(fact_id)
        self.ids_of_predicate[predicate].append(fact_id)
        self.instances_of_predicate[predicate].append(instance)
        return fact_id

    def _add_operation(self, operation):
        operation_id = len(self.operations)
        self.operations.append(operation)
        self.groups.append([])
        return operation_id

    def construct(self, construction, added=True):
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
            construction (str): Constructed entities and constraints. The constraints can be either constraints defined
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
        letters = list(self.letters)  # available letters

        t_entities, d_entities, constraints, added_facts = self._parse_construction(construction, letters)
        solved_values = self._solve_constraints(t_entities, constraints)

        if len(solved_values) == 0:  # no solved entity
            return False

        if not added:
            return True

        # add entity to self.operations
        operation_id = self._add_operation(construction)

        premise_ids = []
        for dependent_entity in d_entities:
            premise_ids.append(self.id[dependent_entity])
        premise_ids = tuple(premise_ids)

        # add target_entities to self.facts
        for target_entity in t_entities:
            self._add_entity(target_entity[0], target_entity[1], premise_ids, premise_ids, operation_id)
            self.letters.remove(target_entity[1][0])  # update self.letters

        # set entity's parameter to solved value
        target_syms = self._get_para_sym_of_entities(t_entities)
        solved_value = solved_values.pop(0)
        for i in range(len(target_syms)):
            self.value_of_para_sym[target_syms[i]] = solved_value[i]
        self._update_range(t_entities)

        # add relation to self.facts
        for predicate, instance in added_facts:
            entity_ids = tuple(self._get_entity_ids(predicate, instance))
            self._add_fact(predicate, instance, premise_ids, entity_ids, operation_id)

        # add (t_entities, d_entities, constraints, solved_values) to self.constructions
        self.constructions[operation_id] = (t_entities, d_entities, constraints, tuple(solved_values))

        return True

    def _update_range(self, new_entities):
        for predicate, instance in new_entities:
            if predicate != 'Point':
                continue
            x = symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['XOfPoint']['sym']}")
            y = symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['YOfPoint']['sym']}")
            if float(self.value_of_para_sym[x]) > self.range['x_max']:
                self.range['x_max'] = self.value_of_para_sym[x]
            if float(self.value_of_para_sym[x]) < self.range['x_min']:
                self.range['x_min'] = self.value_of_para_sym[x]
            if float(self.value_of_para_sym[y]) > self.range['y_max']:
                self.range['y_max'] = self.value_of_para_sym[y]
            if float(self.value_of_para_sym[y]) < self.range['y_min']:
                self.range['y_min'] = self.value_of_para_sym[y]

    def _parse_construction(self, construction, letters):
        target_entity, constraints = construction.split(':')
        target_predicate, target_paras = parse_fact(target_entity)
        target_entity = (target_predicate, tuple(target_paras))

        implicit_entities = []  # (predicate, entity)
        dependent_entities = []  # (predicate, entity)
        parsed_constraints = []  # (predicate, instance)

        if target_predicate not in ['Point', 'Line', 'Circle']:
            e_msg = f"Incorrect entity type: '{target_predicate}'. Expected: 'Point', 'Line' and 'Circle'."
            raise Exception(e_msg)
        if target_entity in self.id:
            e_msg = f"Entity {target_entity} already exits. "
            raise Exception(e_msg)
        letters.remove(target_paras[0])

        for constraint in parse_disjunctive(constraints):
            if (constraint.startswith('Eq(') or constraint.startswith('G(')  # algebraic constraint
                    or constraint.startswith('Geq(') or constraint.startswith('L(')
                    or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):

                algebra_relation, expr = parse_algebra(constraint)

                has_target_entity = False  # EE check and Linear Construction check
                for dependent_entity in self._get_dependent_entities('Equation', expr):
                    if target_entity == dependent_entity:
                        has_target_entity = True
                    elif dependent_entity not in self.id:
                        e_msg = (f"Incorrect algebraic constraint: '{constraint}'. "
                                 f"Dependent entity {dependent_entity} not exists.")
                        raise Exception(e_msg)
                    elif dependent_entity not in dependent_entities:
                        dependent_entities.append(dependent_entity)

                if not has_target_entity:
                    e_msg = (f"Incorrect algebraic constraint: '{constraint}'. "
                             f"Target entity {target_entity} not in algebraic constraints.")
                    raise Exception(e_msg)

                parsed_constraints.append((algebra_relation, expr))  # pass check, add constraint
            else:  # predefined constraint
                constraint_name, constraint_paras = parse_fact(constraint)

                # check the length of constraint para
                if len(constraint_paras) != len(self.parsed_gdl['Relations'][constraint_name]["paras"]):
                    e_msg = (f"Incorrect number of paras: '{constraint}'. "
                             f"Expected: {len(self.parsed_gdl['Relations'][constraint_name]['paras'])}, "
                             f"Actual: {len(constraint_paras)}.")
                    raise Exception(e_msg)

                if constraint_name in {'FreePoint', 'FreeLine', 'FreeCircle'}:  # Free entity
                    if constraint_paras[0] != target_paras[0]:
                        e_msg = f"Target entity {target_entity} not in the constraint '{constraint}'."
                        raise Exception(e_msg)
                    return [target_entity], [], [], [(constraint_name, tuple(constraint_paras))]

                has_target_entity = False
                for i in range(len(constraint_paras)):  # parse constraint
                    predicate = self.parsed_gdl['Relations'][constraint_name]["ee_checks"][i]

                    if len(constraint_paras[i]) == 1:  # norm form
                        dependent_entity = (predicate, (constraint_paras[i],))
                        if target_entity == dependent_entity:
                            has_target_entity = True
                        elif dependent_entity not in self.id:
                            e_msg = (f"Incorrect relation constraint: '{constraint}'. "
                                     f"Dependent entity {dependent_entity} not exists.")
                            raise Exception(e_msg)
                        elif dependent_entity not in dependent_entities:
                            dependent_entities.append(dependent_entity)
                    else:  # temporary form
                        has_target_entity_temp, implicit_entity = self._parse_temporary_entity(
                            predicate, constraint_paras[i], target_entity, letters,
                            implicit_entities, dependent_entities, parsed_constraints
                        )
                        constraint_paras[i] = implicit_entity  # set it to norm form
                        has_target_entity = has_target_entity_temp or has_target_entity

                if not has_target_entity:
                    e_msg = f"Target entity {target_entity} not in the constraint '{constraint}'."
                    raise Exception(e_msg)

                parsed_constraints.append((constraint_name, tuple(constraint_paras)))

        constraints = []  # (algebra_relation, expr)
        added_facts = []  # (predicate, instance)
        for predicate, instance in parsed_constraints:
            if type(instance) is not tuple:
                constraints.append((predicate, instance))
            else:
                added_facts.append((predicate, instance))  # add current constraint
                relation = self.parsed_gdl['Relations'][predicate]
                replace = dict(zip(relation['paras'], instance))

                # add implicit entities
                for implicit_predicate in relation['implicit_entities']:
                    for implicit_instance in relation['implicit_entities'][implicit_predicate]:
                        if implicit_instance in letters:
                            implicit_entities.append((implicit_predicate, (implicit_instance,)))
                            letters.remove(implicit_instance)
                        else:
                            replaced_instance = letters.pop(0)
                            replace[implicit_instance] = replaced_instance
                            implicit_entities.append((implicit_predicate, (replaced_instance,)))

                # merge fact: add constraint's implicit extends
                for implicit_predicate, implicit_instance in relation['implicit_extends']:
                    if implicit_predicate == 'Equation':
                        implicit_instance = replace_expr(implicit_instance, replace)
                    else:
                        implicit_instance = tuple(replace_paras(implicit_instance, replace))
                    added_facts.append((implicit_predicate, implicit_instance))

                # merge constraints
                for algebra_relation, expr in relation['constraints']:
                    expr = replace_expr(expr, replace)
                    constraints.append((algebra_relation, expr))

        return [target_entity] + implicit_entities, dependent_entities, constraints, added_facts

    def _parse_temporary_entity(self, predicate, temporary_entity, target_entity, letters,
                                implicit_entities, dependent_entities, parsed_constraints):
        has_target_entity = False

        if predicate == 'Line' and len(temporary_entity) == 2:  # Line(AB)
            if target_entity[0] == 'Point' and target_entity[1][0] in temporary_entity:
                has_target_entity = True
            if ('Point', (temporary_entity[0],)) not in self.id or ('Point', (temporary_entity[1],)) not in self.id:
                e_msg = f"Some entities in temporary entity '{temporary_entity}' do not exist."
                raise Exception(e_msg)

            dependent_entities.append(('Point', (temporary_entity[0],)))
            dependent_entities.append(('Point', (temporary_entity[1],)))

            implicit_entity = letters.pop(0)  # add temporary entity to implicit_entities
            implicit_entities.append(('Line', (implicit_entity,)))
            parsed_constraints.append(('PointOnLine', (temporary_entity[0], implicit_entity)))
            parsed_constraints.append(('PointOnLine', (temporary_entity[1], implicit_entity)))

            return has_target_entity, implicit_entity

        elif predicate == 'Line' and len(temporary_entity) == 3:  # Line(A;l)

            if target_entity[0] == 'Point' and target_entity[1][0] == temporary_entity[0]:
                has_target_entity = True
            if target_entity[0] == 'Line' and target_entity[1][0] == temporary_entity[2]:
                has_target_entity = True

            if ('Point', (temporary_entity[0],)) not in self.id or ('Line', (temporary_entity[2],)) not in self.id:
                e_msg = f"Some entities in temporary entity '{temporary_entity}' do not exist."
                raise Exception(e_msg)

            dependent_entities.append(('Point', (temporary_entity[0],)))
            dependent_entities.append(('Line', (temporary_entity[2],)))

            implicit_entity = letters.pop(0)  # add temporary entity to implicit_entities
            implicit_entities.append(('Line', (implicit_entity,)))
            parsed_constraints.append(('PointOnLine', (temporary_entity[0], implicit_entity)))
            parsed_constraints.append(('Parallel', (temporary_entity[0], implicit_entity)))

            return has_target_entity, implicit_entity

        elif predicate == 'Circle' and len(temporary_entity) == 3:  # Circle(ABC)
            if target_entity[0] == 'Point' and target_entity[1][0] == temporary_entity[0]:
                has_target_entity = True
            if target_entity[0] == 'Point' and target_entity[1][0] == temporary_entity[1]:
                has_target_entity = True
            if target_entity[0] == 'Point' and target_entity[1][0] == temporary_entity[2]:
                has_target_entity = True

            if (('Point', (temporary_entity[0],)) not in self.id or ('Point', (temporary_entity[1],)) not in self.id
                    or ('Point', (temporary_entity[2],)) not in self.id):
                e_msg = f"Some entities in temporary entity '{temporary_entity}' do not exist."
                raise Exception(e_msg)

            dependent_entities.append(('Point', (temporary_entity[0],)))
            dependent_entities.append(('Point', (temporary_entity[1],)))
            dependent_entities.append(('Point', (temporary_entity[2],)))

            implicit_entity = letters['Circle'].pop(0)
            implicit_entities.append(('Circle', (implicit_entity,)))
            parsed_constraints.append(('PointOnCircle', (temporary_entity[0], implicit_entity)))
            parsed_constraints.append(('PointOnCircle', (temporary_entity[1], implicit_entity)))
            parsed_constraints.append(('PointOnCircle', (temporary_entity[2], implicit_entity)))

            return has_target_entity, temporary_entity

        elif len(temporary_entity) == 4 and predicate == 'Circle':  # Circle(O;AB)
            if target_entity[0] == 'Point' and target_entity[1][0] == temporary_entity[0]:
                has_target_entity = True
            if target_entity[0] == 'Point' and target_entity[1][0] == temporary_entity[2]:
                has_target_entity = True
            if target_entity[0] == 'Point' and target_entity[1][0] == temporary_entity[3]:
                has_target_entity = True

            if (('Point', (temporary_entity[0],)) not in self.id or ('Point', (temporary_entity[1],)) not in self.id
                    or ('Point', (temporary_entity[2],)) not in self.id):
                e_msg = f"Some entities in temporary entity '{temporary_entity}' do not exist."
                raise Exception(e_msg)

            dependent_entities.append(('Point', (temporary_entity[0],)))
            dependent_entities.append(('Point', (temporary_entity[2],)))
            dependent_entities.append(('Point', (temporary_entity[3],)))

            implicit_entity = letters['Circle'].pop(0)
            implicit_entities.append(('Circle', (implicit_entity,)))
            parsed_constraints.append(('PointIsCircleCenter', (temporary_entity[0], implicit_entity)))
            parsed_constraints.append(('PointOnCircle', (temporary_entity[2], implicit_entity)))
            parsed_constraints.append(('PointOnCircle', (temporary_entity[3], implicit_entity)))

            return has_target_entity, temporary_entity

        else:
            e_msg = f"Incorrect temporary form '{temporary_entity}' for '{predicate}'."
            raise Exception(e_msg)

    def _solve_constraints(self, target_entities, constraints):
        solved_values = []  # list of values, such as [[1, 0.5], [1.5, 0.5]]
        constraint_values = []  # list of constraint values, contains symbols, such as [[y, y - 1], [x, 0.5]]

        replaced_equations = []  # expr
        replaced_inequalities = []  # (algebra_relation, expr)

        for algebra_relation, expr in constraints:
            expr = expr.subs(self.value_of_para_sym)
            if algebra_relation == 'Eq':
                replaced_equations.append(expr)
            else:
                replaced_inequalities.append((algebra_relation, expr))

        target_syms = self._get_para_sym_of_entities(target_entities)
        if len(replaced_equations) == 0:  # free entity
            constraint_values.append(target_syms)
        else:
            try:
                solved_results = func_timeout(
                    timeout=self.timeout,
                    func=nonlinsolve,
                    args=(replaced_equations, target_syms)
                )
            except FunctionTimedOut:
                return solved_values
            if type(solved_results) is not FiniteSet:
                return solved_values

            for solved_value in list(solved_results):
                has_free_symbol = False
                for item in solved_value:
                    if len(item.free_symbols) > 0:
                        has_free_symbol = True
                        break
                if has_free_symbol:  # has free sym
                    constraint_values.append(solved_value)
                elif satisfy_inequalities(replaced_inequalities, dict(zip(target_syms, solved_value))):  # no free sym
                    solved_values.append([value.evalf(n=15, chop=False) for value in solved_value])

        if len(constraint_values) == 0:
            return solved_values

        epoch = 0
        while len(solved_values) < self.max_samples and epoch < self.max_epoch:  # random sampling
            constraint_value = constraint_values[epoch % len(constraint_values)]
            solved_value = self._random_value(target_syms, constraint_value)
            sym_to_value = dict(zip(target_syms, solved_value))
            if satisfy_inequalities(replaced_inequalities, sym_to_value):
                solved_values.append(solved_value)
            epoch += 1

        return solved_values

    def _get_para_sym_of_entities(self, entities):
        syms = []  # symbols of parameter
        for predicate, instance in entities:
            if predicate == 'Point':
                syms.append(symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['XOfPoint']['sym']}"))
                syms.append(symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['YOfPoint']['sym']}"))
            elif predicate == 'Line':
                syms.append(symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['KOfLine']['sym']}"))
                syms.append(symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['BOfLine']['sym']}"))
            else:
                syms.append(symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['UOfCircle']['sym']}"))
                syms.append(symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['VOfCircle']['sym']}"))
                syms.append(symbols(f"{instance[0]}.{self.parsed_gdl['Measures']['ROfCircle']['sym']}"))
        return syms

    def _random_value(self, syms, constraint_value):
        random_values = {}
        free_symbols = set()
        for i in range(len(syms)):  # save k for sampling b
            if len(constraint_value[i].free_symbols) == 0:
                random_values[syms[i]] = float(constraint_value[i])
            else:
                free_symbols.update(constraint_value[i].free_symbols)
        free_symbols = sorted(list(free_symbols), key=str)  # sorting ensures reproducibility

        for sym in free_symbols:  # sample k first, because the value of k is used when sampling b
            if str(sym).split('.')[1] != 'k':
                continue
            random_k = tan(random.uniform(-89, 89) * pi / 180)
            random_values[sym] = random_k

        for sym in free_symbols:
            if str(sym).split('.')[1] in ['x', 'cx']:
                middle_x = (self.range['x_max'] + self.range['x_min']) / 2
                range_x = (self.range['x_max'] - self.range['x_min']) / 2 * self.rate
                random_x = random.uniform(float(middle_x - range_x), float(middle_x + range_x))
                random_values[sym] = random_x
            elif str(sym).split('.')[1] in ['y', 'cy']:
                middle_y = (self.range['y_max'] + self.range['y_min']) / 2
                range_y = (self.range['y_max'] - self.range['y_min']) / 2 * self.rate
                random_y = random.uniform(float(middle_y - range_y), float(middle_y + range_y))
                random_values[sym] = random_y
            elif str(sym).split('.')[1] == 'r':
                max_distance = float(((self.range['y_max'] - self.range['y_min']) ** 2 +
                                      (self.range['x_max'] - self.range['x_min']) ** 2) ** 0.5) / 2 * self.rate
                random_r = random.uniform(0, max_distance)
                random_values[sym] = random_r
            elif str(sym).split('.')[1] == 'b':
                middle_x = (self.range['x_max'] + self.range['x_min']) / 2
                range_x = (self.range['x_max'] - self.range['x_min']) / 2 * self.rate
                middle_y = (self.range['y_max'] + self.range['y_min']) / 2
                range_y = (self.range['y_max'] - self.range['y_min']) / 2 * self.rate

                k_value = random_values[symbols(str(sym).split('.')[0] + '.k')]
                b_range = [float(middle_y + range_y - k_value * (middle_x - range_x)),
                           float(middle_y - range_y - k_value * (middle_x - range_x)),
                           float(middle_y + range_y - k_value * (middle_x + range_x)),
                           float(middle_y - range_y - k_value * (middle_x + range_x))]
                random_b = random.uniform(min(b_range), max(b_range))
                random_values[sym] = random_b

        solved_value = [item.subs(random_values).evalf(n=15, chop=False) for item in constraint_value]

        return solved_value

    def _get_dependent_entities(self, predicate, instance):
        dependent_entities = set()  # (predicate, instance)
        if predicate == 'Equation':
            for sym in instance.free_symbols:
                entities, sym = str(sym).split('.')
                measure_name = self.parsed_gdl['sym_to_measure'][sym]
                for predicate, instance in zip(self.parsed_gdl['Measures'][measure_name]['ee_check'], list(entities)):
                    dependent_entities.add((predicate, (instance,)))
        else:
            for predicate, instance in zip(self.parsed_gdl['Relations'][predicate]['ee_check'], instance):
                dependent_entities.add((predicate, (instance,)))
        return sorted(list(dependent_entities), key=lambda x: x[1])

    def _get_existed_dependent_entities(self, predicate, instance):
        entities = set()
        if predicate == 'Equation':
            for sym in instance.free_symbols:
                entity, sym = str(sym).split('.')
                entities.update(list(entity))
        else:
            entities.update(instance)

        dependent_entities = []
        for entity in entities:
            dependent_entities.append((self.entity_map[entity], (entity,)))

        return sorted(dependent_entities, key=lambda x: x[1])

    def _get_entity_ids(self, predicate, instance):
        entity_ids = []
        for dependent_entity in self._get_existed_dependent_entities(predicate, instance):
            entity_ids.append(self.id[dependent_entity])
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
        if '(' in theorem:
            theorem_name, theorem_paras = parse_fact(theorem)
        else:
            theorem_name = theorem
            theorem_paras = None
        if theorem_name not in self.parsed_gdl["Theorems"]:
            e_msg = f"Unknown theorem name: '{theorem_name}'."
            raise Exception(e_msg)

        added = False
        if theorem_paras is not None:  # parameterized form
            if len(theorem_paras) != len(self.parsed_gdl["Theorems"][theorem_name]['paras']):
                e_msg = f"Theorem '{theorem_name}' has wrong number of paras."
                raise Exception(e_msg)

            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
            replace = dict(zip(theorem_gdl['paras'], theorem_paras))
            premise_ids = []

            for gpl_one_term in theorem_gdl['gpl']:  # run gdl with theorem parameter ()
                product = gpl_one_term['product']
                ac_checks = gpl_one_term['ac_checks']
                geometric_premises = gpl_one_term['geometric_premises']
                algebraic_premises = gpl_one_term['algebraic_premises']
                predicate = product[0]
                instance = tuple(replace_paras(product[1], replace))

                if (predicate, instance) not in self.id:  # verification mode, not cartesian product
                    return False
                premise_ids.append(self.id[(predicate, instance)])

                # check constraints
                passed, premise_ids = self._pass_constraints(
                    geometric_premises, ac_checks, algebraic_premises, replace)
                if not passed:
                    return False
                premise_ids.extend(premise_ids)

            # add operation
            operation_id = self._add_operation(theorem)

            # add conclusions
            added = self._add_conclusions(theorem_gdl['conclusions'], replace, premise_ids, operation_id) or added
        else:  # parameter-free form
            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]

            paras, instances, premise_ids = self._run_gpl(theorem_gdl['gpl'])
            for i in range(len(instances)):
                replace = dict(zip(paras, instances[i]))

                # add operation
                theorem_paras = replace_paras(theorem_gdl['paras'], replace)
                operation_id = self._add_operation(theorem_name + '(' + ','.join(theorem_paras) + ')')

                # add conclusions
                added = self._add_conclusions(
                    theorem_gdl['conclusions'], replace, premise_ids[i], operation_id
                ) or added

        return added

    def _run_gpl(self, gpl):
        paras = []
        instances = [[]]
        premise_ids = [[]]

        for gpl_one_term in gpl:
            product = gpl_one_term['product']  # (predicate, paras, inherent_same_index, mutual_same_index, added_index)
            ac_checks = gpl_one_term['ac_checks']  # [(relation_type, expr)]
            geometric_premises = gpl_one_term['geometric_premises']  # [(predicate, paras)]
            algebraic_premises = gpl_one_term['algebraic_premises']  # [expr]

            new_instances = []
            new_premise_ids = []
            paras.extend([product[1][j] for j in product[4]])
            for k in range(len(instances)):
                instance = instances[k]
                for product_instance in self.instances_of_predicate[product[0]]:
                    # check inherent same index constraint
                    passed = True
                    for i, j in product[2]:
                        if product_instance[i] != product_instance[j]:
                            passed = False
                            break
                    if not passed:
                        continue

                    # check mutual same index constraint
                    passed = True
                    for i, j in product[3]:
                        if instance[i] != product_instance[j]:
                            passed = False
                            break
                    if not passed:
                        continue

                    # constrained cartesian product: add different letter
                    new_instance = list(instance)
                    new_instance.extend([product_instance[j] for j in product[4]])

                    replace = dict(zip(paras, new_instance))

                    # check constraints
                    passed, constraints_premise_id = self._pass_constraints(
                        geometric_premises, ac_checks, algebraic_premises, replace)
                    if not passed:
                        continue

                    new_premise_id = list(premise_ids[k])
                    new_premise_id.append(self.id[(product[0], product_instance)])
                    new_premise_id.extend(constraints_premise_id)

                    new_instances.append(new_instance)
                    new_premise_ids.append(new_premise_id)

            instances = new_instances
            premise_ids = new_premise_ids

        return paras, instances, premise_ids

    def _pass_constraints(self, geometric_premises, ac_checks, algebraic_premises, replace):
        premise_ids = []

        # check geometric premises
        for predicate, paras in geometric_premises:
            fact = (predicate, tuple(replace_paras(paras, replace)))
            if fact not in self.id:
                return False, None
            premise_ids.append(self.id[fact])

        # check algebraic constraint of dependent entity
        for algebraic_relation, expr in ac_checks:
            expr = replace_expr(expr, replace)
            if not satisfy_algebra[algebraic_relation](expr, self.value_of_para_sym):
                return False, None

        # check algebraic premises
        for expr in algebraic_premises:
            expr = replace_expr(expr, replace)

            if ('Equation', expr) in self.id:  # expr in self.facts
                premise_ids.append(self.id[('Equation', expr)])
                continue

            syms, equations, premise_id = self._get_minimum_dependent_equations(expr)
            try:
                solved_values = func_timeout(
                    timeout=self.timeout,
                    func=nonlinsolve,
                    args=(equations, syms)
                )
            except FunctionTimedOut:
                return False, None
            if solved_values is EmptySet:
                e_smg = f'Equations no solution: {equations}'
                raise Exception(e_smg)

            if type(solved_values) is not FiniteSet:
                return False, None

            solved_values = list(solved_values)

            for solved_value in solved_values:
                if solved_value[0] != 0:  # t must equal to 0 in every solved value
                    return False, None

            operation_id = self._add_operation('solve_eq')  # add the solved values of symbols
            for j in range(1, len(syms)):  # skip symbol t
                if len(solved_values[0][j].free_symbols) != 0:  # no numeric solution
                    continue

                same = True
                for i in range(1, len(solved_values)):
                    if solved_values[i][j] != solved_values[0][j]:
                        same = False
                        break
                if not same:  # numeric solution not same in every solved result
                    continue

                self._set_value_of_attr_sym(syms[j], solved_values[0][j], premise_id, operation_id)

            premise_ids += premise_id

        return True, premise_ids

    def _get_minimum_dependent_equations(self, target_expr):
        syms = [symbols('t')]
        premise_ids = []
        for sym in sorted(list(target_expr.free_symbols), key=str):  # sorting ensures reproducibility
            if sym not in self.value_of_attr_sym:
                syms.append(sym)
            else:
                premise_ids.append(self.id[('Equation', sym - self.value_of_attr_sym[sym])])
                target_expr = target_expr.subs({sym: self.value_of_attr_sym[sym]})
        equations = [syms[0] - target_expr]

        i = 1
        while i < len(syms):  # find all related equations
            if syms[i] in self.attr_sym_to_equations:
                for equation_id in self.attr_sym_to_equations[syms[i]]:
                    if self.equations[equation_id][0] in equations:
                        continue

                    for sym in sorted(list(self.equations[equation_id][0].free_symbols), key=str):
                        if sym not in syms:
                            syms.append(sym)

                    equations.append(self.equations[equation_id][0])
                    premise_ids.append(self.equations[equation_id][1])
                    premise_ids.extend(self.equations[equation_id][2])
            i += 1

        return syms, equations, premise_ids

    def _set_value_of_attr_sym(self, sym, value, premise_ids, operation_id):
        if sym in self.value_of_attr_sym:
            return False

        self.value_of_attr_sym[sym] = value
        entity_ids = self._get_entity_ids('Equation', sym - value)
        added = self._add_fact('Equation', sym - value, premise_ids, entity_ids, operation_id)

        for equation_id in self.attr_sym_to_equations[sym]:
            self.equations[equation_id][0] = self.equations[equation_id][0].subs({sym: value})
            self.equations[equation_id][2].append(self.id[('Equation', sym - value)])
        self.attr_sym_to_equations.pop(sym)

        return added

    def _add_conclusions(self, conclusions, replace, premise_ids, operation_id):
        add_new_fact = False
        for predicate, instance in conclusions:
            if predicate == "Equation":
                instance = replace_expr(instance, replace)
                if len(instance.free_symbols) == 0:
                    continue
            else:
                instance = tuple(replace_paras(instance, replace))
            entity_ids = self._get_entity_ids(predicate, instance)
            add_new_fact = self._add_fact(predicate, instance, premise_ids, entity_ids, operation_id) or add_new_fact
        return add_new_fact
