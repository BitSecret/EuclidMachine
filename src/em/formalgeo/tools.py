from sympy import sympify, symbols, sqrt, atan, pi
import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 解决后端兼容性问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

"""↓------Available Entity Vocabulary------↓"""

_lu = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z')  # latin_upper 26
_lsu = ('𝓐', '𝓑', '𝓒', '𝓓', '𝓔', '𝓕', '𝓖', '𝓗', '𝓘', '𝓙', '𝓚', '𝓛', '𝓜', '𝓝', '𝓞', '𝓟', '𝓠', '𝓡', '𝓢', '𝓣', '𝓤',
        '𝓥', '𝓦', '𝓧', '𝓨', '𝓩')  # latin_script_upper26
_ll = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
       'w', 'x', 'y', 'z')  # latin_lower 26
_lsl = ('𝓪', '𝓫', '𝓬', '𝓭', '𝓮', '𝓯', '𝓰', '𝓱', '𝓲', '𝓳', '𝓴', '𝓵', '𝓶', '𝓷', '𝓸', '𝓹', '𝓺', '𝓻', '𝓼', '𝓽', '𝓾', '𝓿',
        '𝔀', '𝔁', '𝔂', '𝔃')  # latin_script_lower 26
_gu = ('Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Σ', 'Τ', 'Υ', 'Φ',
       'Χ', 'Ψ', 'Ω')  # greek_upper 24
_giu = ('𝜜', '𝜝', '𝜞', '𝜟', '𝜠', '𝜡', '𝜢', '𝜣', '𝜤', '𝜥', '𝜦', '𝜧', '𝜨', '𝜩', '𝜪', '𝜫', '𝜬', '𝜭', '𝜮', '𝜯', '𝜰', '𝜱',
        '𝜲', '𝜳', '𝜴')  # greek_italic_upper 24
_gl = ('α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ',
       'χ', 'ψ', 'ω')  # greek_lower 24
_gil = ('𝜶', '𝜷', '𝜸', '𝜹', '𝜺', '𝜻', '𝜼', '𝜽', '𝜾', '𝜿', '𝝀', '𝝁', '𝝂', '𝝃', '𝝄', '𝝅', '𝝆', '𝝇', '𝝈', '𝝉', '𝝊', '𝝋',
        '𝝌', '𝝍', '𝝎')  # greek_italic_lower 24

letters_p = tuple(list(_lu) + list(_lsu) + list(_gu) + list(_giu) + list(_ll) + list(_lsl) + list(_gl) + list(_gil))
letters_l = tuple(list(_ll) + list(_lsl) + list(_gl) + list(_gil) + list(_lu) + list(_lsu) + list(_gu) + list(_giu))
letters_c = tuple(list(_gu) + list(_giu) + list(_lu) + list(_lsu) + list(_gl) + list(_gil) + list(_ll) + list(_lsl))

"""↑------Available Entity Vocabulary------↑"""
"""↓--------------Useful Tools-------------↓"""


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


"""↑------------Useful Tools------------↑"""
"""↓------------Data Parsing------------↓"""


def parse_gdl(gdl):
    """Parse Geometry Definition Language into a usable format for Problem."""
    parsed_gdl = {
        'Entities': {},
        'sym_to_measure': {},
        'Measures': {},
        'Relations': {},
        'Theorems': {}
    }

    for entity in gdl['Entities']:
        _parse_one_entity(entity, gdl, parsed_gdl)

    for measure in gdl['Measures']:
        _parse_one_measure(measure, gdl, parsed_gdl)

    for relation in gdl['Relations']:
        _parse_one_relation(relation, gdl, parsed_gdl)

    for theorem in gdl['Theorems']:
        _parse_one_theorem(theorem, gdl, parsed_gdl)

    return parsed_gdl


def save_readable_parsed_gdl(parsed_gdl, filename):
    for relation_name in parsed_gdl['Relations']:
        relation = parsed_gdl['Relations'][relation_name]

        for i in range(len(relation['extend'])):
            predicate_name, para = relation['extend'][i]
            if predicate_name == 'Equation':
                relation['extend'][i] = (predicate_name, str(para))

        for constraint_class in relation['constraints']:
            for i in range(len(relation['constraints'][constraint_class])):
                relation['constraints'][constraint_class][i] = str(relation['constraints'][constraint_class][i])

    for theorem_name in parsed_gdl['Theorems']:
        theorem = parsed_gdl['Theorems'][theorem_name]

        for constraint_class in theorem['ac_check']:
            for i in range(len(theorem['ac_check'][constraint_class])):
                theorem['ac_check'][constraint_class][i] = str(theorem['ac_check'][constraint_class][i])

        for i in range(len(theorem['algebraic_premise'])):
            theorem['algebraic_premise'][i] = str(theorem['algebraic_premise'][i])

        for i in range(len(theorem['algebraic_conclusion'])):
            theorem['algebraic_conclusion'][i] = str(theorem['algebraic_conclusion'][i])

    save_json(parsed_gdl, filename)


def _parse_one_entity(entity, gdl, parsed_gdl):
    entity_name, entity_paras = parse_predicate(entity)
    entity_temporary = []
    for temporary_form in gdl['Entities'][entity]['temporary']:
        _, paras = parse_predicate(temporary_form)
        entity_temporary.append(paras[0])

    parsed_gdl['Entities'][entity_name] = {
        'paras': entity_paras,
        'temporary': entity_temporary
    }


def _parse_one_measure(measure, gdl, parsed_gdl):
    measure_name, measure_paras = parse_predicate(measure)

    if measure_name in parsed_gdl['Measures']:
        e_msg = f"Measure {measure_name} already defined."
        raise Exception(e_msg)

    measure_type = gdl['Measures'][measure]['type']
    if measure_type not in ['parameter', 'attribution']:
        e_msg = (f"The measure type must be one of the following: 'parameter' or 'attribution'. "
                 f"The '{measure_type}' type defined in '{measure_name}' is incorrect.")
        raise Exception(e_msg)

    measure_ee_check = _parse_ee_check(gdl['Measures'][measure]['ee_check'], measure_name, measure_paras)

    measure_sym = gdl['Measures'][measure]['sym']
    if measure_sym in parsed_gdl['sym_to_measure']:
        e_msg = (f"Conflicting definition of symbol '{measure_sym}'. "
                 f"'{parsed_gdl['sym_to_measure'][measure_sym]}' and {measure_name} both use this symbol.")
        raise Exception(e_msg)

    parsed_gdl['Measures'][measure_name] = {
        'type': measure_type,
        'paras': measure_paras,
        'ee_check': measure_ee_check,
        'sym': measure_sym
    }
    parsed_gdl['sym_to_measure'][measure_sym] = measure_name


def _parse_one_relation(relation, gdl, parsed_gdl):
    relation_name, relation_paras = parse_predicate(relation)

    if relation_name in parsed_gdl['Measures']:
        e_msg = f"Relation {relation_name} already defined."
        raise Exception(e_msg)

    relation_type = gdl['Relations'][relation]['type']
    if relation_type not in ['basic', 'composite', 'indirect']:
        e_msg = (f"The relation type must be one of the following: 'basic', 'composite' or 'indirect'. "
                 f"The '{relation_type}' type defined in '{relation_name}' is incorrect.")
        raise Exception(e_msg)

    relation_ee_check = _parse_ee_check(gdl['Relations'][relation]['ee_check'], relation_name, relation_paras)

    relation_extend = []
    for extend in gdl['Relations'][relation]['extend']:
        if not extend.startswith('Eq('):
            extend_name, extend_paras = parse_predicate(extend)
            relation_extend.append([extend_name, extend_paras])
        else:
            relation_extend.append(["Equation", parse_algebra(extend)[1]])

    relation_implicit_entity = {'Point': [], 'Line': [], 'Circle': []}

    relation_constraints = {'Eq': [], 'G': [], 'Geq': [], 'L': [], 'Leq': [], 'Ueq': []}

    letters = list(letters_p)
    for e in relation_paras:
        letters.remove(e)

    constraints = []
    for entity in gdl['Relations'][relation]['implicit_entity']:
        entity, constraint = entity.split(':')
        entity_name, entity_paras = parse_predicate(entity)
        relation_implicit_entity[entity_name].append(entity_paras[0])
        letters.remove(entity_paras[0])
        if len(constraint) > 0:
            constraints += constraint.split("&")
    if len(gdl['Relations'][relation]['constraints']) > 0:
        constraints += gdl['Relations'][relation]['constraints'].split("&")

    while len(constraints) > 0:
        constraint = constraints.pop()
        if (constraint.startswith('Eq(') or constraint.startswith('G(')
                or constraint.startswith('Geq(') or constraint.startswith('L(')
                or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):
            algebra_relation, expr = parse_algebra(constraint)
            relation_constraints[algebra_relation].append(expr)
        else:
            constraint_name, constraint_paras = parse_predicate(constraint)
            if constraint_name not in parsed_gdl['Relations']:
                has_child_constraint = False
                for dependent_relation in gdl['Relations']:
                    if dependent_relation.startswith(constraint_name):
                        _parse_one_relation(dependent_relation, gdl, parsed_gdl)
                        has_child_constraint = True
                if not has_child_constraint:
                    e_msg = f"Unknown child constraint '{constraint_name}' when define '{relation_name}."
                    raise Exception(e_msg)

            replace = dict(zip(parsed_gdl['Relations'][constraint_name]['paras'], constraint_paras))
            for entity_class in parsed_gdl['Relations'][constraint_name]['implicit_entity']:
                for e in parsed_gdl['Relations'][constraint_name]['implicit_entity'][entity_class]:
                    if e in letters:
                        relation_implicit_entity[entity_class].append(e)
                        letters.remove(e)
                    else:
                        replace[e] = letters.pop(0)

            for constraint_class in parsed_gdl['Relations'][constraint_name]['constraints']:
                for dependent_constraint in parsed_gdl['Relations'][constraint_name]['constraints'][constraint_class]:
                    dependent_constraint = replace_expr(dependent_constraint, replace)
                    relation_constraints[constraint_class].append(dependent_constraint)

            if len(set(constraint_paras) - set(relation_paras)) == 0:
                relation_extend.append([constraint_name, constraint_paras])

    parsed_gdl['Relations'][relation_name] = {
        'type': relation_type,
        'paras': relation_paras,
        'ee_check': relation_ee_check,
        'extend': relation_extend,
        'implicit_entity': relation_implicit_entity,
        'constraints': relation_constraints
    }


def _parse_one_theorem(theorem, gdl, parsed_gdl):
    theorem_name, theorem_paras = parse_predicate(theorem)

    if theorem_name in parsed_gdl['Measures']:
        e_msg = f"Theorem {theorem_name} already defined."
        raise Exception(e_msg)

    theorem_type = gdl['Theorems'][theorem]['type']
    if theorem_type not in ['basic', 'extend']:
        e_msg = (f"The theorem type must be one of the following: 'basic' or 'extend'. "
                 f"The '{theorem_type}' type defined in '{theorem_name}' is incorrect.")
        raise Exception(e_msg)

    theorem_ee_check = _parse_ee_check(gdl['Theorems'][theorem]['ee_check'], theorem_name, theorem_paras)

    theorem_ac_check = {'Eq': [], 'G': [], 'Geq': [], 'L': [], 'Leq': [], 'Ueq': []}
    if len(gdl['Theorems'][theorem]['ac_check']) > 0:
        for constraint in gdl['Theorems'][theorem]['ac_check'].split('&'):
            algebra_relation, expr = parse_algebra(constraint)
            theorem_ac_check[algebra_relation].append(expr)

    theorem_geometric_premise = []
    theorem_algebraic_premise = []

    if len(gdl['Theorems'][theorem]['premise']) > 0:
        for premise in gdl['Theorems'][theorem]['premise'].split('&'):
            if premise.startswith('Eq('):
                _, expr = parse_algebra(premise)
                theorem_algebraic_premise.append(expr)
            else:
                premise_name, premise_paras = parse_predicate(premise)
                theorem_geometric_premise.append([premise_name, premise_paras])

    theorem_geometric_conclusion = []
    theorem_algebraic_conclusion = []
    for conclusion in gdl['Theorems'][theorem]['conclusion']:
        if conclusion.startswith('Eq('):
            _, expr = parse_algebra(conclusion)
            theorem_algebraic_conclusion.append(expr)
        else:
            conclusion_name, conclusion_paras = parse_predicate(conclusion)
            theorem_geometric_conclusion.append([conclusion_name, conclusion_paras])

    parsed_gdl['Theorems'][theorem_name] = {
        'type': theorem_type,
        'paras': theorem_paras,
        'ee_check': theorem_ee_check,
        'ac_check': theorem_ac_check,
        'geometric_premise': theorem_geometric_premise,
        'algebraic_premise': theorem_algebraic_premise,
        'geometric_conclusion': theorem_geometric_conclusion,
        'algebraic_conclusion': theorem_algebraic_conclusion
    }


def _parse_ee_check(ee_check, predicate_name, predicate_paras):
    parsed_ee_check_entity = []
    parsed_ee_check_paras = []
    for entity in ee_check:
        entity_name, entity_paras = parse_predicate(entity)
        if entity_name not in ['Point', 'Line', 'Circle']:
            e_msg = (f"Incorrect EE Check definition for '{predicate_name}'. "
                     f"'{entity_name}' is not an valid entity used for EE Check.")
            raise Exception(e_msg)
        parsed_ee_check_entity.append(entity_name)
        parsed_ee_check_paras.append(entity_paras[0])

    if parsed_ee_check_paras != predicate_paras:
        e_msg = (f"EE Check does not match the {predicate_name} paras."
                 f"EE Check paras: {parsed_ee_check_paras}, relation paras: {predicate_paras}")
        raise Exception(e_msg)

    return parsed_ee_check_entity


def parse_predicate(predicate):
    if ' ' in predicate:
        e_msg = f"The format of '{predicate}' is incorrect. Spaces are not allowed in it's definition."
        raise Exception(e_msg)
    name, paras = predicate.split('(')
    paras = paras[:-1].split(',')
    return name, paras


def replace_paras(paras, replace):
    paras = [p if p not in replace else replace[p] for p in paras]
    return paras


def parse_algebra(algebra_constraint):
    if ' ' in algebra_constraint:
        e_msg = f"The format of '{algebra_constraint}' is incorrect. Spaces are not allowed in it's definition."
        raise Exception(e_msg)

    algebra_relation, expr_str = algebra_constraint.split("(", 1)
    expr_str = expr_str[:-1]

    i = 0
    j = 0
    stack = []
    while j < len(expr_str):
        if expr_str[j] == "(":
            stack.append(expr_str[i:j])
            stack.append(expr_str[j])
            i = j + 1
        elif expr_str[j] == ",":
            if i < j:
                stack.append(expr_str[i: j])
                i = j + 1
            else:
                i = i + 1
        elif expr_str[j] == ")":
            if i < j:
                stack.append(expr_str[i: j])
                i = j + 1
            else:
                i = i + 1

            paras = []
            while True:
                p = stack.pop()
                if p == "(":
                    break
                if type(p) is str:
                    if '.' in p:
                        p = symbols(p)
                    else:
                        p = sympify(p)

                paras.append(p)
            paras = paras[::-1]

            operation = stack.pop()

            if operation == 'Add':
                result = paras[0]
                for p in paras[1:]:
                    result += p
            elif operation == 'Sub':
                result = paras[0] - paras[1]
            elif operation == 'Mul':
                result = paras[0]
                for p in paras[1:]:
                    result *= p
            elif operation == 'Div':
                result = paras[0] / paras[1]
            elif operation == 'Pow':
                result = paras[0] ** paras[1]
            elif operation == 'Abs':
                result = sqrt(paras[0] ** 2)
            elif operation == 'DPP':  # DPP(x1,y1,x2,y2)
                result = sqrt((paras[2] - paras[0]) ** 2 + (paras[3] - paras[1]) ** 2)
            elif operation == 'DPL':  # DPL(x,y,k,b)
                result = (paras[2] * paras[0] - paras[1] + paras[3]) / sqrt(paras[2] ** 2 + 1)
            elif operation == 'MA':  # MA(k1,k2)
                result = (atan(paras[0]) - atan(paras[1])) * 180 / pi
            elif operation == 'MAM':  # MAM(k1,k2)
                result = ((atan(paras[0]) - atan(paras[1])) * 180 / pi + 180) % 180
            elif operation == 'PP':  # PP(x,y,cx,cy,r)
                result = (paras[2] - paras[0]) ** 2 + (paras[3] - paras[1]) ** 2 - paras[4] ** 2
            else:
                e_msg = f"Unknown operation: {operation}."
                raise Exception(e_msg)

            stack.append(result)

        j = j + 1

    if len(stack) > 1:
        e_msg = f"Syntax error in algebraic relation '{algebra_constraint}' : missing ')'?"
        raise Exception(e_msg)

    expr = stack.pop()

    return algebra_relation, expr


def replace_expr(expr, replace):
    replace_temp = {}
    syms = list(expr.free_symbols)
    for sym in syms:
        entity, attr = str(sym).split('.')
        sym_temp = symbols("".join([e if e not in replace else e + "'" for e in entity]) + '.' + attr)
        sym_replaced = symbols("".join([e if e not in replace else replace[e] for e in entity]) + '.' + attr)
        if sym_temp not in replace_temp and sym_temp != sym_replaced:
            expr = expr.replace(sym, sym_temp)
            replace_temp[sym_temp] = sym_replaced

    for sym_temp in replace_temp:
        expr = expr.replace(sym_temp, replace_temp[sym_temp])

    return expr


def parse_disjunctive(general_form):  # 过几天在搞
    """
    :param general_form: [general_form]
    :return: norm_form: [[conjunctive_clauses]]
    """
    pass


"""↑----------------Data Parsing-----------------↑"""
"""↓-----------Geometric Configuration-----------↓"""


def show(gc):
    used_operation_ids = set()

    print('\033[33mEntities:\033[0m')
    for entity in ['Point', 'Line', 'Circle']:
        if len(gc.ids_of_predicate[entity]) == 0:
            continue
        print(f'{entity}:')
        for fact_id in gc.ids_of_predicate[entity]:
            used_operation_ids.add(gc.facts[fact_id][4])
            if entity == 'Point':
                values = [(round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1]}.x')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1]}.y')]), 4))]
            elif entity == 'Line':
                values = [(round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1]}.k')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1]}.b')]), 4))]
            else:
                values = [(round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1]}.cx')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1]}.cy')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1]}.r')]), 4))]
            print('{0:<6}{1:<15}{2:<60}{3:<60}{4:<6}{5:<30}'.format(
                fact_id,
                gc.facts[fact_id][1],
                str(gc.facts[fact_id][2]),
                str(gc.facts[fact_id][3]),
                gc.facts[fact_id][4],
                str(values)
            ))
    print()

    print("\033[33mConstructions:\033[0m")
    for operation_id in gc.constructions:
        print('{0:<4}{1:<40}'.format(operation_id, gc.operations[operation_id]))
        target_predicate, target_entity = gc.constructions[operation_id][0]
        print(f'    target entity: {target_predicate}({target_entity})')
        implicit_entities = [f'{p}({i})' for p, i in gc.constructions[operation_id][1]]
        print(f'    implicit entities: {implicit_entities}')
        dependent_entities = [f'{p}({i})' for p, i in gc.constructions[operation_id][2]]
        print(f'    dependent entities: {dependent_entities}')
        print(f"    Eq: {str(gc.constructions[operation_id][3]['Eq']).replace(' ', '').replace(',', ', ')}")
        print(f"    G: {str(gc.constructions[operation_id][3]['G']).replace(' ', '').replace(',', ', ')}")
        print(f"    Geq: {str(gc.constructions[operation_id][3]['Geq']).replace(' ', '').replace(',', ', ')}")
        print(f"    L: {str(gc.constructions[operation_id][3]['L']).replace(' ', '').replace(',', ', ')}")
        print(f"    Leq: {str(gc.constructions[operation_id][3]['Leq']).replace(' ', '').replace(',', ', ')}")
        print(f"    Ueq: {str(gc.constructions[operation_id][3]['Ueq']).replace(' ', '').replace(',', ', ')}")
    print()

    print("\033[33mRelations:\033[0m")
    for predicate in gc.ids_of_predicate:
        if len(gc.ids_of_predicate[predicate]) == 0:
            continue
        if predicate in ['Point', 'Line', 'Circle', 'Equation']:
            continue
        print(f"{predicate}:")
        for fact_id in gc.ids_of_predicate[predicate]:
            used_operation_ids.add(gc.facts[fact_id][4])
            print("{0:<6}{1:<15}{2:<60}{3:<60}{4:<6}".format(
                fact_id,
                ','.join(gc.facts[fact_id][1]),
                str(gc.facts[fact_id][2]).replace(' ', ''),
                str(gc.facts[fact_id][3]).replace(' ', ''),
                gc.facts[fact_id][4]
            ))
    print()

    print("\033[33mEquations:\033[0m")
    for fact_id in gc.ids_of_predicate['Equation']:
        used_operation_ids.add(gc.facts[fact_id][4])
        print("{0:<6}{1:<15}{2:<60}{3:<60}{4:<6}".format(
            fact_id,
            str(gc.facts[fact_id][1]).replace(' ', ''),
            str(gc.facts[fact_id][2]).replace(' ', ''),
            str(gc.facts[fact_id][3]).replace(' ', ''),
            gc.facts[fact_id][4]
        ))
    print()

    print("\033[33mSymbols and Values:\033[0m")
    for sym in gc.value_of_attr_sym:
        equation_id = gc.id['Equation', sym - gc.value_of_attr_sym[sym]]
        predicate = gc.parsed_gdl['sym_to_measure'][str(sym).split('.')[1]]
        instance = ",".join(list(str(sym).split('.')[0]))
        print("{0:<6}{1:<25}{2:<25}{3:<6}".format(
            equation_id,
            f"{predicate}({instance})",
            str(sym),
            str(gc.value_of_attr_sym[sym])
        ))
    print()

    print("\033[33mOperations:\033[0m")
    for i in range(len(gc.operations)):
        if i not in used_operation_ids:
            continue
        print("{0:<6}{1:<50}".format(
            i,
            f'{gc.operations[i]}'
        ))
    print()


def get_reasoning_hypergraph(gc):
    pass


def get_dependent_graph(gc):
    pass


def get_geometric_figure(gc):
    pass


def draw_reasoning_hypergraph(gc, filename):
    pass


def draw_dependent_graph(gc, filename):
    pass


def draw_geometric_figure(gc, filename):
    gc.range = {'x_max': 1, 'x_min': -1, 'y_max': 1, 'y_min': -1}  # 点坐标的范围
    _, ax = plt.subplots()
    ax.axis('equal')  # maintain the circle's aspect ratio
    ax.axis('off')  # hide the axes
    middle_x = (gc.range['x_max'] + gc.range['x_min']) / 2
    range_x = (gc.range['x_max'] - gc.range['x_min']) / 2 * gc.rate
    middle_y = (gc.range['y_max'] + gc.range['y_min']) / 2
    range_y = (gc.range['y_max'] - gc.range['y_min']) / 2 * gc.rate
    # print(gc.range)
    # print(middle_x - range_x, middle_x + range_x)
    # print(middle_y - range_y, middle_y + range_y)
    ax.set_xlim(middle_x - range_x, middle_x + range_x)
    ax.set_ylim(middle_y - range_y, middle_y + range_y)

    for line in gc.instances_of_predicate['Line']:
        k = float(gc.value_of_para_sym[symbols(f'{line}.k')])
        b = float(gc.value_of_para_sym[symbols(f'{line}.b')])
        ax.axline((0, b), slope=k, color='blue')

    for circle in gc.instances_of_predicate['Circle']:
        center_x = float(gc.value_of_para_sym[symbols(f'{circle}.cx')])
        center_y = float(gc.value_of_para_sym[symbols(f'{circle}.cy')])
        r = float(gc.value_of_para_sym[symbols(f'{circle}.r')])
        ax.add_artist(plt.Circle((center_x, center_y), r, color="green", fill=False))

    for point in gc.instances_of_predicate['Point']:
        x = float(gc.value_of_para_sym[symbols(f'{point}.x')])
        y = float(gc.value_of_para_sym[symbols(f'{point}.y')])
        ax.plot(x, y, "o", color='red')
        ax.text(x, y, point, ha='center', va='bottom')

    plt.savefig(filename)


"""↑-----------Geometric Configuration-----------↑"""
"""↓----------Latent Relation Discovery----------↓"""


def find_possible_relations(gc):
    pass


"""↑----------Latent Relation Discovery----------↑"""
