from sympy import symbols, atan, pi, sqrt
import json

"""Useful tools.
1.实体命名空间定义
2.文件输入输出操作
3.数据格式转换
  A、GDL解析为Problem可用的形式
  B、Problem转化为超图
  C、Problem的所有实体作图
"""

latin_upper = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']  # 26
latin_script_upper = ['𝓐', '𝓑', '𝓒', '𝓓', '𝓔', '𝓕', '𝓖', '𝓗', '𝓘', '𝓙', '𝓚', '𝓛', '𝓜', '𝓝', '𝓞', '𝓟', '𝓠', '𝓡',
                      '𝓢', '𝓣', '𝓤', '𝓥', '𝓦', '𝓧', '𝓨', '𝓩']  # 26
latin_lower = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']  # 26
latin_script_lower = ['𝓪', '𝓫', '𝓬', '𝓭', '𝓮', '𝓯', '𝓰', '𝓱', '𝓲', '𝓳', '𝓴', '𝓵', '𝓶', '𝓷', '𝓸', '𝓹', '𝓺', '𝓻',
                      '𝓼', '𝓽', '𝓾', '𝓿', '𝔀', '𝔁', '𝔂', '𝔃']  # 26
greek_upper = ['Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Σ', 'Τ', 'Υ',
               'Φ', 'Χ', 'Ψ', 'Ω']  # 24
greek_italic_upper = ['𝜜', '𝜝', '𝜞', '𝜟', '𝜠', '𝜡', '𝜢', '𝜣', '𝜤', '𝜥', '𝜦', '𝜧', '𝜨', '𝜩', '𝜪', '𝜫', '𝜬', '𝜭', '𝜮',
                      '𝜯', '𝜰', '𝜱', '𝜲', '𝜳', '𝜴']  # 24
greek_lower = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ',
               'φ', 'χ', 'ψ', 'ω']  # 24
greek_italic_lower = ['𝜶', '𝜷', '𝜸', '𝜹', '𝜺', '𝜻', '𝜼', '𝜽', '𝜾', '𝜿', '𝝀', '𝝁', '𝝂', '𝝃', '𝝄', '𝝅', '𝝆', '𝝇', '𝝈',
                      '𝝉', '𝝊', '𝝋', '𝝌', '𝝍', '𝝎']  # 24


# line、circle和point共享命名空间
# GDL定义中，只能使用latin_upper和latin_lower


def load_json(file_path_and_name):
    with open(file_path_and_name, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path_and_name):
    with open(file_path_and_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_gdl(gdl):
    """Parse Geometry Definition Language into a usable format for Problem."""
    # 如果共享命名空间，sym_to_measure似乎用不到了
    parsed_gdl = {
        'Entities': {'Point': {'paras': ['A'], "temporary": []},  # built-in basic entities
                     'Line': {'paras': ['l'], "temporary": ["AB", "A;l"]},
                     'Circle': {'paras': ['O'], "temporary": ["ABC", "O;AB"]}},
        'Measures': {},
        'Relations': {},
        'Theorems': {}
    }

    sym_to_measure = {}
    for measure in gdl['Measures']:
        _parse_one_measure(measure, sym_to_measure, gdl, parsed_gdl)

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

        for i in range(len(theorem['premise'])):
            predicate_name, para = theorem['premise'][i]
            if predicate_name == 'Equation':
                theorem['premise'][i] = (predicate_name, str(para))

        for i in range(len(theorem['conclusion'])):
            predicate_name, para = theorem['conclusion'][i]
            if predicate_name == 'Equation':
                theorem['conclusion'][i] = (predicate_name, str(para))

    save_json(parsed_gdl, filename)


def _parse_one_measure(measure, sym_to_measure, gdl, parsed_gdl):
    measure_name, measure_paras = _parse_predicate(measure)

    measure_type = gdl['Measures'][measure]['type']
    if measure_type not in ['parameter', 'attribution']:
        e_msg = (f"The measure type must be one of the following: 'parameter' or 'attribution'. "
                 f"The '{measure_type}' type defined in '{measure_name}' is incorrect.")
        raise Exception(e_msg)

    measure_ee_check = _parse_ee_check(gdl['Measures'][measure]['ee_check'], measure_name, measure_paras)

    measure_sym = gdl['Measures'][measure]['sym']
    if measure_sym in sym_to_measure:
        e_msg = (f"Conflicting definition of symbol '{measure_sym}'. "
                 f"'{sym_to_measure[measure_sym]}' and {measure_name} both use this symbol.")
        raise Exception(e_msg)

    measure_multi = []
    for multi in gdl['Measures'][measure]['multi']:
        _, multi_paras = _parse_predicate(multi)
        if multi_paras in measure_multi or multi_paras == measure_paras:
            e_msg = f"Duplicate definition of Multiple Forms for '{measure_name}'."
            raise Exception(e_msg)
        measure_multi.append(multi_paras)

    parsed_gdl['Measures'][measure_name] = {
        'type': measure_type,
        'paras': measure_paras,
        'ee_check': measure_ee_check,
        'sym': measure_sym,
        'multi': measure_multi
    }
    sym_to_measure[measure_sym] = measure_name


def _parse_one_relation(relation, gdl, parsed_gdl):
    relation_name, relation_paras = _parse_predicate(relation)

    if relation_name in parsed_gdl["Relations"]:
        return

    relation_type = gdl['Relations'][relation]['type']
    if relation_type not in ['basic', 'composite', 'indirect']:
        e_msg = (f"The relation type must be one of the following: 'basic', 'composite' or 'indirect'. "
                 f"The '{relation_type}' type defined in '{relation_name}' is incorrect.")
        raise Exception(e_msg)

    relation_ee_check = _parse_ee_check(gdl['Relations'][relation]['ee_check'], relation_name, relation_paras)

    relation_extend = []
    for extend in gdl['Relations'][relation]['extend']:
        if not extend.startswith('Eq('):
            extend_name, extend_paras = _parse_predicate(extend)
            relation_extend.append([extend_name, extend_paras])
        else:
            relation_extend.append(["Equation", _parse_algebra(extend)[1]])

    relation_implicit_entity = {'Point': [], 'Line': [], 'Circle': []}

    relation_constraints = {'Eq': [], 'G': [], 'Geq': [], 'L': [], 'Leq': [], 'Ueq': []}

    available_entity = latin_upper + latin_lower
    for e in relation_paras:
        available_entity.remove(e)

    constraints = []
    for entity in gdl['Relations'][relation]['implicit_entity']:
        entity, constraint = entity.split(':')
        entity_name, entity_paras = _parse_predicate(entity)
        relation_implicit_entity[entity_name].append(entity_paras[0])
        available_entity.remove(entity_paras[0])
        if len(constraint) > 0:
            constraints += constraint.split("&")
    if len(gdl['Relations'][relation]['constraints']) > 0:
        constraints += gdl['Relations'][relation]['constraints'].split("&")

    while len(constraints) > 0:
        constraint = constraints.pop()
        if (constraint.startswith('Eq(') or constraint.startswith('G(')
                or constraint.startswith('Geq(') or constraint.startswith('L(')
                or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):
            algebra_relation, expr = _parse_algebra(constraint)
            relation_constraints[algebra_relation].append(expr)
        else:
            constraint_name, constraint_paras = _parse_predicate(constraint)
            if constraint_name not in parsed_gdl['Relations']:
                for dependent_relation in gdl['Relations']:
                    if dependent_relation.startswith(constraint_name):
                        _parse_one_relation(dependent_relation, gdl, parsed_gdl)

            replace = dict(zip(parsed_gdl['Relations'][constraint_name]['paras'], constraint_paras))
            for entity_class in parsed_gdl['Relations'][constraint_name]['implicit_entity']:
                for e in parsed_gdl['Relations'][constraint_name]['implicit_entity'][entity_class]:
                    if e in available_entity:
                        relation_implicit_entity[entity_class].append(e)
                        available_entity.remove(e)
                    else:
                        replace[e] = available_entity.pop(0)

            for dependent_extend_name, dependent_extend_paras in parsed_gdl['Relations'][constraint_name]['extend']:
                if dependent_extend_name == 'Equation':
                    dependent_extend_paras = _replace_expr(dependent_extend_paras, replace)
                else:
                    dependent_extend_paras = _replace_paras(dependent_extend_paras, replace)
                relation_extend.append((dependent_extend_name, dependent_extend_paras))

            for constraint_class in parsed_gdl['Relations'][constraint_name]['constraints']:
                for dependent_constraint in parsed_gdl['Relations'][constraint_name]['constraints'][constraint_class]:
                    dependent_constraint = _replace_expr(dependent_constraint, replace)
                    relation_constraints[constraint_class].append(dependent_constraint)

    parsed_gdl['Relations'][relation_name] = {
        'type': relation_type,
        'paras': relation_paras,
        'ee_check': relation_ee_check,
        'extend': relation_extend,
        'implicit_entity': relation_implicit_entity,
        'constraints': relation_constraints
    }


def _parse_one_theorem(theorem, gdl, parsed_gdl):
    theorem_name, theorem_paras = _parse_predicate(theorem)

    theorem_type = gdl['Theorems'][theorem]['type']
    if theorem_type not in ['basic', 'extend']:
        e_msg = (f"The theorem type must be one of the following: 'basic' or 'extend'. "
                 f"The '{theorem_type}' type defined in '{theorem_name}' is incorrect.")
        raise Exception(e_msg)

    theorem_ee_check = _parse_ee_check(gdl['Theorems'][theorem]['ee_check'], theorem_name, theorem_paras)

    theorem_ac_check = {'Eq': [], 'G': [], 'Geq': [], 'L': [], 'Leq': [], 'Ueq': []}
    if len(gdl['Theorems'][theorem]['ac_check']) > 0:
        for constraint in gdl['Theorems'][theorem]['ac_check'].split('&'):
            algebra_relation, expr = _parse_algebra(constraint)
            theorem_ac_check[algebra_relation].append(expr)

    theorem_premise = []

    if len(gdl['Theorems'][theorem]['premise']) > 0:
        for premise in gdl['Theorems'][theorem]['premise'].split('&'):
            if premise.startswith('Eq('):
                premise_name = 'Equation'
                _, premise_paras = _parse_algebra(premise)
            else:
                premise_name, premise_paras = _parse_predicate(premise)
            theorem_premise.append([premise_name, premise_paras])

    theorem_conclusion = []
    for conclusion in gdl['Theorems'][theorem]['conclusion']:
        if conclusion.startswith('Eq('):
            conclusion_name = 'Equation'
            _, conclusion_paras = _parse_algebra(conclusion)
        else:
            conclusion_name, conclusion_paras = _parse_predicate(conclusion)
        theorem_conclusion.append([conclusion_name, conclusion_paras])

    parsed_gdl['Theorems'][theorem_name] = {
        'type': theorem_type,
        'paras': theorem_paras,
        'ee_check': theorem_ee_check,
        'ac_check': theorem_ac_check,
        'premise': theorem_premise,
        'conclusion': theorem_conclusion
    }


def _parse_ee_check(ee_check, predicate_name, predicate_paras):
    parsed_ee_check = {'Point': [], 'Line': [], 'Circle': []}
    if len(ee_check) != len(predicate_paras):
        e_msg = (f"Incorrect EE Check definition for '{predicate_name}'. "
                 f"There are '{len(predicate_paras)}' parameters but '{len(ee_check)}' EE Check are defined.")
        raise Exception(e_msg)
    for entity in ee_check:
        entity_name, entity_paras = _parse_predicate(entity)
        if entity_name not in parsed_ee_check:
            e_msg = (f"Incorrect EE Check definition for '{predicate_name}'. "
                     f"'{entity_name}' is not an valid entity used for EE Check.")
            raise Exception(e_msg)
        if entity_paras[0] not in predicate_paras:
            e_msg = (f"Incorrect EE Check definition for '{predicate_name}'. "
                     f"'{entity_paras[0]}' is not in para '{predicate_paras}'.")
            raise Exception(e_msg)
        if entity_paras[0] in parsed_ee_check[entity_name]:
            e_msg = f"Duplicate definition of EE Check for '{predicate_name}'."
            raise Exception(e_msg)
        parsed_ee_check[entity_name].append(entity_paras[0])

    return parsed_ee_check


def _parse_predicate(predicate):
    name, paras = predicate.split('(')
    paras = paras[:-1].split(',')
    return name, paras


def _replace_paras(paras, replace):
    paras = [p if p not in replace else replace[p] for p in paras]
    return paras


def _parse_algebra(algebra_constraint):
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
                    p = symbols(p)
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
            elif operation == 'DPP':  # DPP(x1,y1,x2,y2)
                result = sqrt((paras[2] - paras[0]) ** 2 + (paras[3] - paras[1]) ** 2)
            elif operation == 'DPL':  # DPL(x,y,k,b)
                result = (paras[2] * paras[0] - paras[1] + paras[3]) / sqrt(paras[2] ** 2 + 1)
            elif operation == 'MA':  # MA(k1,k2)
                result = (atan(paras[0]) - atan(paras[1])) * 180 / pi
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


def _replace_expr(expr, replace):
    replace_temp = {}
    syms = list(expr.free_symbols)
    for sym in syms:
        if str(sym)[0] in replace:
            expr = expr.replace(sym, symbols(str(sym)[0] + "'" + str(sym)[1:]))
            replace_temp[str(sym)[0] + "'"] = replace[str(sym)[0]]

    syms = list(expr.free_symbols)
    for sym in syms:
        if str(sym)[0:2] in replace_temp:
            expr = expr.replace(sym, symbols(replace_temp[str(sym)[0:2]] + str(sym)[2:]))

    return expr


if __name__ == '__main__':
    save_readable_parsed_gdl(parse_gdl(load_json('../../../data/gdl/gdl.json')),
                             '../../../data/gdl/parsed_gdl.json')

    # print(_parse_predicate("PointOnLine(A,l)"))
    # print(_parse_predicate("PointOnLine(A,l)", {'l': 'k'}))
    # print()
    #
    # print(_parse_algebra("Eq(Sub(A.x,B.x))"))
    # print(_parse_algebra("Eq(Sub(MA(a.k,b.k),MA(l.k,k.k)))"))
    # print(_parse_algebra("Eq(Sub(DPP(A.x,A.y,B.x,B.y),DPP(C.x,C.y,D.x,D.y)))"))
    # print(_parse_algebra("Eq(Sub(DPP(A.x,A.y,B.x,B.y),DPP(C.x,C.y,D.x,D.y)))", {'C': 'X', 'D': 'A', 'A': 'D'}))
    # print()
    #
    # print(_parse_logical("PointOnCircle(M,O)&EqualDistance(M,A,M,B)&PointLeftSegment(M,B,A)"))
    # print(_parse_logical("PointOnCircle(M,O)"))
    # print()

    # _, a = _parse_algebra("Eq(Sub(DPP(A.x,A.y,B.x,B.y),DPP(C.x,C.y,D.x,D.y)))")
    # print(a)
    # print(_replace_expr(a, {'B': 'A', 'A': 'X'}))

