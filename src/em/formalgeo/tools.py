from sympy import sympify, symbols, atan, pi, log
import json
import matplotlib
import matplotlib.pyplot as plt
import re

matplotlib.use('TkAgg')  # Resolve backend compatibility issues
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Use Microsoft YaHei font
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issues

"""↓------Vocabulary------↓"""

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

entity_letters = tuple(  # available entity letters
    list(_lu) + list(_lsu) + list(_gu) + list(_giu) + list(_ll) + list(_lsl) + list(_gl) + list(_gil)
)
expr_letters = (  # letter in expr
    '+', '-', '**', '*', '/', 'sqrt', 'atan', 'Mod',
    'nums', 'pi',
    '(', ')'
)
delimiter_letters = (  # letter distinguish between different part of fact and operation
    '&', '|', '~', ':',
    '<p>',  # premise
    '<cons>',  # construction,
    '<t>',  # theorem
    '<c>',  # conclusion
)
theorem_letters = (  # preset theorem name
    'multiple_forms', 'auto_extend', 'solve_eq', 'same_entity_extend'
)


def get_vocab(parsed_gdl):
    """Vocab for deep learning model."""
    vocab = list(entity_letters)
    vocab += list(expr_letters)
    vocab += sorted(['.' + parsed_gdl['Measures'][measure]['sym'] for measure in parsed_gdl['Measures']])
    vocab += list(delimiter_letters)
    vocab += list(theorem_letters)
    vocab += sorted(list(parsed_gdl['Theorems']))
    return vocab


"""↑---------------Vocabulary--------------↑"""
"""↓--------------Useful Tools-------------↓"""


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


"""↑------------Useful Tools------------↑"""
"""↓-------------algebraic--------------↓"""

precision = 20
chop = 1e-10


def _satisfy_eq(expr, sym_to_value):
    return expr.subs(sym_to_value).evalf(n=precision, chop=chop) == 0


def _satisfy_g(expr, sym_to_value):
    return expr.subs(sym_to_value).evalf(n=precision, chop=chop) > 0


def _satisfy_geq(expr, sym_to_value):
    return expr.subs(sym_to_value).evalf(n=precision, chop=chop) >= 0


def _satisfy_l(expr, sym_to_value):
    return expr.subs(sym_to_value).evalf(n=precision, chop=chop) < 0


def _satisfy_leq(expr, sym_to_value):
    return expr.subs(sym_to_value).evalf(n=precision, chop=chop) <= 0


def _satisfy_ueq(expr, sym_to_value):
    return expr.subs(sym_to_value).evalf(n=precision, chop=chop) != 0


satisfy_algebra = {"Eq": _satisfy_eq, "G": _satisfy_g, "Geq": _satisfy_geq,
                   "L": _satisfy_l, "Leq": _satisfy_leq, "Ueq": _satisfy_ueq}


def satisfy_inequalities(inequalities, sym_to_value):
    for algebraic_relation, expr in inequalities:
        if not satisfy_algebra[algebraic_relation](expr, sym_to_value):
            return False
    return True


"""↑-------------algebraic--------------↑"""
"""↓------------Data Parsing------------↓"""


def parse_gdl(gdl):
    """Parse Geometry Definition Language into a usable format for Problem."""
    parsed_gdl = {
        'Entities': {},
        'Measures': {},
        'sym_to_measure': {},
        'Relations': {},
        'Theorems': {}
    }

    _add_entities(parsed_gdl)

    # parse Measures
    for measure in gdl['Measures']:
        try:
            _parse_one_measure(measure, gdl, parsed_gdl)
        except Exception as e:
            e_msg = f"An error occurred while parsing measure '{measure}'."
            raise Exception(e_msg) from e

    # circularly parse Relations until all Relation is parsed
    while True:
        update = False
        for relation in gdl['Relations']:
            try:
                update = _parse_one_relation(relation, gdl, parsed_gdl) or update
            except Exception as e:
                e_msg = f"An error occurred while parsing relation '{relation}'."
                raise Exception(e_msg) from e
        if not update:
            break

    # parse Theorems
    for theorem in gdl['Theorems']:
        try:
            _parse_one_theorem(theorem, gdl, parsed_gdl)
        except Exception as e:
            e_msg = f"An error occurred while parsing theorem '{theorem}'."
            raise Exception(e_msg) from e

    return parsed_gdl


def _add_entities(parsed_gdl):
    """Preset Entities"""
    parsed_gdl['Entities']['Point'] = {
        'type': 'entity',
        "paras": ("A",),
        "temporary": ()
    }
    parsed_gdl['Entities']['Line'] = {
        'type': 'entity',
        'paras': ('l',),
        'temporary': ('AB', 'A;k')
    }
    parsed_gdl['Entities']['Circle'] = {
        'type': 'entity',
        'paras': ('O',),
        'temporary': ('ABC', 'O;AB')
    }


def _parse_one_measure(measure, gdl, parsed_gdl):
    measure_name, measure_paras = parse_fact(measure)

    if measure_name in parsed_gdl['Measures']:
        e_msg = f"Measure {measure_name} already defined."
        raise Exception(e_msg)

    measure_type = gdl['Measures'][measure]['type']
    if measure_type not in {'parameter', 'attribution'}:
        e_msg = "The measure type must be one of the following: 'parameter' or 'attribution'."
        raise Exception(e_msg)

    measure_ee_checks = _parse_ee_check(gdl['Measures'][measure]['ee_checks'], measure_paras)

    measure_sym = gdl['Measures'][measure]['sym']
    if measure_sym in parsed_gdl['sym_to_measure']:
        e_msg = (f"Conflicting definition of symbol '{measure_sym}'. "
                 f"'{parsed_gdl['sym_to_measure'][measure_sym]}' and {measure_name} both use this symbol.")
        raise Exception(e_msg)

    parsed_gdl['Measures'][measure_name] = {
        'type': measure_type,
        'paras': tuple(measure_paras),
        'ee_checks': measure_ee_checks,
        'sym': measure_sym
    }
    parsed_gdl['sym_to_measure'][measure_sym] = measure_name


def _parse_one_relation(relation, gdl, parsed_gdl):
    if not _can_parse_relation(relation, gdl, parsed_gdl):
        return False

    relation_name, relation_paras = parse_fact(relation)

    relation_type = gdl['Relations'][relation]['type']
    if relation_type not in {'basic', 'composite', 'indirect'}:
        e_msg = f"The relation type must be one of the following: 'basic', 'composite' or 'indirect'. "
        raise Exception(e_msg)

    relation_ee_checks = _parse_ee_check(gdl['Relations'][relation]['ee_checks'], relation_paras)

    relation_extends = []
    for extend in gdl['Relations'][relation]['extends']:
        if extend.startswith('Eq('):
            relation_extends.append(("Equation", parse_algebra(extend)[1]))
        else:
            extend_name, extend_paras = parse_fact(extend)
            relation_extends.append((extend_name, tuple(extend_paras)))

    relation_multiple_forms = []
    for multiple_form in gdl['Relations'][relation]['multiple_forms']:
        multiple_form_name, multiple_form_paras = parse_fact(multiple_form)
        if multiple_form_name != relation_name or len(multiple_form_paras) != len(relation_paras):
            e_msg = f"Error in the definition of Multiple forms. It is inconsistent with the relation name."
            raise Exception(e_msg)
        multiple_form = tuple([relation_paras.index(entity) for entity in multiple_form_paras])
        if multiple_form in relation_multiple_forms:
            e_msg = f"Error in the definition of Multiple forms. Redundant definition."
            raise Exception(e_msg)
        relation_multiple_forms.append(multiple_form)

    letters = list(entity_letters)
    for e in relation_paras:
        letters.remove(e)

    relation_implicit_entities = {'Point': [], 'Line': [], 'Circle': []}
    relation_constraints = []
    for construction in gdl['Relations'][relation]['implicit_entities']:  # add implicit constraint
        entity, constraints = construction.split(':')
        entity_name, entity_paras = parse_fact(entity)
        relation_implicit_entities[entity_name].append(entity_paras[0])
        letters.remove(entity_paras[0])
        relation_constraints.extend(parse_disjunctive(constraints))

    relation_constraints.extend(parse_disjunctive(gdl['Relations'][relation]['constraints']))  # add constraint

    parsed_relation_constraints = []
    for constraint in relation_constraints:
        # direct constraint
        if (constraint.startswith('Eq(') or constraint.startswith('G(')
                or constraint.startswith('Geq(') or constraint.startswith('L(')
                or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):
            algebra_relation, expr = parse_algebra(constraint)
            parsed_relation_constraints.append((algebra_relation, expr))
        else:  # composite constraint
            constraint_name, constraint_paras = parse_fact(constraint)
            # map para defined in GDL to constraint_paras
            replace = dict(zip(parsed_gdl['Relations'][constraint_name]['paras'], constraint_paras))

            # merge implicit entities
            for entity_class in {'Point', 'Line', 'Circle'}:
                for entity in parsed_gdl['Relations'][constraint_name]['implicit_entities'][entity_class]:
                    if entity in letters:  # entity not used
                        relation_implicit_entities[entity_class].append(entity)
                        letters.remove(entity)
                        replace[entity] = entity
                    else:  # entity has been used, replace new one
                        replace_entity = letters.pop(0)
                        relation_implicit_entities[entity_class].append(replace_entity)
                        replace[entity] = replace_entity

            # merge dependent relation constraints
            for algebra_relation, expr in parsed_gdl['Relations'][constraint_name]['constraints']:
                expr = replace_expr(expr, replace)
                parsed_relation_constraints.append((algebra_relation, expr))

            # add dependent relation extends
            if len(set(constraint_paras) - set(relation_paras)) == 0:
                relation_extends.append((constraint_name, tuple(constraint_paras)))

    relation_implicit_entities['Point'] = tuple(relation_implicit_entities['Point'])
    relation_implicit_entities['Line'] = tuple(relation_implicit_entities['Line'])
    relation_implicit_entities['Circle'] = tuple(relation_implicit_entities['Circle'])

    parsed_gdl['Relations'][relation_name] = {
        'type': relation_type,
        'paras': tuple(relation_paras),
        'ee_checks': relation_ee_checks,
        'multiple_forms': tuple(relation_multiple_forms),
        'extends': tuple(relation_extends),
        'implicit_entities': relation_implicit_entities,
        'constraints': tuple(parsed_relation_constraints)
    }

    return True


def _can_parse_relation(relation, gdl, parsed_gdl):
    relation_name, _ = parse_fact(relation)

    if relation_name in parsed_gdl['Relations']:
        return False

    for construction in gdl['Relations'][relation]['implicit_entities']:  # dependent relations
        constraints = construction.split(':')[1]
        for constraint in parse_disjunctive(constraints):
            if (constraint.startswith('Eq(') or constraint.startswith('G(')
                    or constraint.startswith('Geq(') or constraint.startswith('L(')
                    or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):
                continue
            constraint_name, _ = parse_fact(constraint)
            if constraint_name not in parsed_gdl['Relations']:
                return False

    for constraint in parse_disjunctive(gdl['Relations'][relation]['constraints']):  # dependent relations
        if (constraint.startswith('Eq(') or constraint.startswith('G(')
                or constraint.startswith('Geq(') or constraint.startswith('L(')
                or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):
            continue
        constraint_name, _ = parse_fact(constraint)
        if constraint_name not in parsed_gdl['Relations']:
            return False

    return True


def _parse_one_theorem(theorem, gdl, parsed_gdl):
    theorem_name, theorem_paras = parse_fact(theorem)

    if theorem_name in parsed_gdl['Theorems']:
        e_msg = f"Theorem {theorem_name} already defined."
        raise Exception(e_msg)

    theorem_type = gdl['Theorems'][theorem]['type']
    if theorem_type not in {'basic', 'extend'}:
        e_msg = "The theorem type must be one of the following: 'basic' or 'extend'. "
        raise Exception(e_msg)

    theorem_ee_checks = _parse_ee_check(gdl['Theorems'][theorem]['ee_checks'], theorem_paras)  # [entity_name]

    theorem_ac_checks = []  # (relation_type, expr, paras)
    if len(gdl['Theorems'][theorem]['ac_checks']) > 0:
        for constraint in gdl['Theorems'][theorem]['ac_checks'].split('&'):
            algebra_relation, expr = parse_algebra(constraint)
            paras = [str(sym).split('.')[0] for sym in expr.free_symbols]
            theorem_ac_checks.append((algebra_relation, expr, paras))

    theorem_geometric_premises = []  # (predicate, paras)
    theorem_algebraic_premises = []  # (expr, paras)
    for premise in parse_disjunctive(gdl['Theorems'][theorem]['premises']):
        if premise.startswith('Eq('):
            _, expr = parse_algebra(premise)
            paras = []
            for sym in expr.free_symbols:
                paras.extend(list(str(sym).split('.')[0]))
            theorem_algebraic_premises.append((expr, paras))
        else:
            premise_name, premise_paras = parse_fact(premise)
            theorem_geometric_premises.append((premise_name, tuple(premise_paras)))

    # adjust the execution order
    products = []
    added_paras = set()

    # map para to geometric_premises
    paras_to_geometric_premises = {}
    for premise_name, premise_paras in theorem_geometric_premises:
        for p in list(set(premise_paras)):
            if p not in paras_to_geometric_premises:
                paras_to_geometric_premises[p] = [(premise_name, premise_paras)]
            else:
                paras_to_geometric_premises[p].append((premise_name, premise_paras))

    # those paras not exist in all geometric_premises, only exist in ee checks
    for i in range(len(theorem_paras)):
        if theorem_paras[i] not in paras_to_geometric_premises:
            products.append((theorem_ee_checks[i], (theorem_paras[i],)))
            added_paras.add(theorem_paras[i])

    # add geometric_premise to product, entity p only exist in those geometric_premise
    for p in paras_to_geometric_premises:
        if len(paras_to_geometric_premises[p]) == 1 and paras_to_geometric_premises[p][0] not in products:
            products.append(paras_to_geometric_premises[p][0])
            theorem_geometric_premises.remove(paras_to_geometric_premises[p][0])
            added_paras.update(paras_to_geometric_premises[p][0][1])

    # for the remaining geometric_premise, select a portion to add to product, according to:
    # 1. the number of not added entities in it paras
    # 2. the number of paras
    while len(added_paras) < len(theorem_paras):
        max_index = 0
        max_not_added_paras_len = len(set(theorem_geometric_premises[0][1]) - added_paras)
        max_paras_len = len(theorem_geometric_premises[0][1])

        for i in range(1, len(theorem_geometric_premises)):
            not_added_paras_len = len(set(theorem_geometric_premises[i][1]) - added_paras)
            paras_len = len(theorem_geometric_premises[i][1])

            if not_added_paras_len > max_not_added_paras_len or (
                    not_added_paras_len == max_not_added_paras_len and paras_len > max_paras_len):
                max_index = i
                max_not_added_paras_len = not_added_paras_len
                max_paras_len = paras_len

        products.append(theorem_geometric_premises[max_index])
        added_paras.update(theorem_geometric_premises[max_index][1])
        theorem_geometric_premises.pop(max_index)

    # sort product according to the number of its paras
    products.sort(key=len, reverse=True)

    gpl = []

    predicate = products[0][0]  # the first item
    paras = products[0][1]
    added_paras = list(paras)

    # fact internal constraints. SameAngle(a,l,l,b) has constraint position 1 = position 2.
    same_index = []
    for i in range(len(paras)):
        for j in range(i + 1, len(paras)):
            if paras[i] == paras[j]:
                same_index.append((i, j))

    ac_checks = _get_ac_checks(theorem_ac_checks, added_paras)  # (relation_type, expr)
    geometric_premises = _get_geometric_premises(theorem_geometric_premises, added_paras)  # (predicate, paras)
    algebraic_premises = _get_algebraic_premises(theorem_algebraic_premises, added_paras)  # (expr)

    gpl.append({
        "product": (predicate, paras, tuple(same_index)),
        "ac_checks": ac_checks,
        "geometric_premises": geometric_premises,
        "algebraic_premises": algebraic_premises
    })

    for predicate, paras in products[1:]:
        same_index = []
        for i in range(len(added_paras)):
            for j in range(len(paras)):
                if added_paras[i] == paras[j]:
                    same_index.append((i, j))
        added_index = []
        for j in range(len(paras)):
            if paras[j] not in added_paras:
                added_index.append(j)
                added_paras.append(paras[j])

        ac_checks = _get_ac_checks(theorem_ac_checks, added_paras)  # (relation_type, expr)

        geometric_premises = _get_geometric_premises(theorem_geometric_premises, added_paras)  # (predicate, paras)

        algebraic_premises = _get_algebraic_premises(theorem_algebraic_premises, added_paras)  # (expr)

        gpl.append({
            "product": (predicate, paras, tuple(same_index), tuple(added_index)),
            "ac_checks": ac_checks,
            "geometric_premises": geometric_premises,
            "algebraic_premises": algebraic_premises
        })

    if (len(theorem_ac_checks) + len(theorem_algebraic_premises) + len(theorem_geometric_premises)) != 0:
        e_msg = f"There exist unadded constraints."
        raise Exception(e_msg)

    # parse theorem conclusions
    theorem_conclusions = []
    for conclusion in gdl['Theorems'][theorem]['conclusions']:
        if conclusion.startswith('Eq('):
            _, expr = parse_algebra(conclusion)
            theorem_conclusions.append(('Equation', expr))
        else:
            conclusion_name, conclusion_paras = parse_fact(conclusion)
            theorem_conclusions.append((conclusion_name, tuple(conclusion_paras)))

    parsed_gdl['Theorems'][theorem_name] = {
        'type': theorem_type,
        'paras': tuple(theorem_paras),
        'gpl': tuple(gpl),
        'conclusions': tuple(theorem_conclusions)
    }


def _get_ac_checks(theorem_ac_checks, added_paras):
    ac_checks = []  # (relation_type, expr, paras)
    for i in range(len(theorem_ac_checks))[::-1]:
        ac_check_type, ac_check_expr, ac_check_paras = theorem_ac_checks[i]
        if len(set(ac_check_paras) - set(added_paras)) == 0:
            ac_checks.append(theorem_ac_checks[i])
            theorem_ac_checks.pop(i)
    # sort according to the number of paras
    ac_checks = sorted(ac_checks, key=lambda x: (len(x[2]), len(set(x[2]))), reverse=True)
    ac_checks = tuple([(relation_type, expr) for relation_type, expr, _ in ac_checks])
    return ac_checks


def _get_geometric_premises(theorem_geometric_premises, added_paras):
    geometric_premises = []  # (predicate, paras)
    for i in range(len(theorem_geometric_premises))[::-1]:
        geometric_premises_predicate, geometric_premises_paras = theorem_geometric_premises[i]
        if len(set(geometric_premises_paras) - set(added_paras)) == 0:
            geometric_premises.append(theorem_geometric_premises[i])
            theorem_geometric_premises.pop(i)
    # sort according to the number of paras
    geometric_premises = tuple(sorted(geometric_premises, key=lambda x: (len(x[1]), len(set(x[1]))), reverse=True))
    return geometric_premises


def _get_algebraic_premises(theorem_algebraic_premises, added_paras):
    algebraic_premises = []  # (expr, paras)
    for i in range(len(theorem_algebraic_premises))[::-1]:
        algebraic_premises_expr, algebraic_premises_paras = theorem_algebraic_premises[i]
        if len(set(algebraic_premises_paras) - set(added_paras)) == 0:
            algebraic_premises.append(theorem_algebraic_premises[i])
            theorem_algebraic_premises.pop(i)
    algebraic_premises = sorted(algebraic_premises, key=lambda x: (len(x[1]), len(set(x[1]))), reverse=True)
    algebraic_premises = tuple([expr for expr, _ in algebraic_premises])
    return algebraic_premises


def _parse_ee_check(ee_check, paras):
    parsed_ee_check_entity = []
    parsed_ee_check_paras = []
    for entity in ee_check:
        entity_name, entity_paras = parse_fact(entity)
        if entity_name not in {'Point', 'Line', 'Circle'}:
            e_msg = f"'{entity_name}' is not an valid entity used for EE Check."
            raise Exception(e_msg)
        parsed_ee_check_entity.append(entity_name)
        parsed_ee_check_paras.append(entity_paras[0])

    if parsed_ee_check_paras != paras:
        e_msg = "EE Check does not match with the paras."
        raise Exception(e_msg)

    return tuple(parsed_ee_check_entity)


def parse_fact(fact):
    """Parse fact to logic form.

    Args:
        fact (str): Entity, relation, measure or theorem. Such as: 'Point(A)', 'D6(a,b,c)', 'PointOnLine(A,l)',
        'DistanceBetweenPointAndPoint(A,B)'.

    Returns:
        parsed_predicate (tuple): Predicate name and instance list. Such as: ('Point', ['A']), ('D6', ['a', 'b', 'c']),
        ('PointOnLine', ['A', 'l']), ('DistanceBetweenPointAndPoint', ['A', 'B']).
    """
    if not bool(re.match(r'^\w+\(\S(?:,\s*\S)*\)$', fact)):
        e_msg = f"The format of '{fact}' is incorrect."
        raise Exception(e_msg)

    predicate, paras = fact.split('(')
    paras = paras[:-1].split(',')

    for entity in paras:
        if entity not in entity_letters:
            e_msg = f"Character '{entity}' is not in em.formalgeo.tools.letters."
            raise Exception(e_msg)

    return predicate, paras


def replace_paras(paras, replace):
    """Replace instances according to the replacement mapping.

    Args:
        paras (list): List of entity. Such as ['A', 'B', 'C'].
        replace (dict): Keys are the old entity and values are the new entity. Such As {'A': 'B', 'B': 'C', 'C': 'D'}.
    Returns:
        replaced_instances: Replaced instances. Such as ['B', 'C', 'D'].
    """
    return [replace[entity] for entity in paras]


def parse_algebra(algebra_constraint):
    """
    Parse an algebra constraint to logic form.

    Args:
        algebra_constraint (str): Algebra relation and expression. The components include algebra relation types,
        algebraic operations, the symbolic representations of measures and constants. Such as:
        'Eq(Sub(A.y,Add(Mul(l.k,A.x),l.b)))', 'G(Sub(Mul(Sub(C.x,B.x),Sub(A.y,B.y)),Mul(Sub(C.y,B.y),Sub(A.x,B.x))))'.

    Returns:
        parsed_algebra_constraint (tuple): Algebra relation type and instance of sympy expression. Such as:
        (('Eq', -A.x*l.k + A.y - l.b)), ('G', -(A.x - B.x)*(-B.y + C.y) + (A.y - B.y)*(-B.x + C.x)).
    """
    if ' ' in algebra_constraint:
        e_msg = f"The format of '{algebra_constraint}' is incorrect. Spaces are not allowed in it's definition."
        raise Exception(e_msg)

    algebra_relation, expr_str = algebra_constraint.split("(", 1)
    expr_str = expr_str[:-1]

    if '(' not in expr_str:  # such as 'Eq(lk.ma)'
        entities = list(expr_str.split('.')[0])
        for entity in entities:
            if entity not in entity_letters:
                e_msg = (f"Character '{entity}' of expr '{algebra_constraint}' "
                         f"is not in em.formalgeo.tools.letters.")
                raise Exception(e_msg)
        return algebra_relation, symbols(expr_str, real=True)

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
                para = stack.pop()
                if para == "(":
                    break
                if type(para) is str:
                    if '.' in para:
                        entities = list(para.split('.')[0])
                        for entity in entities:
                            if entity not in entity_letters:
                                e_msg = (f"Character '{entity}' of expr '{algebra_constraint}' "
                                         f"is not in em.formalgeo.tools.letters.")
                                raise Exception(e_msg)
                        para = symbols(para, real=True)  # symbol representation of measure
                    else:
                        para = sympify(para)  # constant
                paras.append(para)
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
            elif operation == 'ABS':
                result = paras[0] ** 2
            elif operation == 'DPP':  # DPP(x1,y1,x2,y2)
                result = (paras[2] - paras[0]) ** 2 + (paras[3] - paras[1]) ** 2
            elif operation == 'DPL':  # DPL(x,y,k,b)
                result = (paras[2] * paras[0] - paras[1] + paras[3]) ** 2 / (paras[2] ** 2 + 1)
            elif operation == 'MA':  # MA(k1,k2)
                result = (atan(paras[0]) - atan(paras[1])) * 180 / pi
            elif operation == 'MAM':  # MAM(k1,k2)
                result = ((atan(paras[0]) - atan(paras[1])) * 180 / pi + 180) % 180
            elif operation == 'PP':  # PP(x,y,cx,cy,r)
                result = (paras[2] - paras[0]) ** 2 + (paras[3] - paras[1]) ** 2 - paras[4] ** 2
            elif operation == 'Log':  # Log(x)
                result = log(paras[0])
            else:
                e_msg = f"Unknown operation '{operation}' in algebra constraint '{algebra_constraint}'."
                raise Exception(e_msg)

            stack.append(result)

        j = j + 1

    if len(stack) > 1:
        e_msg = f"Syntax error in algebra constraint '{algebra_constraint}': missing ')'?"
        raise Exception(e_msg)

    return algebra_relation, stack.pop()


def replace_expr(expr, replace):
    """Replace instances according to the replacement mapping.

    Args:
        expr (sympy_expr): instance of sympy expression. Such as -A.x*l.k + A.y - l.b.
        replace (dict): Keys are the old entity and values are the new entity. Such As {'A': 'B', 'l': 'k'}.

    Returns:
        replaced_expr: Replaced expr. Such as -B.x*k.k + B.y - k.b.
    """
    replace_old_to_temp = {}
    replace_temp_to_new = {}
    for sym_old in expr.free_symbols:
        entities_old, attr = str(sym_old).split('.')

        entities_temp = [e + "'" for e in entities_old]
        sym_temp = symbols("".join(entities_temp) + '.' + attr)

        entities_new = [replace[e] for e in entities_old]
        if attr == 'dpp':  # Only dpp has multiple forms (AB.dpp is BA.dpp), so we hard-coded it directly.
            entities_new = sorted(entities_new)
        sym_new = symbols("".join(entities_new) + '.' + attr)

        replace_old_to_temp[sym_old] = sym_temp
        replace_temp_to_new[sym_temp] = sym_new

    expr = expr.subs(replace_old_to_temp).subs(replace_temp_to_new)

    return expr


def parse_disjunctive(general_form):  # 过几天在搞
    """
    :param general_form: [general_form]
    :return: norm_form: [[conjunctive_clauses]]
    """
    if len(general_form) == 0:
        return []
    return general_form.split('&')


"""↑----------------Data Parsing-----------------↑"""
"""↓-----------Geometric Configuration-----------↓"""


def show_gc(gc, target=None):
    used_operation_ids = set()
    used_premise_ids = set()
    if target is not None:
        pass

    print('\033[33mEntities:\033[0m')
    for entity in ['Point', 'Line', 'Circle']:
        if len(gc.ids_of_predicate[entity]) == 0:
            continue
        print(f'{entity}:')
        for fact_id in gc.ids_of_predicate[entity]:
            used_operation_ids.add(gc.facts[fact_id][4])
            if entity == 'Point':
                values = [(round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1][0]}.x')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1][0]}.y')]), 4))]
            elif entity == 'Line':
                values = [(round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1][0]}.k')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1][0]}.b')]), 4))]
            else:
                values = [(round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1][0]}.cx')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1][0]}.cy')]), 4),
                           round(float(gc.value_of_para_sym[symbols(f'{gc.facts[fact_id][1][0]}.r')]), 4))]
            print('{0:<6}{1:<15}{2:<60}{3:<60}{4:<6}{5:<30}'.format(
                fact_id,
                gc.facts[fact_id][1][0],
                str(gc.facts[fact_id][2]).replace(' ', ''),
                str(gc.facts[fact_id][3]).replace(' ', ''),
                gc.facts[fact_id][4],
                str(values).replace(' ', '')
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
        print(f"    constraints: {str(gc.constructions[operation_id][3]).replace(' ', '')}")
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


def draw_gc(gc, filename):
    _, ax = plt.subplots()
    ax.axis('equal')  # maintain the circle's aspect ratio
    ax.axis('off')  # hide the axes
    middle_x = (gc.range['x_max'] + gc.range['x_min']) / 2
    range_x = (gc.range['x_max'] - gc.range['x_min']) / 2 * gc.rate
    middle_y = (gc.range['y_max'] + gc.range['y_min']) / 2
    range_y = (gc.range['y_max'] - gc.range['y_min']) / 2 * gc.rate
    ax.set_xlim(float(middle_x - range_x), float(middle_x + range_x))
    ax.set_ylim(float(middle_y - range_y), float(middle_y + range_y))

    for line in gc.instances_of_predicate['Line']:
        k = float(gc.value_of_para_sym[symbols(f'{line[0]}.k')])
        b = float(gc.value_of_para_sym[symbols(f'{line[0]}.b')])
        ax.axline((0, b), slope=k, color='blue')

    for circle in gc.instances_of_predicate['Circle']:
        u = float(gc.value_of_para_sym[symbols(f'{circle[0]}.u')])
        v = float(gc.value_of_para_sym[symbols(f'{circle[0]}.v')])
        r = float(gc.value_of_para_sym[symbols(f'{circle[0]}.r')])
        ax.add_artist(plt.Circle((u, v), r, color="green", fill=False))

    for point in gc.instances_of_predicate['Point']:
        x = float(gc.value_of_para_sym[symbols(f'{point[0]}.x')])
        y = float(gc.value_of_para_sym[symbols(f'{point[0]}.y')])
        ax.plot(x, y, "o", color='red')
        ax.text(x, y, point[0], ha='center', va='bottom')

    plt.savefig(filename)


def split_expr(expr, letters):
    parsed_expr = []

    expr = str(expr).replace(' ', '')  # remove ' '

    for matched in re.findall(r'\d+\.*\d*', expr):  # replace number with 'nums'
        expr = expr.replace(matched, 'nums', 1)

    i = 0
    while i < len(expr):  # tokenize
        added = False
        for matched_part in letters:  # expr letters
            if expr[i:].startswith(matched_part):
                parsed_expr.append(matched_part)
                i = i + len(matched_part)
                added = True
                break
        if not added:  # entity letters
            parsed_expr.append(expr[i])
            i = i + 1

    return parsed_expr


def split_construction(construction, letters):
    parsed_construction = []

    entity, constraints = construction.split(':')

    predicate, paras = parse_fact(entity)  # parse target entity
    parsed_construction.append(predicate)
    parsed_construction.extend(paras)
    parsed_construction.append(':')

    for constraint in parse_disjunctive(constraints):
        if parsed_construction[-1] != ':':
            parsed_construction.append('&')

        if (constraint.startswith('Eq(') or constraint.startswith('G(')  # parse algebraic constraint
                or constraint.startswith('Geq(') or constraint.startswith('L(')
                or constraint.startswith('Leq(') or constraint.startswith('Ueq(')):
            algebra_relation, expr = parse_algebra(constraint)
            expr = split_expr(expr, letters)
            parsed_construction.append(algebra_relation)
            parsed_construction.extend(expr)
        else:  # parse predefined constraint
            predicate, paras = parse_fact(constraint)
            parsed_construction.append(predicate)
            parsed_construction.extend(paras)

    return parsed_construction


def get_hypergraph(gc, serialize=False):
    """
    Return hypergraph (dict format).
    If serialize=True, return serialize '<premise> <theorem> <conclusion>' (list format).
    """
    hypergraph = {
        'notes': [],  # node i, node_id is same with fact_id
        'dependent_entities': [],  # dependent entities of node i
        'edges': [],  # edge i, node_id is not same with operation_id
        'hypergraph': []  # (head_node_ids,), edge_id, (tail_node_ids,))
    }
    syms = ['.' + gc.parsed_gdl['Measures'][measure]['sym'] for measure in gc.parsed_gdl['Measures']]
    letters = tuple(list(expr_letters) + syms)

    added_edges = []
    for fact_id in range(len(gc.facts)):
        predicate, instance, premise_ids, entity_ids, operation_id = gc.facts[fact_id]

        if predicate != 'Equation':
            node = tuple([predicate] + list(instance))
        else:
            node = tuple([predicate] + split_expr(instance, letters))

        hypergraph['notes'].append(node)  # add node
        hypergraph['dependent_entities'].append(entity_ids)  # add dependent_entity

        if operation_id in added_edges:
            continue
        added_edges.append(operation_id)

        edge_id = len(hypergraph['edges'])
        edge = gc.operations[operation_id]
        if ':' in edge:  # construction
            edge = tuple(split_construction(edge, letters))
        elif edge in theorem_letters:  # theorem with no paras
            edge = [edge]
        else:  # theorem with paras
            edge_predicate, edge_instance = parse_fact(edge)
            edge = tuple([edge_predicate] + edge_instance)

        hypergraph['edges'].append(edge)  # add edge
        hypergraph['hypergraph'].append((premise_ids, edge_id, tuple(sorted(list(set(gc.groups[operation_id]))))))

    if serialize:
        serialized_hypergraph = []
        for head_node_ids, edge_id, tail_node_ids in hypergraph['hypergraph']:
            one_step = ['<p>']

            for head_node_id in head_node_ids:  # add premise
                if one_step[-1] != '<p>':
                    one_step.append('&')
                one_step.extend(hypergraph['notes'][head_node_id])

            if ':' in hypergraph['edges'][edge_id]:  # add operation
                one_step.append('<cons>')
            else:
                one_step.append('<t>')
            one_step.extend(hypergraph['edges'][edge_id])

            one_step.append('<c>')  # add conclusions
            for tail_node_id in tail_node_ids:
                if one_step[-1] != '<c>':
                    one_step.append('|')
                one_step.extend(hypergraph['notes'][tail_node_id])

            serialized_hypergraph.append(one_step)

        return serialized_hypergraph

    return hypergraph


def draw_hypergraph(gc, filename):
    """实体依赖关系与reasoning关系用不同颜色的线表示"""
    pass


def find_possible_relations(gc):
    pass


"""↑-----------Geometric Configuration-----------↑"""
