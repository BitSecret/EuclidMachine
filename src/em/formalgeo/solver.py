from em.formalgeo.problem import Problem
from em.formalgeo.tools import load_json, save_json, parse_gdl, save_readable_parsed_gdl


save_readable_parsed_gdl(
    parse_gdl(load_json('../../../data/gdl/gdl.json')),
    '../../../data/gdl/parsed_gdl.json')


# problem = Problem(parse_gdl(load_json('../../../data/gdl/gdl.json')))
# problem.construct("Point(A)")
