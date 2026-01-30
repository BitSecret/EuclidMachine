from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, debug_execute, draw_gpl


def run(gdl, cdl, outputs_path='./'):
    gc = GeometricConfiguration(gdl, random_seed=cdl['random_seed'])

    for construction in cdl['constructions']:
        debug_execute(gc.construct, [construction])

    gc.draw_gc(save_path=outputs_path, scale=2)

    if len(cdl['goals']) > 0:
        debug_execute(gc.set_goal, [cdl['goals']])

    for theorem in cdl['theorems']:
        operation, theorem = theorem.split(':')
        if operation == 'apply':
            debug_execute(gc.apply, [theorem])
        else:
            debug_execute(gc.decompose, [theorem])

    gc.show_gc()
    gc.draw_sg(save_path=outputs_path)

    return gc


if __name__ == '__main__':
    # run(parse_gdl(load_json('../../../data/gdl/gdl-xiaokai.json')),
    #     load_json('../../../data/cdl/test_incenter.json'))
    # run(parse_gdl(load_json('../../../data/gdl/gdl-xiaokai.json')),
    #     load_json('../../../data/cdl/test_parallel.json'))
    # run(parse_gdl(load_json('../../../data/gdl/gdl-xiaokai.json')),
    #     load_json('../../../data/cdl/test_imo_auto.json'))
    # run(parse_gdl(load_json('../../../data/gdl/gdl-xiaokai.json')),
    #     load_json('../../../data/cdl/test_imo_forward.json'))
    # run(parse_gdl(load_json('../../../data/gdl/gdl-xiaokai.json')),
    #     load_json('../../../data/cdl/test_imo_backward.json'))
    run(parse_gdl(load_json('../../../data/gdl/gdl-yuchang.json')),
        load_json('../../../data/cdl/test_imo_yuchang.json'))
