import os

from em.formalgeo.construction import *
from em.formalgeo.tools import *


def solve_problem(pid):
    problem = load_json(f"../../../data/imo sl/{pid}.json")

    figure = Figure(seed=problem['seed'])
    for sentence in problem['construction_cdl']:
        added = figure.add(sentence)
        print(f"{sentence}, success={added}")
    print()

    figure.show()
    figure.draw()


if __name__ == '__main__':
    solve_problem(pid=2)
