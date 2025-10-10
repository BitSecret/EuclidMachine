from sympy import nonlinsolve, sqrt, atan, pi
import matplotlib
import matplotlib.pyplot as plt
import random

matplotlib.use('TkAgg')  # 解决后端兼容性问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

greek = ['Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
         'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']
english_upper = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class Point:
    def __init__(self, x, y, name='default'):
        self.name = name
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point {self.name}: [{self.x}, {self.y}]"


class Line:
    def __init__(self, k, b, name='default'):
        """y = k * x + b"""
        self.name = name
        self.k = k
        self.b = b

    def __str__(self):
        return f"Line {self.name}: y=[{self.k}]x+[{self.b}]"


class Circle:
    def __init__(self, center_x, center_y, r, name='default'):
        """(center_x - x) ** 2 + (center_y - y) ** 2 = r ** 2"""
        self.name = name
        self.center_x = center_x
        self.center_y = center_y
        self.r = r

    def __str__(self):
        return f"Circle {self.name}: (x-[{self.center_x}])**2+(x-[{self.center_y}])**2=[{self.r}]**2"


def distance_metric(point_a, point_b):
    """Calculate the distance between two points."""
    return sqrt((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2)


def angle_metric(line_a, line_b):
    """Calculate the angle between two lines based on their slopes.
    if result > 0, it is the angle of clockwise rotation from line_a to line_b.
    if result = 0, line_a and line_b are parallel.
    if result < 0, it is the angle of anticlockwise rotation from line_a to line_b.
    """
    return (atan(line_a.k) - atan(line_b.k)) * 180 / pi


def position_metric(point, point_a, point_b):
    """Determine the relative position between a point and a line based on the cross product.
    if result > 0, the point is on the left side of line AB.
    if result = 0, the point is on the line AB.
    if result < 0, the point is on the right side of line AB.
    """
    return (point_b.x - point_a.x) * (point.y - point_a.y) - (point_b.y - point_a.y) * (point.x - point_a.x)


def point_on_line(point, line):
    return point.y - line.k * point.x - line.b


def point_on_circle(target_point, circle):
    return (circle.center_x - target_point.x) ** 2 + (circle.center_y - target_point.y) ** 2 - circle.r ** 2


def solve_constraint(syms, constraint, n=1, range_sample=None, max_epoch=1000):
    """Find points that satisfy the algebraic constraints.
    :param syms: <list>, symbols, such as [x, y], [k, b] or [x, y, r].
    :param constraint: <list>, constraint, such as {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}.
    :param n: <int>, number of samples, such as 1.
    :param range_sample: <dict>, range of samples (only point need), such as {x: [-2, 2], y: [-3, 3]}.
    :param max_epoch: <int>, maximum number of attempts to generate values, such as 1000.
    :return: values: <list>, values that satisfy the algebraic constraints, such as [[1, 0.5], [1.5, 0.5]].
    """
    values = []  # list of values, such as [[1, 0.5], [1.5, 0.5]]
    constraint_values = []  # list of constraint values, contains symbols, such as [[y, y - 1], [x, 0.5]]

    if constraint is None:
        return values

    def satisfy_inequalities(value):
        sym_to_value = dict(zip(syms, value))
        for g in constraint['g']:
            if g.subs(sym_to_value).evalf(chop=True) <= 0:
                return False
        for geq in constraint['geq']:
            if geq.subs(sym_to_value).evalf(chop=True) < 0:
                return False
        for l in constraint['l']:
            if l.subs(sym_to_value).evalf(chop=True) >= 0:
                return False
        for leq in constraint['leq']:
            if leq.subs(sym_to_value).evalf(chop=True) > 0:
                return False
        for ueq in constraint['ueq']:
            if ueq.subs(sym_to_value).evalf(chop=True) == 0:
                return False
        return True

    def has_free_symbols(value):
        for v in value:
            if len(v.free_symbols) > 0:
                return True
        return False

    if len(constraint['eq']) == 0:  # free point
        constraint_values.append(syms)
    else:  # constraint value
        for solved_value in list(nonlinsolve(constraint['eq'], syms)):
            if not has_free_symbols(solved_value):
                if satisfy_inequalities(solved_value):
                    values.append([float(s_v) for s_v in list(solved_value)])
            else:
                constraint_values.append(list(solved_value))

    if len(constraint_values) == 0 or range_sample is None:  # no further solvable values
        return values

    epoch = 0  # randomly generate points that satisfy the constraint
    while len(values) < n and epoch < max_epoch:
        random_sample = {sym: random.uniform(range_sample[sym][0], range_sample[sym][1]) for sym in syms}
        random_value = [float(cv.subs(random_sample)) for cv in constraint_values[epoch % len(constraint_values)]]

        if satisfy_inequalities(random_value):
            values.append(random_value)
        epoch += 1
    return values
