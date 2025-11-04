from em.formalgeo.basic import *
from sympy import symbols
import random

x, y, k, b = symbols('x y k b', real=True)
r = symbols('r', real=True, positive=True)


def parse_sentence(sentence):
    """Parse a sentence to a tree.
    :param sentence: <str>, construction sentence, such as 'Line(a): (A,a), PointOnLine(B,a)'.
    :return: entity: <list>, target entity, such as ['Line', 'a'].
    :return: constraints: <list>, constraints list, such as [['PointOnLine', ['A', 'a']], ['PointOnLine', ['B', 'a']]].
    """
    entity, constraints = sentence.split(": ")
    entity = entity.split("(")
    entity[1] = entity[1].replace(")", "")

    constraints = constraints.split(", ")
    for i in range(len(constraints)):
        constraints[i] = constraints[i].split("(")
        constraints[i][1] = constraints[i][1].replace(")", "")
        constraints[i][1] = constraints[i][1].replace(",", "")
        constraints[i][1] = list(constraints[i][1])

    return entity, constraints


class Figure:
    def __init__(self, seed):
        random.seed(seed)

        self.points = {}
        self.lines = {}
        self.circles = {}

    def add(self, sentence):
        entity, constraints = parse_sentence(sentence)
        if entity[0] == "Point":
            target = Point(x, y, name=entity[1])
            values = solve_constraint(syms=[x, y], constraint=generate_constraint(constraints, self, target),
                                      n=1, range_sample=self.get_point_range(ratio=1.5), max_epoch=1000)
            if len(values) > 0:
                self.add_point(Point(*values[0], name=entity[1]))
                return True
        elif entity[0] == "Line":
            target = Line(k, b, name=entity[1])
            values = solve_constraint(syms=[k, b], constraint=generate_constraint(constraints, self, target))
            if len(values) > 0:
                self.add_line(Line(*values[0], name=entity[1]))
                return True
        else:
            target = Circle(x, y, r, name=entity[1])
            values = solve_constraint(syms=[x, y, r], constraint=generate_constraint(constraints, self, target))
            if len(values) > 0:
                self.add_circle(Circle(*values[0], name=entity[1]))
                return True

        return False

    def add_point(self, point):
        self.points[point.name] = point

    def pop_point(self, point):
        self.points.pop(point.name)

    def add_line(self, line):
        self.lines[line.name] = line

    def pop_line(self, line):
        self.lines.pop(line.name)

    def add_circle(self, circle):
        self.circles[circle.name] = circle

    def pop_circle(self, circle):
        self.circles.pop(circle.name)

    def show(self):
        print("Points:")
        for point in self.points.values():
            print(point)
        print("Lines:")
        for line in self.lines.values():
            print(line)
        print("Circles:")
        for circle in self.circles.values():
            print(circle)
        print()

    def draw(self):
        """Draw figure using matplotlib."""
        _, ax = plt.subplots()
        # plt.gca().set_aspect('equal', adjustable='box')  # maintain the circle's aspect ratio
        ax.axis('equal')
        ax.axis('off')  # hide the axes
        ax.set_xlim(*self.get_point_range(ratio=1.5)[x])
        ax.set_ylim(*self.get_point_range(ratio=1.5)[y])

        for line in self.lines.values():
            ax.axline((0, line.b), slope=line.k, color='blue')

        for circle in self.circles.values():
            ax.add_artist(plt.Circle((circle.center_x, circle.center_y), circle.r, color="green", fill=False))

        for point in self.points.values():
            ax.plot(point.x, point.y, "o", color='red')
            ax.text(point.x - 0.02, point.y, point.name, ha='center', va='bottom')

        plt.show()

    def get_point_range(self, ratio=1.0):
        point_range = {x: [-1, 1], y: [-1, 1]}
        if len(self.points) > 1:
            x_range = [point.x for point in self.points.values()]
            x_m, x_r = (max(x_range) + min(x_range)) / 2, (max(x_range) - min(x_range)) / 2
            point_range[x] = [x_m - ratio * x_r, x_m + ratio * x_r]
            y_range = [point.y for point in self.points.values()]
            y_m, y_r = (max(y_range) + min(y_range)) / 2, (max(y_range) - min(y_range)) / 2
            point_range[y] = [y_m - ratio * y_r, y_m + ratio * y_r]
        return point_range


def constraint_free_point(point):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    return constraint


def constraint_acute_triangle(point_a, point_b, point_c):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    k_a = (point_b.y - point_a.y) / (point_b.x - point_a.x)
    b_a = -k_a * point_a.x + point_a.y
    line_a = Line(k_a, b_a)
    k_b = (point_c.y - point_b.y) / (point_c.x - point_b.x)
    b_b = -k_b * point_b.x + point_b.y
    line_b = Line(k_b, b_b)
    k_c = (point_a.y - point_c.y) / (point_a.x - point_c.x)
    b_c = -k_b * point_c.x + point_c.y
    line_c = Line(k_c, b_c)
    constraint['l'].append((angle_metric(line_a, line_b) + 180) % 180 - 90)
    constraint['l'].append((angle_metric(line_b, line_c) + 180) % 180 - 90)
    constraint['l'].append((angle_metric(line_c, line_a) + 180) % 180 - 90)
    return constraint


def constraint_point_on_line(point, line):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(point_on_line(point, line))
    return constraint


def constraint_point_on_circle(point, circle):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(point_on_circle(point, circle))
    constraint['g'].append(circle.r)
    return constraint


def constraint_equal_angle(line_a, line_b, line_c, line_d):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(angle_metric(line_a, line_b) - angle_metric(line_c, line_d))
    return constraint


def constraint_midpoint(point_m, point_a, point_b):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(point_a.x + point_b.x - 2 * point_m.x)
    constraint['eq'].append(point_a.y + point_b.y - 2 * point_m.y)
    return constraint


def constraint_circumcircle_of_triangle(circle, point_a, point_b, point_c):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(point_on_circle(point_a, circle))
    constraint['eq'].append(point_on_circle(point_b, circle))
    constraint['eq'].append(point_on_circle(point_c, circle))
    constraint['g'].append(circle.r)
    return constraint


def constraint_incircle_of_triangle(circle, point_a, point_b, point_c):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    length_a = distance_metric(point_b, point_c)
    length_b = distance_metric(point_a, point_c)
    length_c = distance_metric(point_a, point_b)
    perimeter = length_a + length_b + length_c
    area = sqrt(perimeter / 2 * (perimeter / 2 - length_a) * (perimeter / 2 - length_b) * (perimeter / 2 - length_c))
    constraint['eq'].append(
        circle.center_x - (length_a * point_a.x + length_b * point_b.x + length_c * point_c.x) / perimeter)
    constraint['eq'].append(
        circle.center_y - (length_a * point_a.y + length_b * point_b.y + length_c * point_c.y) / perimeter)
    constraint['eq'].append(circle.r - 2 * area / perimeter)
    constraint['g'].append(circle.r)
    return constraint


def constraint_between_points(point_m, point_a, point_b):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['g'].append(point_m.x - (point_a.x + point_b.x - sqrt((point_a.x - point_b.x) ** 2)) / 2)
    constraint['l'].append(point_m.x - (point_a.x + point_b.x + sqrt((point_a.x - point_b.x) ** 2)) / 2)
    constraint['g'].append(point_m.y - (point_a.y + point_b.y - sqrt((point_a.y - point_b.y) ** 2)) / 2)
    constraint['l'].append(point_m.y - (point_a.y + point_b.y + sqrt((point_a.y - point_b.y) ** 2)) / 2)
    return constraint


def constraint_distance_greater(point_a, point_b, point_c, point_d):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['g'].append(distance_metric(point_a, point_b) - distance_metric(point_c, point_d))
    return constraint


def constraint_distance_equal(point_a, point_b, point_c, point_d):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(distance_metric(point_a, point_b) - distance_metric(point_c, point_d))
    return constraint


def constraint_is_center_of_circle(point, circle):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(point.x - circle.center_x)
    constraint['eq'].append(point.y - circle.center_y)
    return constraint


def constraint_perpendicular(line_a, line_b):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append(angle_metric(line_a, line_b) - 90)
    return constraint


def constraint_is_diameter_of_circle(point_a, point_b, circle):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['eq'].append((point_a.x + point_b.x) / 2 - circle.center_x)
    constraint['eq'].append((point_a.y + point_b.y) / 2 - circle.center_y)
    constraint['eq'].append(point_on_circle(point_a, circle))
    constraint['g'].append(circle.r)
    return constraint


def constraint_left_side(point, point_a, point_b):
    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    constraint['g'].append(position_metric(point, point_a, point_b))
    return constraint


def generate_constraint(constraints, figure, target):
    algebra_constraints = []

    if type(target) is Point:
        figure.add_point(target)
    elif type(target) is Line:
        figure.add_line(target)
    else:
        figure.add_circle(target)

    for constraint, para in constraints:

        if constraint == "FreePoint":
            algebra_constraints.append(constraint_free_point(
                figure.points[para[0]]))
        elif constraint == "AcuteTriangle":
            algebra_constraints.append(constraint_acute_triangle(
                figure.points[para[0]],
                figure.points[para[1]],
                figure.points[para[2]]))
        elif constraint == "PointOnLine":
            algebra_constraints.append(constraint_point_on_line(
                figure.points[para[0]],
                figure.lines[para[1]]))
        elif constraint == "PointOnCircle":
            algebra_constraints.append(constraint_point_on_circle(
                figure.points[para[0]],
                figure.circles[para[1]]))
        elif constraint == "EqualAngle":
            algebra_constraints.append(constraint_equal_angle(
                figure.lines[para[0]],
                figure.lines[para[1]],
                figure.lines[para[2]],
                figure.lines[para[3]]))
        elif constraint == "Midpoint":
            algebra_constraints.append(constraint_midpoint(
                figure.points[para[0]],
                figure.points[para[1]],
                figure.points[para[2]]))
        elif constraint == "CircumcircleOfTriangle":
            algebra_constraints.append(constraint_circumcircle_of_triangle(
                figure.circles[para[0]],
                figure.points[para[1]],
                figure.points[para[2]],
                figure.points[para[3]]))
        elif constraint == "IncircleOfTriangle":
            algebra_constraints.append(constraint_incircle_of_triangle(
                figure.circles[para[0]],
                figure.points[para[1]],
                figure.points[para[2]],
                figure.points[para[3]]))
        elif constraint == "BetweenPoints":
            algebra_constraints.append(constraint_between_points(
                figure.points[para[0]],
                figure.points[para[1]],
                figure.points[para[2]]))
        elif constraint == "DistanceGreater":
            algebra_constraints.append(constraint_distance_greater(
                figure.points[para[0]],
                figure.points[para[1]],
                figure.points[para[2]],
                figure.points[para[3]]))
        elif constraint == "DistanceEqual":
            algebra_constraints.append(constraint_distance_equal(
                figure.points[para[0]],
                figure.points[para[1]],
                figure.points[para[2]],
                figure.points[para[3]]))
        elif constraint == "IsCenterOfCircle":
            algebra_constraints.append(constraint_is_center_of_circle(
                figure.points[para[0]],
                figure.circles[para[1]]))
        elif constraint == "Perpendicular":
            algebra_constraints.append(constraint_perpendicular(
                figure.lines[para[0]],
                figure.lines[para[1]]))
        elif constraint == "IsDiameterOfCircle":
            algebra_constraints.append(constraint_is_diameter_of_circle(
                figure.points[para[0]],
                figure.points[para[1]],
                figure.circles[para[2]]))
        elif constraint == "LeftSide":
            algebra_constraints.append(constraint_left_side(
                figure.points[para[0]],
                figure.points[para[1]],
                figure.points[para[2]]))
        else:
            raise Exception(f"Unknown constraint: {constraint}.")

    if type(target) is Point:
        figure.pop_point(target)
    elif type(target) is Line:
        figure.pop_line(target)
    else:
        figure.pop_circle(target)

    constraint = {'eq': [], 'g': [], 'geq': [], 'l': [], 'leq': [], 'ueq': []}
    for algebra_constraint in algebra_constraints:
        for key in algebra_constraint:
            for c in algebra_constraint[key]:
                if type(c) in [int, float] or len(c.free_symbols) == 0:
                    continue
                constraint[key].append(c)

    return constraint
