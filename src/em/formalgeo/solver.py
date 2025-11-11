from em.formalgeo.configuration import Configuration
from em.formalgeo.tools import load_json, parse_gdl, save_readable_parsed_gdl


def test_problem():
    problem = Configuration(parse_gdl(load_json('test_gdl.json')))
    problem.construct('Point(A):FreePoint(A)')
    problem.construct('Point(B):FreePoint(B)')
    problem.construct('Point(C):PointLeftSegment(C,A,B)')
    problem.construct('Line(a):PointOnLine(B,a)&PointOnLine(C,a)')
    problem.construct('Line(b):PointOnLine(A,b)&PointOnLine(C,b)')
    problem.construct('Line(c):PointOnLine(A,c)&PointOnLine(B,c)')
    problem.construct('Line(x):EqualAngle(b,x,x,c)')
    problem.construct('Line(y):EqualAngle(c,x,x,a)')
    problem.construct('Point(O):PointOnLine(O,x)&PointOnLine(O,y)')
    problem.construct('Line(z):PointOnLine(C,z)&PointOnLine(O,z)')
    problem.construct('Point(X):PointOnLine(X,a)&PointOnLine(X,x)')
    problem.construct('Point(Y):PointOnLine(Y,b)&PointOnLine(Y,y)')
    problem.construct('Point(Z):PointOnLine(Z,c)&PointOnLine(Z,z)')
    problem.apply("angle_bisector_determination_angle_equal")
    problem.apply("angle_bisector_property_distance_equal")
    problem.apply("angle_bisector_determination_distance_equal")


if __name__ == '__main__':
    save_readable_parsed_gdl(parse_gdl(load_json('test_gdl.json')), '../../../data/gdl/parsed_gdl.json')
