from em.formalgeo.configuration import GeometricConfiguration
from em.formalgeo.tools import load_json, parse_gdl, save_readable_parsed_gdl
from em.formalgeo.tools import show, draw_geometric_figure
import copy


def test1():
    gc = GeometricConfiguration(parse_gdl(load_json('../../../data/test_gdl.json')))
    gc.construct('Point(A):FreePoint(A)')
    gc.construct('Point(B):FreePoint(B)')
    gc.construct('Point(C):PointLeftSegment(C,A,B)')
    gc.construct('Line(a):PointOnLine(B,a)&PointOnLine(C,a)')
    gc.construct('Line(b):PointOnLine(A,b)&PointOnLine(C,b)')
    gc.construct('Line(c):PointOnLine(A,c)&PointOnLine(B,c)')
    gc.construct('Line(x):PointOnLine(A,x)&EqualAngle(b,x,x,c)')
    gc.construct('Line(y):PointOnLine(B,y)&EqualAngle(c,y,y,a)')
    gc.construct('Point(O):PointOnLine(O,x)&PointOnLine(O,y)')
    gc.construct('Line(z):PointOnLine(C,z)&PointOnLine(O,z)')
    gc.construct('Point(X):PointOnLine(X,a)&PointOnLine(X,x)')
    gc.construct('Point(Y):PointOnLine(Y,b)&PointOnLine(Y,y)')
    gc.construct('Point(Z):PointOnLine(Z,c)&PointOnLine(Z,z)')
    gc.construct('Circle(O):PointOnCircle(A,O)&PointOnCircle(B,O)&PointOnCircle(C,O)')
    gc.apply("angle_bisector_determination_angle_equal")
    gc.apply("angle_bisector_property_distance_equal")
    gc.apply("angle_bisector_determination_distance_equal")
    show(gc)
    draw_geometric_figure(gc, '../../../data/outputs/geometric_figure.png')


def test2():
    gc = GeometricConfiguration(parse_gdl(load_json('../../../data/mygdl.json')))
    gc.construct('Point(D):FreePoint(D)')
    gc.construct('Point(B):FreePoint(B)')
    gc.construct('Point(C):PointLeftSegment(C,D,B)')
    gc.construct('Line(s):PointOnLine(C,s)&PointOnLine(B,s)')
    gc.construct('Line(l):PointOnLine(D,l)&PointOnLine(B,l)')
    gc.construct('Line(m):PointOnLine(C,m)&LinesParallel(l,m)')
    gc.construct('Line(i):PointOnLine(D,i)&PointOnLine(C,i)')
    gc.construct('Line(j):PointOnLine(B,j)&LinesParallel(i,j)')
    gc.construct('Point(E):PointOnLine(E,j)&PointOnLine(E,m)')
    gc.apply('D40')
    gc.apply('D47')
    show(gc)
    draw_geometric_figure(gc, '../../../data/outputs/geometric_figure.png')


if __name__ == '__main__':
    # save_readable_parsed_gdl(
    #     parsed_gdl=parse_gdl(load_json('../../../data/test_gdl.json')),
    #     filename='../../../data/outputs/parsed_gdl.json'
    # )
    test1()

