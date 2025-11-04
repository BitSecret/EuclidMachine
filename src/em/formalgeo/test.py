from sympy import *
import string

# print(cos(0))
# print(cos(pi / 4))
# print(cos(pi / 2))
# print(cos(pi / 4 * 3))
# print(cos(pi))
#
# print(acos(1))
# print(acos(sqrt(2) / 2))
# print(acos(0))
# print(acos(-sqrt(2) / 2))
# pri

x1, x2, x3 = symbols('x1000000000000000 x2 x3')
print(x1)
print(x2)
print(x3)
f1 = x1 + x2 + x3
print(f1)
print(f1.subs({x1: x2, x2: x3}))
print(f1)
