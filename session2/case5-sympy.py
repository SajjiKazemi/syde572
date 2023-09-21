import numpy as np
import sympy as sp
# Define the probability density function (PDF)
def pdf(x1, x2):
    # Check if the point (x1, x2) is within the shaded region
    if (x1 >= 1 and x1 <= 3) and (x2 >= 1 and x2 <= 4 - x1):
        return 0.5
    else:
        return 0

def pdf1(x1):
    if x1>=1 and x1<=3:
        return 1.5-0.5*x1
    else:
        return 0
    
def pdf2(x2):
    if x2>=1 and x2<=3:
        return 1.5-0.5*x2
    else:
        return 0

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
px1 = sp.integrate(0.5, (x2, 1, 4-x1))
print(px1)
px2 = sp.integrate(0.5, (x1, 1, 4-x2))
print(px2)

ev_x1 = sp.integrate(x1*px1, (x1, 1, 3))
print(ev_x1)
ev_x2 = sp.integrate(x2*px2, (x2, 1, 3))
print(ev_x2)
ev_x1x2 = sp.integrate(x1*x2*0.5, (x1, 1, 4-x2), (x2, 1, 3))
print(ev_x1x2)
ev_x1pow2 = sp.integrate(x1**2*px1, (x1, 1, 3))
print(ev_x1pow2)
