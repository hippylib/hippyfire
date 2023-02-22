# import ufl
import firedrake as fd
import math
import numpy as np
import logging
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../") )
# from hippylib import nb
#
# define mesh
n = 16
degree = 1
mesh = fd.UnitSquareMesh(n, n)

Vh = fd.FunctionSpace(mesh, 'Lagrange', degree)
print( "dim(Vh) = ", Vh.dim() )

# plot function for mesh
fd.triplot(mesh)


# defining boundaries
u_L = fd.Constant(0.)
u_R = fd.Constant(0.)

x = fd.SpatialCoordinate(Vh.mesh())
sigma_top = fd.Constant(0.)
sigma_bottom = - (fd.pi / 2.0) * fd.sin(2 * fd.pi * x[0])

dsl = fd.Measure("ds", mesh, 3)
dsu = fd.Measure("ds", mesh, 4)

bcs = [fd.DirichletBC(Vh, u_L, 1), fd.DirichletBC(Vh, u_R, 2)]

# f = fd.Function(Vh)
f = (4.0 * fd.pi* fd.pi + fd.pi * fd.pi / 4.0) * (fd.sin( 2 * fd.pi * x[0] ) * fd.sin(( fd.pi / 2.0) * x[1]))

# defining the test and trial functions
u = fd.TrialFunction(Vh)
v = fd.TestFunction(Vh)
a = fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
L = f * v * fd.dx + sigma_top * v * dsu + sigma_bottom * v * dsl

uh = fd.Function(Vh)

fd.solve(a == L, uh, bcs=bcs, solver_parameters = {'ksp_type':'cg'})

fig, axes = plt.subplots()
colors = fd.tripcolor(uh, axes = axes)
fig.colorbar(colors)
plt.show()
