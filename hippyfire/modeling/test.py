import sys
sys.path.append(r"/home/karanh97/work/hippyfire/hippyfire/modeling")
sys.path.append(r"/home/karanh97/work/hippyfire/hippyfire/algorithms")
sys.path.append(r"/home/karanh97/work/hippyfire/hippyfire/utils")
# sys.path.append(r"/home/karanh97/work/hippyfire")
# from hippyfire import *
import firedrake as fd
import ufl
import numpy as np
from PDEProblem import PDEVariationalProblem
from variables import STATE, PARAMETER, ADJOINT

ndim = 2
nx = 64
ny = 64
mesh = fd.UnitSquareMesh(nx, ny)
Vh2 = fd.FunctionSpace(mesh, 'Lagrange', 2)
Vh1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
Vh = [Vh2, Vh1, Vh2]
print("DOFS: STATE = {0}, PARAMETER = {1}, ADJOINT = {2}"
      .format(Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()) )

x = fd.SpatialCoordinate(mesh)
u = 1 + x[0] * x[0] + 2 * x[1] * x[1]
u0 = fd.Constant(0.0)
bc0 = fd.DirichletBC(Vh[STATE], u0, "on_boundary")
bc = fd.DirichletBC(Vh[STATE], u, "on_boundary")
print(type(bc))
f = fd.Constant(0.0)

def pde_varf(u, m ,p):
    return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx

pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
