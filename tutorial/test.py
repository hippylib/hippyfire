# loading modules
import sys
import os
import math
import firedrake as fd
import ufl
import numpy as np
import matplotlib.pyplot as plt
import nb

sys.path.append(os.environ.get('HIPPYFIRE_BASE_DIR', '../') )
from hippyfire import *
from variables import STATE, PARAMETER, ADJOINT
from PDEProblem import PDEVariationalProblem
from rand import randomGen
from prior import BiLaplacianPrior
from misfit import ContinuousStateObservation
from linalg import matVecMult
from model import Model
from modelVerify import modelVerify


# Set up mesh and finite element spaces
ndim = 2
nx = 64
ny = 64
mesh = fd.UnitSquareMesh(nx, ny)
Vh2 = fd.FunctionSpace(mesh, 'Lagrange', 2)
Vh1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
Vh = [Vh2, Vh1, Vh2]
print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
    Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()) )

# generate true parameter
mtrue = randomGen(Vh[STATE])

# Set up forward problem
u_bdr = fd.SpatialCoordinate(mesh)[1]
u_bdr0 = fd.Constant(0.0)

bc = fd.DirichletBC(Vh[STATE], u_bdr, [3, 4]) # [3, 4] indicates that bc is applied to y == 0 amd y ==1
bc0 = fd.DirichletBC(Vh[STATE], u_bdr0, [3, 4])

f = fd.Constant(0.0)


def pde_varf(u, m, p):
    return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx

pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

# Set up prior
gamma = .1
delta = .5

theta0 = 2.
theta1 = .5
alpha = math.pi / 4
# tup = (theta0, theta1, alpha)
# anis_diff = fd.Constant(tup)
pr = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, robin_bc=True)
mtrue = (randomGen(Vh[PARAMETER])).vector()
objs = [fd.Function(Vh[PARAMETER], mtrue), fd.Function(Vh[PARAMETER], pr.mean)]
# plt.plot(objs)

# Set up misfit
ntargets = 50
rel_noise = 0.01

#Targets only on the bottom
# targets_x = np.random.uniform(0.1,0.9, [ntargets] )
# targets_y = np.random.uniform(0.1,0.5, [ntargets] )
# targets = np.zeros([ntargets, ndim])
# targets[:,0] = targets_x
# targets[:,1] = targets_y
# print( "Number of observation points: {0}".format(ntargets) )

misfit = ContinuousStateObservation(Vh[STATE], ufl.dx, bcs=[bc0])
utrue = pde.generate_state()
x = [utrue, mtrue, None]
pde.solveFwd(x[STATE], x)

# print(misfit.W.M.handle.size)
misfit.d = matVecMult(misfit.W, x[STATE], misfit.d)
# print(misfit.d.get_local())
vmax = max( utrue.get_local().max(), misfit.d.get_local().max() )
vmin = min( utrue.get_local().min(), misfit.d.get_local().min() )
print(vmax, vmin)
# plt.figure(figsize=(15,5))
# nb.plot(fd.Function(Vh[STATE], utrue), mytitle="True State", subplot_loc=121, vmin=vmin, vmax=vmax)
# nb.plot_pts(targets, misfit.d, mytitle="Observations", subplot_loc=122, vmin=vmin, vmax=vmax)
# plt.show()
model = Model(pde, pr, misfit)
x = fd.SpatialCoordinate(mesh)
m0 = fd.interpolate(fd.sin(x[0]), Vh[PARAMETER])
_ = modelVerify(model, m0.vector())
