# loading modules
import firedrake as fd
import ufl
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

sys.path.append(os.environ.get('HIPPYFIRE_BASE_DIR', '../') )
from hippyfire import *
from variables import STATE, PARAMETER, ADJOINT
from PDEProblem import PDEVariationalProblem
from rand import randomGen
from prior import BiLaplacianPrior

# Generate true parameter
def true_model(prior):
    noise = dl.Vector()
    prior.init_vector(noise,"noise")
    parRandom.normal(1., noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    # prior.sample(noise,mtrue)   # sample not defined in any prior's child classes
    return mtrue

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
pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

# Set up prior
gamma = .1
delta = .5

theta0 = 2.
theta1 = .5
alpha  = math.pi / 4
tup = (theta0, theta1, alpha)
anis_diff = fd.Constant(tup)
prior = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True)

# Set up misfit
ntargets = 50
rel_noise = 0.01

#Targets only on the bottom
targets_x = np.random.uniform(0.1,0.9, [ntargets] )
targets_y = np.random.uniform(0.1,0.5, [ntargets] )
targets = np.zeros([ntargets, ndim])
targets[:,0] = targets_x
targets[:,1] = targets_y
