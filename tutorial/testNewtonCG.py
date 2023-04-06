# loading modules
import sys
import os
import math
import firedrake as fd
import ufl
import numpy as np
import matplotlib.pyplot as plt
# import nb

sys.path.insert(0, os.environ.get('HIPPYFIRE_BASE_DIR'))

from modeling.variables import STATE, PARAMETER, ADJOINT
from modeling.PDEProblem import PDEVariationalProblem
from modeling.prior import BiLaplacianPrior
from modeling.misfit import ContinuousStateObservation
from modeling.model import Model
from modeling.modelVerify import modelVerify

from utils.rand import randomGen

from algorithms.linalg import matVecMult

from algorithms.cgsolverSteihaug import CGSolverSteihaug
from algorithms.NewtonCG import ReducedSpaceNewtonCG
from modeling.reducedHessian import ReducedHessian


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

f = fd.Constant(1.0)


def pde_varf(u, m, p):
    return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx

pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

# Set up prior
gamma = .1
delta = .5

theta0 = 2.
theta1 = .5
alpha = math.pi / 4
# tup = (theta0, theta1, alpha)
# anis_diff = fd.Constant(tup)
pr = BiLaplacianPrior(Vh[PARAMETER], gamma, delta, robin_bc=True)
x = fd.SpatialCoordinate(mesh)
mtrue = fd.interpolate(fd.sin(x[0])*fd.cos(x[1]), Vh[PARAMETER]).vector()
m0 = fd.interpolate(fd.sin(x[0]), Vh[PARAMETER]).vector()
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

misfit = ContinuousStateObservation(Vh[STATE], ufl.dx, bcs=bc0)
misfit.noise_variance = 1e-4
utrue = pde.generate_state()
x = [utrue, mtrue, None]
pde.solveFwd(x[STATE], x)

# print(misfit.W.M.handle.size)
misfit.d.axpy(1., utrue)
#misfit.d.axpy(np.sqrt(misfit.noise_variance), randomGen(Vh[STATE]).vector())
# print(misfit.d.get_local())
vmax = max( utrue.get_local().max(), misfit.d.get_local().max() )
vmin = min( utrue.get_local().min(), misfit.d.get_local().min() )
print(vmax, vmin)
# plt.figure(figsize=(15,5))
# nb.plot(fd.Function(Vh[STATE], utrue), mytitle="True State", subplot_loc=121, vmin=vmin, vmax=vmax)
# nb.plot_pts(targets, misfit.d, mytitle="Observations", subplot_loc=122, vmin=vmin, vmax=vmax)
# plt.show()
model = Model(pde, pr, misfit)

# print("Test only misfit")
# eps, err_grad, err_H = modelVerify(model, m0, misfit_only=True)

# print(err_grad)
# print(err_H)

# print("Test also prior")
# eps, err_grad, err_H = modelVerify(model, m0, misfit_only=False)

# print(err_grad)
# print(err_H)


# verifying NewtonCG
# m = pr.mean.copy()
# z = [None, m, None]
# z[STATE] = model.generate_vector(STATE)
# z[ADJOINT] = model.generate_vector(ADJOINT)
# model.solveFwd(z[STATE], z)
# mhat = model.generate_vector(PARAMETER)
# mg = model.generate_vector(PARAMETER)
# z_star = [None, None, None] + z[3::]
# z_star[STATE] = model.generate_vector(STATE)
# z_star[PARAMETER] = model.generate_vector(PARAMETER)
# cost_old, _, _ = model.cost(z)

# it = 0
# max_iter = 20
# rel_tol = 1e-6
# abs_tol = 1e-12
# GN_iter = 5
# cg_coarse_tolerance = .5
# cg_max_iter = 100
# coverged = False

# model.solveAdj(z[ADJOINT], z)
# model.setPointForHessianEvaluations(z, gauss_newton_approx=True)
# gradnorm = model.evalGradientParameter(z, mg)

# gradnorm_ini = gradnorm
# tol = max(abs_tol, gradnorm_ini * rel_tol)
# tolcg = min(cg_coarse_tolerance, math.sqrt(gradnorm/gradnorm_ini))
# HessApply = ReducedHessian(model)
# solver = CGSolverSteihaug(model.prior.R.getFunctionSpace())
# solver.set_operator(HessApply)
# solver.set_preconditioner(model.Rsolver())
# solver.solve(mhat, (-1. * mg))
# mg_what = mg.inner(mhat)

# alpha = 1.0
# descent = 0
# n_backtrack = 0

# z_star[PARAMETER].assign(0.0)
# z_star[PARAMETER].axpy(1., z[PARAMETER])
# z_star[PARAMETER].axpy(alpha, mhat)
# z_star[STATE].assign(0.0)
# z_star[STATE].axpy(1., z[STATE])
# model.solveFwd(z_star[STATE], z_star)

# cost_new, reg_new, misfit_new = model.cost(z_star)

m = pr.mean.copy()
solver = ReducedSpaceNewtonCG(model)
solver.parameters["rel_tolerance"] = 1e-6
solver.parameters["abs_tolerance"] = 1e-12
solver.parameters["max_iter"]      = 25
solver.parameters["GN_iter"] = 5
solver.parameters["globalization"] = "LS"
solver.parameters["LS"]["c_armijo"] = 1e-4

x = solver.solve([None, m, None])

if solver.converged:
    print( "\nConverged in ", solver.it, " iterations.")
else:
    print( "\nNot Converged")

print( "Termination reason: ", solver.termination_reasons[solver.reason] )
print( "Final gradient norm: ", solver.final_grad_norm )
print( "Final cost: ", solver.final_cost )

# printing graphs of State and Parameter

# plt.figure(figsize=(15, 5))


fig, ax = plt.subplots(1, 2, figsize=(16, 8))
fd.tricontourf(fd.Function(Vh[STATE], x[STATE]), label='STATE', axes=ax[0])
# ax[0].legend()
ax[0].set_title("State space")
plt.colorbar(ax[0].collections[0])
fd.tricontourf(fd.Function(Vh[PARAMETER], x[PARAMETER]), label='PARAMETER', axes=ax[1])
# ax[1].legend()
ax[1].set_title("Parameter space")
plt.colorbar(ax[1].collections[0])
plt.show()
