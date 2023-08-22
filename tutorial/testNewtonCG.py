# loading modules
import sys
import os
import math
import firedrake as fd
import ufl
import numpy as np
import matplotlib.pyplot as plt
import time
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
from utils.vector2function import vector2Function

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
mtrue = randomGen(Vh[PARAMETER])

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
objs = [fd.Function(Vh[PARAMETER], mtrue.vector()), fd.Function(Vh[PARAMETER], pr.mean)]
fig, ax = plt.subplots(1, 2, figsize=(20, 20))
fd.tricontourf(objs[0], antialiased=True, label='True Parameter', axes=ax[0])
# ax[0].legend()
ax[0].set_title("True Parameter", fontsize=20)
cbar1 = plt.colorbar(ax[0].collections[0])
fd.tricontourf(objs[1], label='PARAMETER', axes=ax[1])
# ax[1].legend()
ax[1].set_title("Prior Mean", fontsize=20)
cbar2 = plt.colorbar(ax[1].collections[0])
cbar1.ax.tick_params(labelsize=20)
cbar2.ax.tick_params(labelsize=20)

ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[0].tick_params(axis='both', labelsize=20)
ax[1].tick_params(axis='both', labelsize=20)
plt.show()

# Set up misfit
misfit = ContinuousStateObservation(Vh[STATE], ufl.dx, bcs=bc0)
utrue = pde.generate_state()
x = [utrue, mtrue.vector(), None]
pde.solveFwd(x[STATE], x)
misfit.d.axpy(1., utrue)
rel_noise = 0.01
MAX = np.linalg.norm(misfit.d.get_local(), ord=np.inf)
noise_std_dev = rel_noise * MAX
temp = randomGen(Vh[STATE]).vector()
misfit.d.axpy(float(noise_std_dev), temp)
misfit.noise_variance = noise_std_dev * noise_std_dev

# misfit.d.axpy(float(np.sqrt(misfit.noise_variance)), randomGen(Vh[STATE]).vector())
# print(misfit.dgithub..get_local())
vmax = max( utrue.get_local().max(), misfit.d.get_local().max() )
vmin = min( utrue.get_local().min(), misfit.d.get_local().min() )
print(vmax, vmin)
ntargets = misfit.d.size()
targets_x = np.random.uniform(0.1,0.9, [ntargets])
targets_y = np.random.uniform(0.1,0.5, [ntargets])
targets = np.zeros([ntargets, ndim])
targets[:, 0] = targets_x
targets[:, 1] = targets_y

fig, ax = plt.subplots(1, 2, figsize=(20, 12))
fd.tricontourf(fd.Function(Vh[STATE], utrue).vector(), antialiased=True, label='True Parameter', axes=ax[0])

ax[0].set_title("True State", fontsize=20)
cbar1 = plt.colorbar(ax[0].collections[0])

plt.scatter(targets[:, 0], targets[:, 1], c= misfit.d.get_local(), s=1)
ax[1].set_title("Observations", fontsize=20)
cbar2 = plt.colorbar(ax[1].collections[0])
cbar1.ax.tick_params(labelsize=20)
cbar2.ax.tick_params(labelsize=20)

ax[0].set_aspect(1)
ax[1].set_aspect(((ax[1].get_xlim()[1] - ax[1].get_xlim()[0])/(ax[1].get_ylim()[1] - ax[1].get_ylim()[0])) * 1)
ax[0].tick_params(axis='both', labelsize=20)
ax[1].tick_params(axis='both', labelsize=20)
plt.show()




model = Model(pde, pr, misfit)

# print("Test only misfit")
# eps, err_grad, err_H = modelVerify(model, m0, misfit_only=True)

# print(err_grad)
# print(err_H)

# print("Test also prior")
eps, err_grad, err_H = modelVerify(model, m0, misfit_only=False)

# print(err_grad)
# print(err_H)




m = pr.mean.copy()
solver = ReducedSpaceNewtonCG(model)
solver.parameters["rel_tolerance"] = 1e-6
solver.parameters["abs_tolerance"] = 1e-12
solver.parameters["max_iter"]      = 25
solver.parameters["GN_iter"] = 5
solver.parameters["globalization"] = "LS"
solver.parameters["LS"]["c_armijo"] = 1e-4

start = time.time()
x = solver.solve([None, m, None])
end = time.time()
print(end - start, "Executiion time")


if solver.converged:
    print( "\nConverged in ", solver.it, " iterations.")
else:
    print( "\nNot Converged")

print( "Termination reason: ", solver.termination_reasons[solver.reason] )
print( "Final gradient norm: ", solver.final_grad_norm )
print( "Final cost: ", solver.final_cost )


fig, ax = plt.subplots(1, 2, figsize=(20, 12))
fd.tricontourf(fd.Function(Vh[STATE], x[STATE]), antialiased=True, label='STATE', axes=ax[0])

ax[0].set_title("State", fontsize=20)
cbar1 = plt.colorbar(ax[0].collections[0])
fd.tricontourf(fd.Function(Vh[PARAMETER], x[PARAMETER]), label='PARAMETER', axes=ax[1])
ax[1].set_title("Parameter", fontsize=20)
cbar2 = plt.colorbar(ax[1].collections[0])
cbar1.ax.tick_params(labelsize=20)
cbar2.ax.tick_params(labelsize=20)

ax[0].set_aspect(1)
ax[1].set_aspect(1)
ax[0].tick_params(axis='both', labelsize=20)
ax[1].tick_params(axis='both', labelsize=20)
plt.show()
