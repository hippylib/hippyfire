from firedrake import *
from firedrake.petsc import PETSc

mesh = UnitSquareMesh(5, 2)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
# print(a)
print("Conducting transpose now")

# print(u)
# print(v)
temp = u
a_new = replace(a, {u : v, v : temp})
print(a_new)
A = (assemble(a)).M.handle
Atrans = (assemble(a_new)).M.handle
# assert (A - Atrans).norm() < 1e-14
print(A.norm())
print(A.transpose().norm())
print(Atrans.equal(A.transpose()))
