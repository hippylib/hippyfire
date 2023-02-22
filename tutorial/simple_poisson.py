import firedrake as fd
import matplotlib.pyplot as plt
mesh = fd.UnitSquareMesh(6, 4)
V = fd.FunctionSpace(mesh, "Lagrange", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

x = fd.SpatialCoordinate(V.mesh())
u0 = 1 + x[0] * x[0] + 2 * x[1] * x[1]

def u0_boundary(x, on_boundary):
    pass


bc = fd.DirichletBC(V, u0, "on_boundary")

f = fd.Constant(-6.0)

a = fd.inner(fd.nabla_grad(u), fd.nabla_grad(v)) * fd.dx
L = f * v * fd.dx

u = fd.Function(V)
fd.solve(a == L, u, bc)

# plotting functions
fd.triplot(mesh)


fig, axes = plt.subplots()
colors = fd.tricontour(u, axes = axes)
fig.colorbar(colors)
plt.show()
