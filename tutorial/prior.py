# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

# import dolfin as dl
import firedrake as fd
import ufl
import numpy as np
# import scipy.linalg as scila
import math
from petsc4py import PETSc
import numbers

# from ..algorithms.linalg import MatMatMult, get_diagonal, amg_method, estimate_diagonal_inv2, Solver2Operator, Operator2Solver
# from ..algorithms.linSolvers import PETScKrylovSolver
# from ..algorithms.traceEstimator import TraceEstimator
# from ..algorithms.multivector import MultiVector
# from ..algorithms.randomizedEigensolver import doublePass, doublePassG

# from ..utils.random import parRandom
from linalg import Transpose, matVecMult
from linSolvers import CreateSolver
from vector2function import vector2Function

# from .expression import ExpressionModule


class _Prior:
    """
    Abstract class to describe the prior model.
    Concrete instances of a :code:`_Prior class` should expose
    the following attributes and methods.
    
    Attributes:

    - :code:`R`:       an operator to apply the regularization/precision operator.
    - :code:`Rsolver`: an operator to apply the inverse of the regularization/precision operator.
    - :code:`M`:       the mass matrix in the control space.
    - :code:`mean`:    the prior mean.
    
    Methods:

    - :code:`init_vector(self,x,dim)`: Inizialize a vector :code:`x` to be compatible with the range/domain of :code:`R`
      If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
      white noise used for sampling.
      
    - :code:`sample(self, noise, s, add_mean=True)`: Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample s from the prior.
      If :code:`add_mean==True` add the prior mean value to :code:`s`.
    """ 

        
    def cost(self, m):
        d = self.mean.copy()
        # d.axpy(-1., m)        # axpy gives a compilation error
        d.set_local(d.get_local() - (1. * m.get_local()))
        # Rd = dl.Vector(self.R.mpi_comm())
        # self.init_vector(Rd,0)
        v1, u1 = (self.R.A.form).arguments()
        Rd = fd.Function(v1.function_space()).vector()
        Rd = matVecMult(self.R.A, d, Rd)
        return .5 * Rd.inner(d)



    def grad(self,m, out):
        d = m.copy()
        # d.axpy(-1., self.mean)
        d.set_local(d.get_local() + (-1. * self.mean.get_local()))
        out = matVecMult(self.R.A, d, out)

    def init_vector(self,x,dim):
        raise NotImplementedError("Child class should implement method init_vector")

    def sample(self, noise, s, add_mean=True):
        raise NotImplementedError("Child class should implement method sample")

    def getHessianPreconditioner(self):
        " Return the preconditioner for Newton-CG "
        return self.Rsolver
        

class _BilaplacianR:
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, A, Msolver):
        self.A = A
        self.Msolver = Msolver

        v1, u1 = (self.A.form).arguments()
        self.help1 = fd.Function(u1.function_space()).vector()
        self.help2 = fd.Function(v1.function_space()).vector()
        # self.help1, self.help2 = dl.Vector(self.A.mpi_comm()), dl.Vector(self.A.mpi_comm())
        # self.A.init_vector(self.help1, 0)
        # self.A.init_vector(self.help2, 1)
        
    def init_vector(self, x, dim): # confirm this once
        v1, u1 = (self.A.form).arguments()
        x = fd.Function(v1.function_space()).vector()
        return x
        # self.A.init_vector(x,1)
        
    def mpi_comm(self):         # confirm once. Not defined in firedrake
        return self.A.comm
        
    def mult(self, x, y):         # confirm naming of the methods in this class
        self.help1 = matVecMult(self.A, x, self.help1)
        self.Msolver.solve(self.help2, self.help1)
        y = matVecMult(self.A, self.help2, y)


class _BilaplacianRsolver():
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, Asolver, M):
        self.Asolver = Asolver
        self.M = M

        v1, u1 = (self.M.form).arguments()
        self.help1 = fd.Function(u1.function_space()).vector()
        self.help2 = fd.Function(u1.function_space()).vector()
        # self.help1, self.help2 = dl.Vector(self.M.mpi_comm()), dl.Vector(self.M.mpi_comm())
        # self.init_vector(self.help1, 0)
        # self.init_vector(self.help2, 0)
        
    def init_vector(self, x, dim):
        # self.M.init_vector(x,1)
        v1, u1 = (self.M.form).arguments()
        x = fd.Function(v1.function_space).vector()
        return x

    def solve(self, x, b):
        nit = self.Asolver.solve(self.help1, b)
        self.help2 = matVecMult(self.M, self.help1, self.help2)
        nit += self.Asolver.solve(x, self.help2)
        return nit


def BiLaplacianComputeCoefficients(sigma2, rho, ndim):
    """
    This class is responsible to compute the parameters gamma and delta
    for the BiLaplacianPrior given the marginal variance sigma2 and 
    correlation length rho. ndim is the dimension of the domain 2D or 3D
    """
        
    nu = 2. - 0.5*ndim
    kappa = np.sqrt(8*nu)/rho
    
    s = np.sqrt(sigma2)*np.power(kappa,nu)*np.sqrt(np.power(4.*np.pi, 0.5*ndim)/math.gamma(nu) )
    
    gamma = 1./s
    delta = np.power(kappa,2)/s
    
    return gamma, delta
    
class SqrtPrecisionPDE_Prior(_Prior):
    """
    This class implement a prior model with covariance matrix
    :math:`C = A^{-1} M A^-1`,
    where A is the finite element matrix arising from discretization of sqrt_precision_varf_handler
    
    """
    
    def __init__(self, Vh, sqrt_precision_varf_handler, mean=None, rel_tol=1e-12, max_iter=1000):
        """
        Construct the prior model.
        Input:

        - :code:`Vh`:              the finite element space for the parameter
        - :code:sqrt_precision_varf_handler: the PDE representation of the  sqrt of the covariance operator
        - :code:`mean`:            the prior mean
        """
        # sqrt_precision_varf_handler:
        self.Vh = Vh
        
        trial = fd.TrialFunction(Vh)
        test  = fd.TestFunction(Vh)
        
        varfM = fd.inner(trial, test) * fd.dx
        self.M = fd.assemble(varfM)
        self.Msolver = CreateSolver(self.M, self.Vh.mesh().mpi_comm(), ksp_type="cg", pc_type="jacobi")
        # self.Msolver.set_operator(self.M)
        # self.Msolver.parameters["maximum_iterations"] = max_iter
        # self.Msolver.parameters["relative_tolerance"] = rel_tol
        # self.Msolver.parameters["error_on_nonconvergence"] = True
        # self.Msolver.parameters["nonzero_initial_guess"] = False
        
        self.A = fd.assemble(sqrt_precision_varf_handler(trial, test))
        self.Asolver = CreateSolver(self.A, self.Vh.mesh().mpi_comm(), ksp_type="cg", pc_type="gamg")
        # self.Asolver.set_operator(self.A)
        # self.Asolver.parameters["maximum_iterations"] = max_iter
        # self.Asolver.parameters["relative_tolerance"] = rel_tol
        # self.Asolver.parameters["error_on_nonconvergence"] = True
        # self.Asolver.parameters["nonzero_initial_guess"] = False
        
        # old_qr = dl.parameters["form_compiler"]["quadrature_degree"]
                #
        self.R = _BilaplacianR(self.A, self.Msolver)
        self.Rsolver = _BilaplacianRsolver(self.Asolver, self.M)
         
        self.mean = mean
        
        if self.mean is None:
            self.mean = self.init_vector(self.mean, 0)
     ###

    def init_vector(self, x, dim):  # confirm what is sqrtM
        """
        Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.

        If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            # self.sqrtM.init_vector(x, 1)
            pass
        else:
            v2, u2 = (self.A.form).arguments()
            if dim == 0:
                x = fd.Function(u2.function_space()).vector()
            elif dim == 1:
                x = fd.Function(v2.function_space()).vector()
                # self.A.init_vector(x,dim)
        return x
        
def BiLaplacianPrior(Vh, gamma, delta, Theta = None, mean=None, rel_tol=1e-12, max_iter=1000, robin_bc=False):
    """
    This function construct an instance of :code"`SqrtPrecisionPDE_Prior`  with covariance matrix
    :math:`C = (\\delta I + \\gamma \\mbox{div } \\Theta \\nabla) ^ {-2}`.
    
    The magnitude of :math:`\\delta\\gamma` governs the variance of the samples, while
    the ratio :math:`\\frac{\\gamma}{\\delta}` governs the correlation lenght.
    
    Here :math:`\\Theta` is a SPD tensor that models anisotropy in the covariance kernel.
    
    Input:

    - :code:`Vh`:              the finite element space for the parameter
    - :code:`gamma` and :code:`delta`: the coefficient in the PDE (floats, dl.Constant, dl.Expression, or dl.Function)
    - :code:`Theta`:           the SPD tensor for anisotropic diffusion of the PDE
    - :code:`mean`:            the prior mean
    - :code:`rel_tol`:         relative tolerance for solving linear systems involving covariance matrix
    - :code:`max_iter`:        maximum number of iterations for solving linear systems involving covariance matrix
    - :code:`robin_bc`:        whether to use Robin boundary condition to remove boundary artifacts
    """
    if isinstance(gamma, numbers.Number):
        gamma = fd.Constant(gamma)

    if isinstance(delta, numbers.Number):
        delta = fd.Constant(delta)

    
    def sqrt_precision_varf_handler(trial, test):
        if Theta == None:
            varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
        else:
            varfL = ufl.inner( Theta*ufl.grad(trial), ufl.grad(test))*ufl.dx

        varfM = ufl.inner(trial,test)*ufl.dx

        varf_robin = ufl.inner(trial,test)*ufl.ds

        if robin_bc:
            robin_coeff = gamma*ufl.sqrt(delta/gamma)/fd.Constant(1.42)
        else:
            robin_coeff = fd.Constant(0.)

        return gamma*varfL + delta*varfM + robin_coeff*varf_robin


    return SqrtPrecisionPDE_Prior(Vh, sqrt_precision_varf_handler, mean, rel_tol, max_iter)
