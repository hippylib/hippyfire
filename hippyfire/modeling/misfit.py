# Copyright (c) 201, The University of Texas at Austin
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

import firedrake as fd
import ufl
import numpy as np
import petsc4py
# from .pointwiseObservation import assemblePointwiseObservation
from modeling.variables import STATE, PARAMETER
from algorithms.linalg import Transpose, matVecMult


from utils.vector2function import vector2Function

class Misfit(object):
    """
    Abstract class to model the misfit component of the cost functional.
    In the following :code:`x` will denote the variable :code:`[u, m, p]`, denoting respectively 
    the state :code:`u`, the parameter :code:`m`, and the adjoint variable :code:`p`.

    The methods in the class misfit will usually access the state u and possibly the
    parameter :code:`m`. The adjoint variables will never be accessed. 
    """
    def cost(self, x):
        """
        Given x evaluate the cost functional.
        Only the state u and (possibly) the parameter m are accessed. """
        raise NotImplementedError("Child class should implement method cost")
        return 0

    def grad(self, i, x, out):
        """
        Given the state and the paramter in :code:`x`, compute the partial gradient of the misfit
        functional in with respect to the state (:code:`i == STATE`) or with respect to the parameter (:code:`i == PARAMETER`).
        """
        raise NotImplementedError("Child class should implement method grad")

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        """
        Set the point for linearization.

        Inputs:

            :code:`x=[u, m, p]` - linearization point

            :code:`gauss_newton_approx (bool)` - whether to use Gauss Newton approximation 
        """
        raise NotImplementedError("Child class should implement method setLinearizationPoint")

    def apply_ij(self, i, j, dir, out):
        """
        Apply the second variation :math:`\delta_{ij}` (:code:`i,j = STATE,PARAMETER`) of the cost in direction :code:`dir`.
        """
        raise NotImplementedError("Child class should implement method apply_ij")


class ContinuousStateObservation(Misfit):
    """
    This class implements continuous state observations in a 
    subdomain :math:`X \subset \Omega` or :math:`X \subset \partial \Omega`.
    """
    def __init__(self, Vh, dX, bcs, form=None):
        """
        Constructor:

            :code:`Vh`: the finite element space for the state variable.

            :code:`dX`: the integrator on subdomain `X` where observation are presents. \
            E.g. :code:`dX = ufl.dx` means observation on all :math:`\Omega` and :code:`dX = ufl.ds` means observations on all :math:`\partial \Omega`.

            :code:`bcs`: If the forward problem imposes Dirichlet boundary conditions :math:`u = u_D \mbox{ on } \Gamma_D`;  \
            :code:`bcs` is a list of :code:`dolfin.DirichletBC` object that prescribes homogeneuos Dirichlet conditions :math:`u = 0 \mbox{ on } \Gamma_D`.

            :code:`form`: if :code:`form = None` we compute the :math:`L^2(X)` misfit: :math:`\int_X (u - u_d)^2 dX,` \
            otherwise the integrand specified in the given form will be used.
        """
        self.Vh = Vh

        if isinstance(bcs, fd.bcs.DirichletBC):
            self.bcs = [bcs]
        elif bcs is None:
            self.bcs = []
        else:
            self.bcs = bcs

        if form is None:
            u, v = fd.TrialFunction(Vh), fd.TestFunction(Vh)
            # self.W = fd.assemble(fd.inner(u, v) * fd.dx)
            form = fd.inner(u, v) * dX
        self.W = fd.assemble(form)

                # if len(bcs):
        #     v, u = form.arguments()
        #     temp = u
        #     form_transpose = fd.replace(form, {u : v, v : temp})
        #     Wt = fd.assemble(form_transpose, bcs=bcs)
        #     # Wt = Transpose(self.W)
        #     # [bc.apply(Wt) for bc in bcs]
        #     self.W = Transpose(Wt)
        #     form_new = self.W.form
        #     self.W = fd.assemble(form_new, bcs=bcs)
            # [bc.apply(self.W) for bc in bcs]
        # create a vector compatible for multiplication with W
        v1, u1 = (self.W.form).arguments()
        self.d = fd.Function(v1.function_space()).vector()  # self.d #rows = self.W #cols
        # self.d = dl.Vector(self.W.mpi_comm())
        # self.W.init_vector(self.d,1)
        self.noise_variance = None

    def cost(self, x):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        r = self.d.copy()
        # r.axpy(-1., x[STATE])
        r.set_local(r.get_local() + (-1. * x[STATE].get_local()))
        v1, u1 = (self.W.form).arguments()
        Wr = fd.Function(u1.function_space()).vector()
        # Wr = dl.Vector(self.W.mpi_comm())
        # self.W.init_vector(Wr,0)
        matVecMult(self.W, r, Wr)
        return r.inner(Wr) / (2.*self.noise_variance)

    def grad(self, i, x, out):
        self.noise_variance = 1.0
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE:
            res = x[STATE]
            res.set_local(x[STATE].get_local() - self.d.get_local())
            if len(self.bcs):
                fun = vector2Function(res, res.function_space())
                for bc in self.bcs:
                    bc.apply(fun)
                res = fun.vector()
            matVecMult(self.W, res, out)
            if len(self.bcs):
                fun = vector2Function(out, out.function_space())
                for bc in self.bcs:
                    bc.apply(fun)
                out = fun.vector()
            out.set_local((1. / self.noise_variance) * out.get_local())
        elif i == PARAMETER:
            out.vector().assign(0.0)
        else:
            raise IndexError()

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        # The cost functional is already quadratic. Nothing to be done here
        return

    def apply_ij(self, i, j, dir, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE and j == STATE:
            if len(self.bcs):
                fun = vector2Function(dir, dir.function_space())
                for bc in self.bcs:
                    bc.apply(fun)
                dir = fun.vector()
            matVecMult(self.W, dir, out)
            if len(self.bcs):
                fun = vector2Function(out, out.function_space())
                for bc in self.bcs:
                    bc.apply(fun)
                out = fun.vector()
            out.set_local((1. / self.noise_variance) * out.get_local())
            # out *= (1./self.noise_variance)
        else:
            out.vector().assign(0.0)
