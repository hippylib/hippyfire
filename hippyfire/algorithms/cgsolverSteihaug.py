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

import firedrake as fd
from utils.parameterList import ParameterList
import math


def CGSolverSteihaug_ParameterList():
    """
    Generate a :code:`ParameterList` for :code:`CGSolverSteihaug`.
    Type :code:`CGSolverSteihaug_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"] = [1e-9, "the relative tolerance for the stopping criterion"]
    parameters["abs_tolerance"] = [1e-12, "the absolute tolerance for the stopping criterion"]
    parameters["max_iter"]      = [1000, "the maximum number of iterations"]
    parameters["zero_initial_guess"] = [True, "if True we start with a 0 initial guess; if False we use the x as initial guess."]
    parameters["print_level"] = [0, "verbosity level: -1 --> no output on screen; 0 --> only final residual at convergence or reason for not not convergence"]
    return ParameterList(parameters)


class CGSolverSteihaug:
    """
    Solve the linear system :math:`A x = b` using preconditioned conjugate gradient ( :math:`B` preconditioner)
    and the Steihaug stopping criterion:

    - reason of termination 0: we reached the maximum number of iterations (no convergence)
    - reason of termination 1: we reduced the residual up to the given tolerance (convergence)
    - reason of termination 2: we reached a negative direction (premature termination due to not spd matrix)
    - reason of termination 3: we reached the boundary of the trust region

    The stopping criterion is based on either

    - the absolute preconditioned residual norm check: :math:`|| r^* ||_{B^{-1}} < atol`
    - the relative preconditioned residual norm check: :math:`|| r^* ||_{B^{-1}}/|| r^0 ||_{B^{-1}} < rtol,`

    where :math:`r^* = b - Ax^*` is the residual at convergence and :math:`r^0 = b - Ax^0` is the initial residual.

    The     operator :code:`A` is set using the method :code:`set_operator(A)`.
    :code:`A    ` must provide the following two methods:

    - :code:`A.mult(x,y)`: `y = Ax`
    - :code:`A.init_vector(x, dim)`: initialize the vector `x` so that it is compatible with the range `(dim = 0)` or
      the domain `(dim = 1)` of :code:`A`.

    The preconditioner :code:`B` is set using the method :code:`set_preconditioner(B)`.
    :code:`B` must provide the following method:
    - :code:`B.solve(z,r)`: `z` is the action of the preconditioner :code:`B` on the vector `r`

    To solve the linear system :math:`Ax = b` call :code:`self.solve(x,b)`.
    Here :code:`x` and :code:`b` are assumed to be :code:`dolfin.Vector` objects.

    Type :code:`CGSolverSteihaug_ParameterList().showMe()` for default parameters and their descriptions
    """

    reason = ["Maximum Number of Iterations Reached",
              "Relative/Absolute residual less than tol",
              "Reached a negative direction",
              "Reached a trust region boundary"
              ]
    def __init__(self, parameters=CGSolverSteihaug_ParameterList(), comm=fd.COMM_WORLD, Vh):
        # using Vh instead of comm to construct vectors because Firedrake cannot construct
        # vectors with comm yet. Vh is obtained by extracting the function space of the
        # BilaplcianR attribute of the prior
        self.parameters = parameters

        self.A = None
        self.B_solver = None
        self.B_op = None
        self.converged = False
        self.iter = 0
        self.reasonid = 0
        self.final_norm = 0

        self.TR_radius_2 = None

        self.update_x = self.update_x_without_TR

        self.r = fd.Function(Vh).vector()
        self.z = fd.Function(Vh).vector()
        self.Ad = fd.Function(Vh).vector()
        self.d = fd.Function(Vh).vector()
        self.Bx = fd.Function(Vh).vector()
