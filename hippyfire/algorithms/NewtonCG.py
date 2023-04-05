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

import math
from utils.parameterList import ParameterList
from modeling.reducedHessian import ReducedHessian
from modeling.variables import STATE, PARAMETER, ADJOINT
from algorithms.cgsolverSteihaug import CGSolverSteihaug

def LS_ParameterList():
    """
    Generate a ParameterList for line search globalization.
    type: :code:`LS_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["c_armijo"]              = [1e-4, "Armijo constant for sufficient reduction"]
    parameters["max_backtracking_iter"] = [10, "Maximum number of backtracking iterations"]

    return ParameterList(parameters)

def TR_ParameterList():
    """
    Generate a ParameterList for Trust Region globalization.
    type: :code:`RT_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["eta"] = [0.05, "Reject step if (actual reduction)/(predicted reduction) < eta"]

    return ParameterList(parameters)

def ReducedSpaceNewtonCG_ParameterList():
    """
    Generate a ParameterList for ReducedSpaceNewtonCG.
    type: :code:`ReducedSpaceNewtonCG_ParameterList().showMe()` for default values and their descriptions
    """
    parameters = {}
    parameters["rel_tolerance"]         = [1e-6, "we converge when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
    parameters["abs_tolerance"]         = [1e-12, "we converge when sqrt(g,g) <= abs_tolerance"]
    parameters["gdm_tolerance"]         = [1e-18, "we converge when (g,dm) <= gdm_tolerance"]
    parameters["max_iter"]              = [20, "maximum number of iterations"]
    parameters["globalization"]         = ["LS", "Globalization technique: line search (LS)  or trust region (TR)"]
    parameters["print_level"]           = [0, "Control verbosity of printing screen"]
    parameters["GN_iter"]               = [5, "Number of Gauss Newton iterations before switching to Newton"]
    parameters["cg_coarse_tolerance"]   = [.5, "Coarsest tolerance for the CG method (Eisenstat-Walker)"]
    parameters["cg_max_iter"]           = [100, "Maximum CG iterations"]
    parameters["LS"]                    = [LS_ParameterList(), "Sublist containing LS globalization parameters"]
    parameters["TR"]                    = [TR_ParameterList(), "Sublist containing TR globalization parameters"]

    return ParameterList(parameters)
