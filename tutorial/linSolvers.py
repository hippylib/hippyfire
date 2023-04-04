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

def CreateSolver(A, comm, ksp_type='preonly', pc_type='lu'):
    if not isinstance(A, (fd.matrix.Matrix)):
        raise TypeError("Provided Operator is a '%s', not a Firedrake Matrix".type(A).__name__)
    # DEFAULT_KSP_PARAMETERS = {'ksp_rtol': 1e-07, 'ksp_type': 'preonly',
    # 'mat_mumps_icntl_14': 200, 'mat_type': 'aij',
    # 'pc_factor_mat_solver_type': 'mumps', 'pc_type': 'lu'}
    solver_parameters = {'ksp_type' : ksp_type, 'pc_type' : pc_type}
    solver = fd.LinearSolver(A, solver_parameters=solver_parameters)
    return solver
