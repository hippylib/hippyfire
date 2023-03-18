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
from pyop2 import op2
from firedrake.petsc import PETSc

# from ..utils.random import parRandom
import numpy as np


def Transpose(A):
    a = A.form                  # extract UFL form of matrix
    v, u = a.arguments()        # extract trial and test functions
    temp = u
    a_new = fd.replace(a, {u : v, v : temp}) # creating transposed bilinear form
    AT = fd.assemble(a_new)
    return AT

def innerFire(x, y):            # custom function to dot two vectors defined on diff. function spaces
    arr = x.get_local() * y.get_local()
    x.set_local(arr)
    return x
