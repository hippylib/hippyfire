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

def vector2Function(x, Vh):
    # pass
    fun = fd.Function(Vh)
    fun.vector().assign(0.0)
    fun.vector().axpy(1., x)          # axpy throws a compilation error.
    #fun.vector().set_local(x.get_local())
    return fun

def applyBC(x, Vh, bcs):
    if len(bcs) == 0:
        return
    
    xfun = vector2Function(x, Vh)
    for bc in bcs:
        bc.apply(xfun)

    x.assign(xfun.vector())
