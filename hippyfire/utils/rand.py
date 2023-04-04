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
import time

def randomGen(Vh):
    pcg = fd.PCG64(seed=int(time.time()))
    rg = fd.Generator(pcg)
    #f_beta = rg.beta(Vh, 1.0, 2.0)
    return rg.normal(Vh, 0., 1.)
