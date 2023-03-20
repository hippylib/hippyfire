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
from variables import STATE, PARAMETER, ADJOINT
from linalg import Transpose
from linSolvers import CreateSolver
from vector2function import vector2Function

class PDEProblem(object):
    """ Consider the PDE problem:
        Given :math:`m`, find :math:`u` such that 
        
            .. math:: F(u, m, p) = ( f(u, m), p) = 0, \\quad \\forall p.
        
        Here :math:`F` is linear in :math:`p`, but it may be non linear in :math:`u` and :math:`m`.
    """

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        raise NotImplementedError("Child class should implement method generate_state")

    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        raise NotImplementedError("Child class should implement method generate_parameter")

    def init_parameter(self, m):
        """ Initialize the parameter. """
        raise NotImplementedError("Child class should implement method init_parameter")

    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that

            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0, \\quad \\forall \\hat{p}.
        """
        raise NotImplementedError("Child class should implement method solveFwd")

    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem: 
            Given :math:`m`, :math:`u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        raise NotImplementedError("Child class should implement method solveAdj")

    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        raise NotImplementedError("Child class should implement method evalGradientParameter")
 
    def setLinearizationPoint(self,x, gauss_newton_approx):

        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. 
            Set whether Gauss Newton approximation of
            the Hessian should be used."""
        raise NotImplementedError("Child class should implement method setLinearizationPoint")
      
    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj = False`:

            Solve the forward incremental system:
            Given :math:`u, m`, find :math:`\\tilde{u}` such that

            .. math::
                \\delta_{pu} F(u, m, p; \\hat{p}, \\tilde{u}) = \\mbox{rhs}, \\quad \\forall \\hat{p}.
            
            If :code:`is_adj = True`:
            
            Solve the adjoint incremental system:
            Given :math:`u, m`, find :math:`\\tilde{p}` such that

            .. math::
                \\delta_{up} F(u, m, p; \\hat{u}, \\tilde{p}) = \\mbox{rhs}, \\quad \\forall \\hat{u}.
        """
        raise NotImplementedError("Child class should implement method solveIncremental")

    def apply_ij(self,i,j, dir, out):   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`, 
            :math:`\\forall \\hat{i}.`
        """
        raise NotImplementedError("Child class should implement method apply_ij")
        
    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        """
            Given :code:`x = [u,a,p]`; compute
            :math:`\\delta_{ijk} F(u,a,p; \\hat{i}, \\tilde{j}, \\tilde{k})`
            in the direction :math:`(\\tilde{j},\\tilde{k}) = (`:code:`jdir,kdir`), :math:`\\forall \\hat{i}.`
        """
        raise NotImplementedError("Child class should implement apply_ijk")

class PDEVariationalProblem(PDEProblem):
    def __init__(self, Vh, varf_handler, bc, bc0, is_fwd_linear = False):
        self.Vh = Vh
        self.varf_handler = varf_handler
        if type(bc) is fd.bcs.DirichletBC:
            self.bc = [bc]
        else:
            self.bc = bc
        if type(bc0) is fd.bcs.DirichletBC:
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0
        
        self.A  = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None
        self.solver = None
        self.solver_fwd_inc = None
        self.solver_adj_inc = None
        
        self.is_fwd_linear = is_fwd_linear
        self.n_calls = {"forward": 0,
                        "adjoint":0 ,
                        "incremental_forward":0,
                        "incremental_adjoint":0}

    def generate_state(self):
        """ Return a vector in the shape of the state. """
        return fd.Function(self.Vh[STATE]).vector()
    
    def generate_parameter(self):
        """ Return a vector in the shape of the parameter. """
        return fd.Function(self.Vh[PARAMETER]).vector()
    
    def init_parameter(self, m):
        """ Initialize the parameter. """
        dummy = self.generate_parameter()
        m.init( dummy.comm, dummy.local_range() )
    
    def solveFwd(self, state, x):
        """ Solve the possibly nonlinear forward problem:
        Given :math:`m`, find :math:`u` such that
        
            .. math:: \\delta_p F(u, m, p;\\hat{p}) = 0,\\quad \\forall \\hat{p}."""
        # Firedrake solver requires an operator A to be defined for a solver
        if self.A is None:      # confirm if the index for self.Vh is correct
            self.A = fd.assemble(fd.inner(fd.TestFunction(self.Vh[STATE]),
                                          fd.TrialFunction(self.Vh[STATE])) * fd.dx)
        self.n_calls["forward"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
        if self.is_fwd_linear:
            u = fd.TrialFunction(self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = fd.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u, m, p)
            A_form = ufl.lhs(res_form)
            b_form = ufl.rhs(res_form)
            A = fd.assemble(A_form, bcs=self.bc)
            b = fd.assemble(b_form, bcs=self.bc)
            self.solver.operator(A)
            self.solver.solve(state, b)
        else:
            u = vector2Function(x[STATE], self.Vh[STATE])
            m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
            p = fd.TestFunction(self.Vh[ADJOINT])
            res_form = self.varf_handler(u, m, p)
            fd.solve(res_form == 0, u, bcs=self.bc)
            state.vector().assign(0.0)
            # state.axpy(1., u.vector())    # axpy in fd gives compilation error
            state.vector().set_local(u.vector().get_local())
        
    def solveAdj(self, adj, x, adj_rhs):
        """ Solve the linear adjoint problem: 
            Given :math:`m, u`; find :math:`p` such that
            
                .. math:: \\delta_u F(u, m, p;\\hat{u}) = 0, \\quad \\forall \\hat{u}.
        """
        self.n_calls["adjoint"] += 1
        if self.solver is None:
            self.solver = self._createLUSolver()
            
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = fd.Function(self.Vh[ADJOINT])
        du = fd.TestFunction(self.Vh[STATE])
        dp = fd.TrialFunction(self.Vh[ADJOINT])
        varf = self.varf_handler(u, m, p)
        adj_form = fd.derivative(fd.derivative(varf, u, du), p, dp )
        Aadj= fd.assemble(adj_form, bcs=self.bc0)
        # dummy = fd.assemble(fd.inner(u , du) * fd.dx, bcs=self.bc0)
        self.solver.operator(Aadj)
        self.solver.solve(adj, adj_rhs)
     
    def evalGradientParameter(self, x, out):
        """Given :math:`u, m, p`; evaluate :math:`\\delta_m F(u, m, p; \\hat{m}),\\, \\forall \\hat{m}.` """
        u = vector2Function(x[STATE], self.Vh[STATE])
        m = vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = fd.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)
        out.vector().assign(0.0)
        fd.assemble(fd.derivative(res_form, m, dm), tensor=out)
         
    def setLinearizationPoint(self, x, gauss_newton_approx):
        """ Set the values of the state and parameter
            for the incremental forward and adjoint solvers. """
            
        x_fun = [vector2Function(x[i], self.Vh[i]) for i in range(3)]
        
        f_form = self.varf_handler(*x_fun)
        
        g_form = [None, None, None]
        for i in range(3):
            g_form[i] = fd.derivative(f_form, x_fun[i])
            
        self.A = fd.assemble(fd.derivative(g_form[ADJOINT],x_fun[STATE]), self.bc0)
        self.At  = fd.assemble(fd.derivative(g_form[STATE],x_fun[ADJOINT]), self.bc0)
        self.C = fd.assemble(fd.derivative(g_form[ADJOINT],x_fun[PARAMETER]))
        # [bc.zero(self.C) for bc in self.bc0]
        for bc in self.bc0:
            bc.homogenize()
            bc.apply(self.C)

        if self.solver_fwd_inc is None:
            self.solver_fwd_inc = self._createLUSolver()
            self.solver_adj_inc = self._createLUSolver()
        
        self.solver_fwd_inc.operator(self.A)
        self.solver_adj_inc.operator(self.At)

        if gauss_newton_approx:
            self.Wuu = None
            self.Wmu = None
            self.Wmm = None
        else:
            self.Wuu = fd.assemble(fd.derivative(g_form[STATE],x_fun[STATE]))
            for bc in self.bc0:
                bc.homogenize()
                bc.apply(self.Wuu)
            Wuu_t = Transpose(self.Wuu)
            for bc in self.bc0:
                bc.homogenize()
                bc.apply(Wuu_t)
            # [bc.zero(Wuu_t) for bc in self.bc0]
            self.Wuu = Transpose(Wuu_t)
            self.Wmu = fd.assemble(fd.derivative(g_form[PARAMETER],x_fun[STATE]))
            Wmu_t = Transpose(self.Wmu)
            for bc in self.bc0:
                bc.homogenize()
                bc.apply(Wmu_t)
            # [bc.zero(Wmu_t) for bc in self.bc0]
            self.Wmu = Transpose(Wmu_t)
            self.Wmm = fd.assemble(fd.derivative(g_form[PARAMETER],x_fun[PARAMETER]))
        
    def solveIncremental(self, out, rhs, is_adj):
        """ If :code:`is_adj == False`:

            Solve the forward incremental system:
            Given :math:`u, m`, find :math:`\\tilde{u}` such that
            
                .. math:: \\delta_{pu} F(u, m, p; \\hat{p}, \\tilde{u}) = \\mbox{rhs},\\quad \\forall \\hat{p}.
            
            If :code:`is_adj == True`:

            Solve the adjoint incremental system:
            Given :math:`u, m`, find :math:`\\tilde{p}` such that
            
                .. math:: \\delta_{up} F(u, m, p; \\hat{u}, \\tilde{p}) = \\mbox{rhs},\\quad \\forall \\hat{u}.
        """
        if is_adj:
            self.n_calls["incremental_adjoint"] += 1
            self.solver_adj_inc.solve(out, rhs)
        else:
            self.n_calls["incremental_forward"] += 1
            self.solver_fwd_inc.solve(out, rhs)
    
    def apply_ij(self,i,j, dir, out):   
        """
            Given :math:`u, m, p`; compute 
            :math:`\\delta_{ij} F(u, m, p; \\hat{i}, \\tilde{j})` in the direction :math:`\\tilde{j} =` :code:`dir`,
            :math:`\\forall \\hat{i}`.
        """
        KKT = {}
        KKT[STATE,STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C
        
        if i >= j:
            if KKT[i,j] is None:
                out.vector().assign(0.0)
            else:
                matVecMult(KKT[i, j], dir, out)
        else:
            if KKT[j,i] is None:
                out.vector().assign(0.0)
            # else:
            #     KKT[j,i].transpmult(dir, out)
                
    def apply_ijk(self,i,j,k, x, jdir, kdir, out):
        x_fun = [vector2Function(x[ii], self.Vh[ii]) for ii in range(3)]
        idir_fun = fd.TestFunction(self.Vh[i])
        jdir_fun = vector2Function(jdir, self.Vh[j])
        kdir_fun = vector2Function(kdir, self.Vh[k])
        
        res_form = self.varf_handler(*x_fun)
        form = fd.derivative(
               fd.derivative(
               fd.derivative(res_form, x_fun[i], idir_fun),
               x_fun[j], jdir_fun),
               x_fun[k], kdir_fun)
        
        out.vector().assign(0.0)
        fd.assemble(form, tensor=out)
        
        if i in [STATE,ADJOINT]:
            [bc.apply(out) for bc in self.bc0]
                   
    def _createLUSolver(self):
        # Can be used to create different solvers by specifying ksp and pre
        solver = CreateSolver(self.A, self.Vh[STATE].mesh().comm() )
        return solver
