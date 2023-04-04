import sys
import os
sys.path.insert(0, os.environ.get('HIPPYFIRE_BASE_DIR'))
from modeling.PDEProblem import PDEVariationalProblem

pde = PDEVariationalProblem()
