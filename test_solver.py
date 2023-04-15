# from z3 import *
import numpy as np
#
#
# solver = Solver()
# problem = [Or(And(Bool('13s'), Not(Bool('22s')), Not(Bool('33s')), Not(Bool('24s'))),
#               And(Not(Bool('13s')), Bool('22s'), Not(Bool('33s')), Not(Bool('24s'))),
#               And(Not(Bool('13s')), Not(Bool('22s')), Bool('33s'), Not(Bool('24s'))),
#               And(Not(Bool('13s')), Not(Bool('22s')), Not(Bool('33s')), Bool('24s')),
#               )]
#
#
# for i in range(10):
#     solver.reset()
#     #sat.random_seed = i
#     #smt.random_seed = i
#     solver.add(problem)
#     if solver.check() == sat:
#         solution = solver.model()
#         print(solution)

# --------------------------

from pysmt.shortcuts import Solver, Symbol, And, Or, Not, Iff, ExactlyOne, AllDifferent, Equals, NotEquals, GT, LE, LT, Int, Minus, Plus
from pysmt.typing import BOOL, INT

k = [Symbol(str(i), INT) for i in range(1, 5)]
cp1 = Int(1) # constant for 1
cm1 = Int(-1) # constant for -1


problem = ExactlyOne(Equals(k[0], Int(5)), Equals(k[1], Int(5)), Equals(k[2], Int(5)), Equals(k[3], Int(5)))

problem = ExactlyOne(Symbol('a', BOOL), Symbol('b', BOOL), Symbol('c', BOOL), Symbol('d', BOOL))


#condition = LT(k, Int(10))

# problem = Or(And(Symbol('13s'), Not(Symbol('22s')), Not(Symbol('33s')), Not(Symbol('24s'))),
#              And(Not(Symbol('13s')), Symbol('22s'), Not(Symbol('33s')), Not(Symbol('24s'))),
#              And(Not(Symbol('13s')), Not(Symbol('22s')), Symbol('33s'), Not(Symbol('24s'))),
#              And(Not(Symbol('13s')), Not(Symbol('22s')), Not(Symbol('33s')), Symbol('24s')))

# problem = ExactlyOne(Symbol('11s'), Symbol('12s'), Symbol('21s'), Symbol('22s'))

for i in range(1):
    solver = Solver(name="z3", random_seed=np.random.randint(100))
    solver.add_assertion(problem)
    if solver.solve():
        solution = solver.get_model()
        print("-------")
        print(solution)
