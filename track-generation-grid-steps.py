import itertools
import numpy as np
from pysmt.shortcuts import Solver, Symbol, And, Or, Not, Implies, Iff, ExactlyOne, get_atoms
from pysmt.typing import BOOL
import pprint
from timeit import timeit


class Generator_grid:
    """Track Generator using SAT solver
       It uses 3 track pieces to generate a random track on a grid. (straight, left, right)
    """

    def __init__(self, grid_size: tuple = (10, 10), start: tuple = None, end: tuple = None, length: int = None):
        """Initialize Generator.
        Args:
            grid_size (tuple): size of map. In future maybe expand automatic. We just consider this as max size for now.
                                We add a false border around so the solver know where is the limit.
        """

        self.solver = Solver(name="z3")  # init solver
        self.runs = 0  # save the number of solutions returned
        self.start = start
        self.end = end
        self.problem = []  # here we save the problem
        self.grid_size = (grid_size[0] + 2, grid_size[1] + 2)  # we save here the real grid size.
        self.max_length = grid_size[0] * grid_size[1]  # start is 0
        if length is None:
            # take maximum of grid if None is given. example: 2x2 = 4
            self.length = self.max_length
        else:
            if length > self.max_length:
                raise ValueError(
                    f'Maximum length is {self.max_length} for grid of size: {grid_size}, but {length} is given.')
            else:
                self.length = length

        # for now we have only street pieces, later we could add other
        self.pieces = {
            'STREET': 's',  # 0
        }
        # save the actual bool values in a numpy array for each piece. So we know where what to put.
        # -1 = empty, 0 = not use, 1 = use
        self.grids = {
            'STREET': np.full(self.grid_size, -1, dtype=int),
        }
        # here we save all the literals for the solver
        self.vars = {
            'STREET': np.empty((self.grid_size[0], self.grid_size[1], len(self.pieces), self.length + 1), dtype=object),
        }
        # here we will save the solution
        self.allsolutions = None
        self.solution = {
            'STREET': np.zeros((self.grid_size[0], self.grid_size[1]), dtype=int),
        }

    def _nformat(self, n: int, m: int, piece, step: int):
        # little helper to format the strings for the literal names.
        # format = piece:row:column:step
        # example = s:3:22:5  = piece:s, row:3, column:22, step:5
        return f'{piece}:{n}:{m}:{step}'

    def _extract_pos(self, node):
        name = node.symbol_name()
        name = name.split(':')
        return int(name[1]), int(name[2])

    def _extract_step(self, node):
        name = node.symbol_name()
        name = name.split(':')
        return int(name[3])

    def _literal_generator(self):
        # generate all literals for the solver and generator
        # for each piece we create a numpy array
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                for i, v in enumerate(self.pieces.values()):
                    for s in range(self.length + 1):
                        lit_name = self._nformat(r, c, v, s)
                        self.vars['STREET'][r, c, i, s] = Symbol(lit_name, BOOL)

    def _border_states(self):
        # set the border states where we never want to go
        for name, v in zip(self.grids.keys(), self.grids.values()):
            v[0, :] = 0
            v[-1, :] = 0
            v[1:-1, 0] = 0
            v[1:-1, -1] = 0

    def _set_states(self, arr: np.array, states: tuple):
        # set True, False states for the solver booleans
        # return the solver booleans as list
        for_solver = []
        for i, s in enumerate(states):
            if s == 1:
                for_solver.append(arr[i])
            else:
                for_solver.append(Not(arr[i]))
        return for_solver

    def _neighbors(self, arr: np.array, radius: int, pos: tuple):
        # get neighbours of a position with a given radius.
        # return them as dictionary with key=position and value = cell value
        # we not consider the corners since they are not reachable
        neighbours = {}
        row_number = pos[0]
        column_number = pos[1]
        for i in range(row_number - radius, row_number + radius + 1):
            for j in range(column_number - radius, column_number + radius + 1):
                if i >= 0 and i < len(arr) and j >= 0 and j < len(arr[0]):
                    neighbours[(i, j)] = arr[i][j]
        # remove the 4 corner pieces and middle piece
        neighbours.pop((row_number - radius, column_number - radius), None)
        neighbours.pop((row_number - radius, column_number + radius), None)
        neighbours.pop((row_number + radius, column_number - radius), None)
        neighbours.pop((row_number + radius, column_number + radius), None)
        neighbours.pop((pos[0], pos[1]), None)
        return neighbours

    def generate_problem(self):
        self._literal_generator() # create all literals
        self._border_states()  # set border to 0

        # restrict border and cells we are not allowed to go for each step
        not_free = np.where(self.grids['STREET'] == 0)
        for s in range(self.length + 1):
            self.problem.append(And([Not(x) for x in self.vars['STREET'][not_free][:, 0].flatten()]))

        # for each cell 0 there can be only 1 track piece from one state
        free = np.nonzero(self.grids['STREET'])
        leftside = self.vars['STREET'][free][:, :, 0].flatten()
        rightside = np.reshape(self.vars['STREET'][free][:, :, 1:], (len(leftside), -1))
        self.problem.append(And([Implies(l, ExactlyOne(r)) for l, r in zip(leftside, rightside)]))

        # for next step we can go any of 4 directions from any of the random selected start cell. up,down,left,right
        nodes_0 = self.vars['STREET'][free][:, :, 0].flatten()
        for i in range(self.length):
            if i == 0:
                # select start
                nodes_l = self.vars['STREET'][free][:, :, i + 1].flatten()
                self.problem.append(ExactlyOne(nodes_l))
            else:
                # select direction
                nodes_l = self.vars['STREET'][free][:, :, i].flatten()
                to_add = []
                for n in nodes_l:
                    pos = self._extract_pos(n)
                    neighbours = self._neighbors(self.vars['STREET'], radius=1, pos=pos)
                    neighbours = np.asarray([x[:, i + 1] for x in neighbours.values()]).flatten()
                    to_add.append(Implies(n, ExactlyOne(neighbours)))
                self.problem.append(And(to_add))

            # reserve the 0 state
            nodes_l = self.vars['STREET'][free][:, :, i + 1].flatten()
            self.problem.append(And([Implies(l, r) for l, r in zip(nodes_l, nodes_0)]))

            # if we got one cell all others need to be false
            nodes_r = self.vars['STREET'][free][:, :, i + 1].flatten()
            to_add = []
            for n in nodes_l:
                r = nodes_r[np.isin(nodes_r, n) == False]
                to_add.append(Implies(n, And([Not(x) for x in r])))
            self.problem.append(And(to_add))

    def get_solution(self):
        """
        :return: grids with filled solution
        """
        self.allsolutions = self.get_random_sol(And(self.problem), method='reset', getall=False)
        self.runs += 1

        true_values = list(itertools.filterfalse(lambda x: x[1].constant_value() is False, self.allsolutions))
        #print(true_values)
        self.grids['STREET'][1:-1, 1:-1] = -1  # reset inner grid
        self.solution['STREET'][1:-1, 1:-1] = 0  # reset inner grid
        for l in true_values:
            pos = self._extract_pos(l[0])
            step = self._extract_step(l[0])
            if step == 0:
                self.grids['STREET'][pos] = 1
            else:
                self.solution['STREET'][pos] = step
        return self.solution['STREET'], self.grids['STREET']

    def get_random_sol(self, problem, method: str = 'reset', getall: bool = False, maxsol=20):
        """
        since SAT solver are too deterministic for small problems we need to generate all and sample one.
        :param method: 'reset' or 'add'. How to get random solution, reset solver with random seed or random sample from many solutions
        :param maxsol: If method 'add' then limit solutions. Otherwise, compute time is so long for big problems.
        :param getall: get all solutions as list
        :param problem: problem to solve
        :return: return random solution
        """
        # reset solver
        self.solver = Solver(name="z3", random_seed=np.random.randint(1000))  # init solver
        self.solver.add_assertion(problem)
        all_solutions = []
        if method == 'reset':
            if self.solver.solve():
                solution = list(self.solver.get_model())  # this return a list of tuples like (name,symbol)
                all_solutions.append(solution)
        if method == 'add':
            while self.solver.solve() and len(all_solutions) <= maxsol:
                solution = list(self.solver.get_model())  # this return a list of tuples like (name,symbol)
                all_solutions.append(solution)
                # check for which of the states we get a true value
                to_add = []
                for var in solution:
                    # need to get the opposite states for the Or clause
                    if self.solver.get_py_value(var[0]):
                        to_add.append(Not(var[0]))
                    else:
                        to_add.append(var[0])
                self.solver.add_assertion(Or(to_add))
        # return solution based on given parameters
        if len(all_solutions) < 1:
            raise RuntimeError('Problem is UNSAT!')
        else:
            if getall:
                return all_solutions
            else:
                return all_solutions[np.random.randint(len(all_solutions))]


if __name__ == "__main__":
    gen = Generator_grid(grid_size=(2, 2), length=2)

    t = timeit(lambda: gen.generate_problem(), number=1)
    print(f'time to generate problem: {t}')
    t = timeit(lambda: gen.get_solution(), number=1)
    print(f'time to get solution: {t}')

    #gen.generate_problem()
    #atoms = get_atoms(And(gen.problem))
    #print(f'Atoms: {len(atoms)}')

    sol, grid = gen.get_solution()
    pprint.pp(sol)
    sol, grid = gen.get_solution()
    pprint.pp(sol)
    sol, grid = gen.get_solution()
    pprint.pp(sol)
