import itertools
import numpy as np
from pysmt.shortcuts import Solver, Symbol, And, Or, Not, Implies, Iff, ExactlyOne, get_atoms, Equals, NotEquals, LT, GT, LE, GE, Int, Plus
from pysmt.typing import BOOL, INT
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
            'STREET': np.empty((self.grid_size[0], self.grid_size[1], len(self.pieces)), dtype=object),
        }
        # here we will save the solution
        self.allsolutions = None
        self.solution = {
            'STREET': np.zeros((self.grid_size[0], self.grid_size[1]), dtype=int),
        }

    def _nformat(self, n: int, m: int, piece,):
        # little helper to format the strings for the literal names.
        # format = piece:row:column:step
        # example = s:3:22:5  = piece:s, row:3, column:22, step:5
        return f'{piece}:{n}:{m}'

    def _extract_pos(self, node):
        name = node.symbol_name()
        name = name.split(':')
        return int(name[1]), int(name[2])

    def _literal_generator(self):
        # generate all literals for the solver and generator
        # for each piece we create a numpy array
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                for i, v in enumerate(self.pieces.values()):
                    lit_name = self._nformat(r, c, v)
                    self.vars['STREET'][r, c, i] = Symbol(lit_name, INT)

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
        self.problem = []
        self._literal_generator() # create all literals
        self._border_states()  # set border to 0

        # restrict border and cells we are not allowed to go for each step
        not_free = np.where(self.grids['STREET'] == 0)
        self.problem.append(And([Equals(x, Int(0)) for x in self.vars['STREET'][not_free][:, 0].flatten()]))

        # for each cell limit track length or 0 for empty
        free = np.nonzero(self.grids['STREET'])
        atoms = self.vars['STREET'][free][:, 0].flatten()
        # less than tracklength
        self.problem.append(And([LE(l, Int(self.length)) for l in atoms]))
        # greater 0
        self.problem.append(And([GE(l, Int(0)) for l in atoms]))

        # additional information: force to get a track: exactly one k for each node
        for i in range(1, self.length+1):
            self.problem.append(ExactlyOne([Equals(l, Int(i)) for l in atoms]))


        # one neighbour need to be +1 and one need to be -1
        to_add = []
        for n in atoms:
            pos = self._extract_pos(n)
            neighbours = self._neighbors(self.vars['STREET'], radius=1, pos=pos)
            neighbours = np.asarray([x[:] for x in neighbours.values()]).flatten()
            #eq = [Equals(x, Plus(n, Int(1))) for x in neighbours]
            #eq.append(Equals(n, Int(0)))
            #self.problem.append(Or(eq))
            eq = [Equals(x, Plus(n, Int(-1))) for x in neighbours]
            eq.append(Equals(n, Int(0)))
            to_add.append(Or(eq))
        self.problem.append(And(to_add))

    def get_solution(self):
        """
        :return: grids with filled solution
        """
        self.allsolutions = self.get_random_sol(And(self.problem), method='reset', getall=False)
        self.runs += 1
        #print(self.allsolutions)
        self.grids['STREET'][1:-1, 1:-1] = -1  # reset inner grid
        self.solution['STREET'][1:-1, 1:-1] = 0  # reset inner grid
        for l in self.allsolutions:
            pos = self._extract_pos(l[0])
            sol = l[1].constant_value()
            self.solution['STREET'][pos] = sol
        return self.solution['STREET']

    def get_random_sol(self, problem, method: str = 'reset', getall: bool = False, maxsol=20, seed=np.random.randint(100)):
        """
        since SAT solver are too deterministic for small problems we need to generate all and sample one.
        :param method: 'reset' or 'add'. How to get random solution, reset solver with random seed or random sample from many solutions
        :param maxsol: If method 'add' then limit solutions. Otherwise, compute time is so long for big problems.
        :param getall: get all solutions as list
        :param problem: problem to solve
        :return: return random solution
        """
        # reset solver
        self.solver = Solver(name="msat", random_seed=seed)  # init solver
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
                    to_add.append(Equals(var[0], var[1]))
                self.solver.add_assertion(Not(And(to_add)))
        # return solution based on given parameters
        if len(all_solutions) < 1:
            raise RuntimeError('Problem is UNSAT!')
        else:
            if getall:
                return all_solutions
            else:
                return all_solutions[np.random.randint(len(all_solutions))]


if __name__ == "__main__":
    gen = Generator_grid(grid_size=(2,2), length=4)
    gen.generate_problem()

    gen.get_random_sol(And(gen.problem), method='add')

    gen.generate_problem()
    print(len(gen.problem))

    #gen.generate_problem()
    #atoms = get_atoms(And(gen.problem))
    #print(f'Atoms: {len(atoms)}')

    sol = gen.get_solution()
    pprint.pp(sol)
