import itertools
import numpy as np
from pysmt.shortcuts import Solver, Symbol, And, Or, Not, Iff
from pysmt.typing import BOOL
import pprint


class Generator_grid:
    """Track Generator using SAT solver
       It uses 3 track pieces to generate a random track on a grid. (straight, left, right)
    """

    def __init__(self, grid_size: tuple = (10, 10), start: tuple = None, end: tuple = None):
        """Initialize Generator.
        Args:
            grid_size (tuple): size of map. In future maybe expand automatic. We just consider this as max size for now.
                                We add a false border around so the solver know where is the limit.
        """

        self.solver = Solver(name="z3")  # init solver
        self.runs = 0  # save the number of solutions returned
        self.start = start
        self.end = end
        self.problem = []
        self.grid_size = (grid_size[0] + 2, grid_size[1] + 2)  # we save here the real grid size.

        # for now we have only street pieces, later we could add other
        self.pieces = {
            'STREET': 's',  # 0
            'UP': 'u',  # 1
            'DOWN': 'd',  # 2
            'LEFT': 'l',  # 3
            'RIGHT': 'r'  # 4
        }
        # save the actual bool values in a numpy array for each piece. So we know where what to put.
        # -1 = empty, 0 = not use, 1 = use
        self.grids = {
            'STREET': np.full(self.grid_size, -1),
        }
        # here we save all the literals for the solver
        self.vars = {
            'STREET': np.empty((self.grid_size[0], self.grid_size[1], len(self.pieces)), dtype=object),
        }
        # here we will save the solution
        self.allsolutions = None
        self.solution = {
            'STREET': np.empty((self.grid_size[0], self.grid_size[1], len(self.pieces)), dtype=bool),
        }
        # create all literals
        self._literal_generator()

    def set_point(self):
        # get random start position from solver.
        free = np.where(self.grids['STREET'] == -1)
        literals = self.vars['STREET'][free][:, 0]
        to_add = []
        for i in range(len(literals)):
            clause = []
            for k, l in enumerate(literals):
                if i == k:
                    clause.append(l)
                else:
                    clause.append(Not(l))
            to_add.append(And(clause))
        problem = Or(to_add)
        solution = self.get_random_sol(problem, method='add', getall=False)

        # check for which of the states we get a true value
        true_values = list(itertools.filterfalse(lambda x: x[1].constant_value() is False, solution))
        if len(true_values) == 1:
            point = self._extract_pos(true_values[0][0])
            self.grids['STREET'][point] = 1
            return point
        else:
            raise RuntimeError('Model does not return single solution!')

    def _nformat(self, n: int, m: int, piece):
        # little helper to format the strings for the literal names.
        # format = piece:row:column
        # example = s:3:22  = piece:s, row:3, column:22
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
                    self.vars['STREET'][r, c, i] = Symbol(lit_name, BOOL)

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

    def generate_problem(self, roundtrack: bool = False):
        self.problem = []
        self._border_states()  # set border to 0
        if not roundtrack:
            if self.start is None:
                self.start = self.set_point()  # get a random startpoint
            if self.end is None:
                self.end = self.set_point()  # get a random endpoint

        # restrict border and cells we are not allowed to go
        not_free = np.where(self.grids['STREET'] == 0)
        self.problem.append(And([Not(x) for x in self.vars['STREET'][not_free][:, 0]]))

        # for each cell we need to do: (s11 <-> (u11 | d11 | l11 |  r11))
        to_add = []
        for r in self.vars['STREET']:
            for c in r:
                to_add.append(Iff(c[0], Or(c[1:])))
        self.problem.append(And(to_add))

        # for each inner cell which is nonzero restrict connections to other cells
        # example: (d01 <-> u11) & (r10 <-> l11) & (l12 <-> r11) & (u21 <-> d11)
        # index: 's'=0, 'u'=1, 'd'=2, 'l'=3, 'r'=4
        to_add = []
        free = np.nonzero(self.grids['STREET'])
        nodes = self.vars['STREET'][free][:, :]
        for n in nodes:
            pos = self._extract_pos(n[0])
            neighbours = self._neighbors(self.vars['STREET'], radius=1, pos=pos)
            up = (pos[0] - 1, pos[1])
            to_add.append(Iff(neighbours[up][2], n[1]))
            left = (pos[0], pos[1] - 1)
            to_add.append(Iff(neighbours[left][4], n[3]))
            right = (pos[0], pos[1] + 1)
            to_add.append(Iff(neighbours[right][3], n[4]))
            down = (pos[0] + 1, pos[1])
            to_add.append(Iff(neighbours[down][1], n[2]))
        self.problem.append(And(to_add))

        # if we want to generate a track, we need to limit each inner cell to a min/max of 2 connections or all false
        # except the start and end can have only 1.
        # example: ((u12 & l12 & !r12 & !d12) | (u12 & !l12 & r12 & !d12) | (u12 & !l12 & !r12 & d12) |
        #           (!u12 & l12 & r12 & !d12) | (!u12 & l12 & !r12 & d12) | (!u12 & !l12 & r12 & d12)|
        #           (!u12 & !l12 & !r12 & !d12))
        combos_2_connections = [(1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 1, 1)]
        combos_1_connections = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        combos_0_connections = [(0, 0, 0, 0)]
        to_add = []
        for n in nodes:
            pos = self._extract_pos(n[0])
            if not roundtrack:  # filter the start and end to set only to 1 connection
                if pos in [self.start, self.end]:
                    states = [And(self._set_states(n[1:], combo)) for combo in combos_1_connections]
                    # states.extend([And(self._set_states(n[1:], combo)) for combo in combos_0_connections])
                    to_add.append(Or(states))
                else:
                    states = [And(self._set_states(n[1:], combo)) for combo in combos_2_connections]
                    states.extend([And(self._set_states(n[1:], combo)) for combo in combos_0_connections])
                    to_add.append(Or(states))
            else:
                states = [And(self._set_states(n[1:], combo)) for combo in combos_2_connections]
                states.extend([And(self._set_states(n[1:], combo)) for combo in combos_0_connections])
                to_add.append(Or(states))
        self.problem.append(And(to_add))

    def get_solution(self):
        """
        :return: grids with filled solution
        """
        self.allsolutions = self.get_random_sol(And(self.problem), method='add', getall=False)
        self.runs += 1

        true_values = list(itertools.filterfalse(lambda x: x[1].constant_value() is False, self.allsolutions))
        true_nodes = [x[0] for x in true_values]
        self.solution['STREET'] = np.isin(self.vars['STREET'], true_nodes)
        self.grids['STREET'][1:-1, 1:-1] = -1  # reset inner grid
        for l in true_values:
            pos = self._extract_pos(l[0])
            self.grids['STREET'][pos] = 1
        #print(true_values)
        return self.solution['STREET'], self.grids['STREET']

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
        self.solver = Solver(name="z3", random_seed=seed)  # init solver
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
    gen = Generator_grid(grid_size=(2, 2))
    gen.generate_problem()
    print(len(gen.problem))
    gen.generate_problem()
    print(len(gen.problem))

    sol, grid = gen.get_solution()
    pprint.pp(grid)
    sol, grid = gen.get_solution()
    pprint.pp(grid)
    # print(gen.solver.z3.statistics())
