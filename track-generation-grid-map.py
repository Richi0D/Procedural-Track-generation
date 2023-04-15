import itertools
import numpy as np
from pysmt.shortcuts import Solver, Symbol, And, Or, Not, Iff, Implies
from pysmt.typing import BOOL
from scipy.sparse import random as sparse
import pprint


class Generator_grid:
    """Track Generator using SAT solver
       It uses 3 track pieces to generate a random track on a grid. (straight, left, right)
    """

    def __init__(self, grid_size: tuple = (10, 10), start: tuple=None, end: tuple=None):
        """Initialize Generator.
        Args:
            grid_size (tuple): size of map. In future maybe expand automatic. We just consider this as max size for now.
                                We add a false border around so the solver know where is the limit.
        """

        self.solver = Solver(name="z3")  # init solver
        self.runs = 0  # save the number of solutions returned
        self.problem = []
        self.grid_size = (grid_size[0] + 2, grid_size[1] + 2)  # we save here the real grid size.

        # for now we have only street pieces, later we could add other
        self.pieces = {
            'STREET': 's',  # 0
            'UP': 'u',      # 1
            'DOWN': 'd',    # 2
            'LEFT': 'l',    # 3
            'RIGHT': 'r',   # 4
            'BUILDING': 'b' # 5
        }
        # save the actual bool values in a numpy array for each piece. So we know where what to put.
        # -1 = empty, 0 = not use, 1 = use
        self.grids = {
            'STREET': np.full(self.grid_size, -1),
            'BUILDING': np.full(self.grid_size, -1)
        }
        # here we save all the literals for the solver
        self.vars = {
            'STREET': np.empty((self.grid_size[0], self.grid_size[1], len(self.pieces)-1), dtype=object),
            'BUILDING': np.empty((self.grid_size[0], self.grid_size[1]), dtype=object),
        }
        # here we will save the solution
        self.allsolutions = None
        self.solution = {
            'STREET': np.empty((self.grid_size[0], self.grid_size[1], len(self.pieces)-1), dtype=bool),
            'BUILDING': np.empty((self.grid_size[0], self.grid_size[1]), dtype=bool),
        }
        # create all literals
        self._literal_generator()

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
                    if v == 'b':
                        lit_name = self._nformat(r, c, v)
                        self.vars['BUILDING'][r, c] = Symbol(lit_name, BOOL)
                    else:
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

    def generate_problem(self):
        # add some random buildings
        # TODO: how to do with SAT?
        # random_choice = np.nonzero(np.random.choice(a=[False, True], size=(self.grid_size[0], self.grid_size[1])))
        random_choice = sparse(self.grid_size[0], self.grid_size[1], density=0.2).A
        self.grids['BUILDING'][random_choice > 0] = 1

        # set border to 0
        self._border_states()

        # restrict border and cells we are not allowed to go
        not_free = np.where(np.logical_or(self.grids['STREET'] == 0, self.grids['BUILDING'] == 1))
        self.problem.append(And([Not(x) for x in self.vars['STREET'][not_free][:, 0]]))
        not_free = np.where(self.grids['BUILDING'] == 0)
        self.problem.append(And([Not(x) for x in self.vars['BUILDING'][not_free]]))


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

        # 4|3|2|1|0 connections possible for streets
        combos_4 = [(1, 1, 1, 1)]
        combos_3 = [(1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)]
        combos_2 = [(1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 1, 1)]
        combos_1 = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
        combos_0 = [(0, 0, 0, 0)]
        # to_add = []
        # for n in nodes:
        #     states = [And(self._set_states(n[1:], combo)) for combo in combos_4]
        #     states.extend([And(self._set_states(n[1:], combo)) for combo in combos_3])
        #     states.extend([And(self._set_states(n[1:], combo)) for combo in combos_2])
        #     states.extend([And(self._set_states(n[1:], combo)) for combo in combos_1])
        #     states.extend([And(self._set_states(n[1:], combo)) for combo in combos_0])
        #     to_add.append(Or(states))
        # self.problem.append(And(to_add))

        # Building or Street but not at the same time.
        to_add = []
        free = np.nonzero(self.grids['STREET'])  # take care here. could be some streets are already 0
        nodes_b = self.vars['BUILDING'][free]
        nodes_s = self.vars['STREET'][free][:, 0]
        for b, s in zip(nodes_b, nodes_s):
            to_add.append(Iff(b, Not(s)))
        self.problem.append(And(to_add))

        # to each building we need minimum 1 street as neighbour.
        to_add = []
        for b in nodes_b:
            states = []
            pos = self._extract_pos(b)
            neighbours = [x for x in (self._neighbors(self.vars['STREET'][:, :, 0], radius=1, pos=pos).values())]
            states.extend([And(self._set_states(neighbours, combo)) for combo in combos_4])
            states.extend([And(self._set_states(neighbours, combo)) for combo in combos_3])
            states.extend([And(self._set_states(neighbours, combo)) for combo in combos_2])
            states.extend([And(self._set_states(neighbours, combo)) for combo in combos_1])
            to_add.append(Implies(b, Or(states)))
        self.problem.append(And(to_add))


    def get_solution(self):
        """
        :return: grids with filled solution
        """
        self.allsolutions = self.get_random_sol(And(self.problem), method='reset', getall=False)
        self.runs += 1

        true_values = list(itertools.filterfalse(lambda x: x[1].constant_value() is False, self.allsolutions))
        true_nodes = [x[0] for x in true_values]
        self.solution['STREET'] = np.isin(self.vars['STREET'], true_nodes)
        self.solution['BUILDING'] = np.isin(self.vars['BUILDING'], true_nodes)
        self.grids['STREET'][1:-1, 1:-1] = -1 # reset inner grid
        self.grids['BUILDING'][1:-1, 1:-1] = -1  # reset inner grid
        for l in true_values:
            typ = l[0].symbol_name()[0]
            pos = self._extract_pos(l[0])
            if typ == 'b':
                self.grids['BUILDING'][pos] = 1
            else:
                self.grids['STREET'][pos] = 1
        return self.solution, self.grids

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
        self.solver = Solver(name="z3", random_seed=np.random.randint(100))  # init solver
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
            print(self.grids['BUILDING'])
            raise RuntimeError('Problem is UNSAT!')
        else:
            if getall:
                return all_solutions
            else:
                return all_solutions[np.random.randint(len(all_solutions))]


if __name__ == "__main__":
    gen = Generator_grid(grid_size=(3, 3))
    gen.generate_problem()
    sol, grid = gen.get_solution()
    pprint.pp(grid)
    sol, grid = gen.get_solution()
    pprint.pp(grid)
    sol, grid = gen.get_solution()
    pprint.pp(grid)
    # print(gen.solver.z3.statistics())
