import itertools
import numpy as np
from pysmt.shortcuts import Solver, Symbol, And, Or, Not
from pysmt.typing import BOOL
import pprint

class Generator_grid:
    """Track Generator using SAT solver
       It uses 3 track pieces to generate a random track on a grid. (straight, left, right)
    """

    def __init__(self, length: int = 10, grid_size: tuple = (10, 10)):
        """Initialize Generator.
        Args:
            length (int): length of track to generate
            grid_size (tuple): size of map. In future maybe expand automatic. We just consider this as max size for now.
                                We add a false border around so the solver know where is the limit.
        """

        self.length = length
        self.strlenght = len(str(max(grid_size)))
        self.steps = []  # save the steps as list
        self.solver = Solver(name="z3")  # init solver
        self.runs = 0  # save the number of solutions returned
        self.start = (int(grid_size[0] / 2), int(grid_size[1] / 2))
        self.grid_size = (grid_size[0] + 2, grid_size[1] + 2)  # we save here the real grid size.

        # for now we have only street pieces, later we could add other
        self.pieces = {
            'STREET': 's',
        }
        # save the actual bool values in a numpy array for each piece. So we know where what to put.
        # -1 = empty, 0 = not use, 1 = use
        self.grids = {
            'STREET': np.full(self.grid_size, -1),
        }
        # here we save all the literals for the solver
        self.vars = {
            'STREET': np.empty(self.grid_size, dtype=object),
        }

        self._literal_generator()
        self._border_states()  # set border to 0
        self.set_startpoint()  # get a random startpoint

    def set_startpoint(self):
        # get random start position from solver.
        free = np.where(self.grids['STREET'] == -1)
        literals = self.vars['STREET'][free]
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
        solution = self.get_random_sol(problem)

        # check for which of the states we get a true value
        true_values = list(itertools.filterfalse(lambda x: x[1].constant_value() is False, solution))
        if len(true_values) == 1:
            index = np.where(self.vars['STREET'] == true_values[0][0])
            self.start = (index[0][0], index[1][0])
            self.steps.append(self.start)
            self.grids['STREET'][self.steps[-1]] = 1
        else:
            raise RuntimeError('Model does not return single solution!')


    def _nformat(self, n: int, m: int, piece):
        # little helper to format the strings for the literal names.
        # format = piece:row:column
        # example = s:3:22  = piece:s, row:3, column:22
        # n = f'{n:0{self.strlenght}d}'
        # m = f'{m:0{self.strlenght}d}'
        return f'{piece}:{n}:{m}'

    def _literal_generator(self):
        # generate all literals for the solver and generator
        # for each piece we create a numpy array
        for name, v in zip(self.vars.keys(), self.vars.values()):
            for r in range(v.shape[0]):
                for c in range(v.shape[1]):
                    lit_name = self._nformat(r, c, self.pieces[name])
                    v[r, c] = Symbol(lit_name, BOOL)

    def _border_states(self):
        # set the border states where we never want to go
        for name, v in zip(self.grids.keys(), self.grids.values()):
            v[0, :] = 0
            v[-1, :] = 0
            v[1:-1, 0] = 0
            v[1:-1, -1] = 0

    def _set_states(self, arr: np.array, states: dict):
        # set True, False states for the solver booleans
        # return the solver booleans as list
        for_solver = []
        # we only set 0 or 1 states, ignore -1 since they are empty
        todo = [k for k, v in states.items() if v >= 0]
        for i in todo:
            if states[i] == 1:
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
        # remove the 4 corner pieces
        neighbours.pop((row_number - radius, column_number - radius), None)
        neighbours.pop((row_number - radius, column_number + radius), None)
        neighbours.pop((row_number + radius, column_number - radius), None)
        neighbours.pop((row_number + radius, column_number + radius), None)
        return neighbours

    def _iteration_move(self):
        # remove all assertions
        self.solver.reset_assertions()
        problem = []
        # right now we have only street pieces.
        states = self._neighbors(self.grids['STREET'], 1, self.steps[-1])

        # Already visited cells need to be 0. We do not want to go there again.
        for s in self.steps:
            if s in states.keys():
                states[s] = 0

        # add current position and neighbour info to solver
        problem.append(And(self._set_states(self.vars['STREET'], states)))

        # add the 4 possible directions but before remove the current step.
        states.pop(self.steps[-1], None)

        combos = []
        for i in range(len(states)):
            combo = np.zeros(len(states), dtype=int)
            combo[i] = 1
            combos.append(combo)
        to_add = []
        for c in combos:
            for k, v in zip(states.keys(), c):
                states[k] = v
            to_add.append(And(self._set_states(self.vars['STREET'], states)))
        problem.append(Or(to_add))
        solution = self.get_random_sol(And(problem))

        # TODO: What if we get stuck or can not move anymore?
        # check for which of the states we get a true value
        true_values = list(itertools.filterfalse(lambda x: x[1].constant_value() is False, solution))
        if len(true_values) == 1:
            index = np.where(self.vars['STREET'] == true_values[0][0])
            self.steps.append((index[0][0], index[1][0]))
            self.grids['STREET'][self.steps[-1]] = 1
        else:
            raise RuntimeError('Model does not return single solution!')

    def get_solution(self):
        """
        :return: array with solution
        """
        for i in range(self.length):
            self._iteration_move()
        return self.steps, self.grids

    def get_random_sol(self, problem):
        """
        since SAT solver are too deterministic for small problems we need to generate all and sample one.
        :param problem: problem to solve
        :return: return random solution
        """
        # remove all assertions
        self.solver.reset_assertions()
        self.solver.add_assertion(problem)
        all_solutions = []
        while self.solver.solve():
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
        return all_solutions[np.random.randint(len(all_solutions))]

    def reset(self):
        """
        Reset the generator to get a new solution with next solution call
        :return:
        """
        self.solver.reset_assertions()
        self.steps = []
        self.grids = {
            'STREET': np.full(self.grid_size, -1),
        }
        self._border_states()  # set border to 0
        self.set_startpoint()  # get a random startpoint


if __name__ == "__main__":
    gen = Generator_grid(length=5, grid_size=(2, 2))
    pprint.pp(gen.get_solution())
    gen.reset()
    pprint.pp(gen.get_solution())
    gen.reset()
    pprint.pp(gen.get_solution())
    #print(gen.solver.z3.statistics())
