from z3 import *
import itertools


class Generator:
    """Track Generator using SAT solver
       It uses 3 track pieces to generate a random track. (straight, left, right)
    """

    def __init__(self, length: int = 10, max_l: int = 2, max_r: int = 2, max_s: int = 2):
        """Initialize Generator.
        Args:
            length (int): length of track to generate
            max_l (int): maximum times l element is chosen in a row
            max_l (int): maximum times r element is chosen in a row
            max_s (int): maximum times s element is chosen in a row
        """

        self.length = length
        self.strlenght = len(str(self.length))
        self.step = 0
        self.max_l = max_l
        self.max_r = max_r
        self.max_s = max_s  #TODO: implement
        self.solver = Solver()
        #self.solver.set_params(random_seed=2)
        #sat.random_seed = 55
        #smt.random_seed = 100
        self.runs = 0  # save the number of solutions returned

        # generate a dictionary with track pieces.
        self.pieces = {
            'START': 'b',
            'GOAL': 'g',
            'STRAIGHT': 's',
            'LEFT': 'l',
            'RIGHT': 'r'
        }
        self.slv_vars = {}  # Dictionary with solver variables
        self.problem = [] # List of clauses for the solver

    def _nformat(self,n:int):
        # return correct number with leading zeros
        return f'{n:0{self.strlenght}d}'

    def _literal_generator(self, pieces: dict, steps: int, slv_type: str = 'Bool'):
        # generate all literals for the solver and generator
        # some solver types: Int(), Real(), Bool(),
        # Example: s-step-0
        for i in range(steps):
            i_format = self._nformat(i)
            for p in pieces.values():
                self.slv_vars[f'{p}{i_format}'] = Bool(f'{p}{i_format}')

    def _set_state_by_type(self, literals: list, piece):
        # set True. False states for the Z3 booleans by a given piece
        # return the Z3 booleans as list
        states = []
        c = [1 if l[0] == piece else 0 for l in literals]
        for i, s in enumerate(literals):
            if c[i] == 1:
                states.append(self.slv_vars[s])
            else:
                states.append(Not(self.slv_vars[s]))
        return states

    def _set_state_by_step(self, literals: list, step: int):
        # set True. False states for the Z3 booleans by a given step
        # return the Z3 booleans as list
        states = []
        c = [1 if l[-self.strlenght:] == self._nformat(step) else 0 for l in literals]
        for i, s in enumerate(literals):
            if c[i] == 1:
                states.append(self.slv_vars[s])
            else:
                states.append(Not(self.slv_vars[s]))
        return states

    def _init_state(self):
        # add the initial state to the solver
        states = list(itertools.filterfalse(lambda x: x[-self.strlenght:] != self._nformat(0), self.slv_vars.keys()))
        states = self._set_state_by_type(states, 'b')
        return And(states)

    def _goal_state(self):
        # add goal state and limit goal state to last step
        to_add = []
        states = list(itertools.filterfalse(lambda x: x[0] != 'g', self.slv_vars.keys()))
        states = self._set_state_by_step(states, self.length - 1)
        to_add.append(And(states))
        states = list(itertools.filterfalse(lambda x: x[-self.strlenght:] != self._nformat(self.length - 1), self.slv_vars.keys()))
        states = self._set_state_by_type(states, 'g')
        to_add.append(And(states))
        return And(to_add)

    def _step_state(self, step):
        # add clauses for each step
        to_add = []
        # right side implications. same for all implications
        states = list(itertools.filterfalse(lambda x: x[-self.strlenght:] != self._nformat(step + 1), self.slv_vars.keys()))
        states_right_imp = Or(And(self._set_state_by_type(states, 's')),
                              And(self._set_state_by_type(states, 'l')),
                              And(self._set_state_by_type(states, 'r')),
                              And(self._set_state_by_type(states, 'g')),
                              )
        # start
        states = list(itertools.filterfalse(lambda x: x[-self.strlenght:] != self._nformat(step), self.slv_vars.keys()))
        states_left_imp = And(self._set_state_by_type(states, 'b'))
        to_add.append(Implies(states_left_imp, states_right_imp))
        # straight
        states_left_imp = And(self._set_state_by_type(states, 's'))
        to_add.append(Implies(states_left_imp, states_right_imp))
        # left
        states_left_imp = And(self._set_state_by_type(states, 'l'))
        to_add.append(Implies(states_left_imp, states_right_imp))
        # right
        states_left_imp = And(self._set_state_by_type(states, 'r'))
        to_add.append(Implies(states_left_imp, states_right_imp))
        return And(to_add)

    def _max_curves(self, step):
        # limit left and right curves to maximum. Usually after 2 left or right pieces another piece must follow.
        to_add = []
        if step > self.max_l:
            states = list(itertools.filterfalse(lambda x: (x[-self.strlenght:] >= self._nformat(step) or
                                                           x[-self.strlenght:] < self._nformat(step-2) or
                                                           x[0] != 'l'), self.slv_vars.keys()))
            states = [self.slv_vars[x] for x in states]
            to_add.append(Implies(And(states), Not(self.slv_vars[f'l{self._nformat(step)}'])))
        if step > self.max_r:
            states = list(itertools.filterfalse(lambda x: (x[-self.strlenght:] >= self._nformat(step) or
                                                           x[-self.strlenght:] < self._nformat(step-2) or
                                                           x[0] != 'r'), self.slv_vars.keys()))
            states = [self.slv_vars[x] for x in states]
            to_add.append(Implies(And(states), Not(self.slv_vars[f'r{self._nformat(step)}'])))
        return And(to_add)

    def generate_problem(self):
        # generate all variables we need
        self._literal_generator(self.pieces, self.length)
        # add init_state
        self.problem.append(self._init_state())
        # add goal states
        self.problem.append(self._goal_state())
        # add step states
        for i in range(self.length - 1):
            self.problem.append(self._step_state(i))
            self.problem.append(self._max_curves(i))

    def get_true_literals(self, method:str='reset'):
        """
        :param method: 'add' or 'reset' to get a new solution
        :return: a list of true literals. Should be already sorted from the dict
        """
        if self.runs == 0:
            # on first run just add the problem
            self.solver.add(self.problem)
        else:
            if method == 'reset':
                # reset the solver, and add again the problem. I do not know why but this gives a new solution
                self.solver.reset()
                self.solver.add(self.problem)
            if method == 'add':
                # add the solution from the previous model so we get a new solution
                to_add = []
                for n, l in zip(self.slv_vars.keys(), self.slv_vars.values()):
                    # need to get the opposite states for the Or clause
                    if is_true(self.solution[l]):
                        to_add.append(Not(l))
                    else:
                        to_add.append(l)
                self.solver.add(Or(to_add))

        if self.solver.check() == sat:
            self.solution = self.solver.model()
            self.runs += 1
        else:
            raise RuntimeError('Model is unsatisfiable')

        literals = []
        for n, l in zip(self.slv_vars.keys(), self.slv_vars.values()):
            if is_true(self.solution[l]):
                literals.append(n)
        return literals


if __name__ == "__main__":
    gen = Generator(length=5)
    gen.generate_problem()
    #for i in range(26):
    print(gen.get_true_literals(method='add'))
    gen.generate_problem()
    print(gen.get_true_literals(method='add'))
    #print(gen.solver.statistics())
