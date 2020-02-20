from copy import deepcopy
import numpy as np
import maze

UP = 1
DOWN = 3
RIGHT = 0
LEFT = 2
NOOP = 4

LOOPS = [
    [NOOP] * 2,
    [NOOP] * 3,
    2 * [LEFT] + 2 * [RIGHT],
    2 * [RIGHT] + 2 * [LEFT],
    2 * [UP] + 2 * [DOWN],
    2 * [DOWN] + 2 * [UP],
    [UP, DOWN],
    [DOWN, UP],
    [RIGHT, LEFT],
    [LEFT, RIGHT],
    [RIGHT, UP, LEFT, DOWN],
    [UP, RIGHT, DOWN, LEFT],
    [RIGHT, DOWN, LEFT, UP],
    [DOWN, RIGHT, UP, LEFT],
    [DOWN, LEFT, UP, RIGHT],
    [LEFT, DOWN, RIGHT, UP],
    [LEFT, UP, RIGHT, DOWN],
    [UP, LEFT, DOWN, RIGHT]
]
N_LOOPS = len(LOOPS)

class PseudoPolicy(object):
    def __init__(self, solution):
        self.transition_probs = [0.7, 0.2, 0.1]
        self.state = 0
        self.action_queue = []
        self.solution = list(solution)

    def __call__(self, obs):
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)
        self.state = np.random.choice(3, p=self.transition_probs)
        n = np.random.randint(4) + 1
        if self.state == 0:
            if len(self.solution) > 0:
                self.action_queue, self.solution = self.solution[:n], self.solution[n:]
            else:
                self.action_queue = [np.random.randint(maze.ACTIONS) for _ in range(n)]
        elif self.state == 1:
            idx = np.random.randint(N_LOOPS)
            self.action_queue = deepcopy(LOOPS[idx])
        else:
            self.action_queue = [np.random.randint(maze.ACTIONS) for _ in range(n)]

        return self.action_queue.pop(0)
