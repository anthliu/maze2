import heapq
import numpy as np

N_ENTITIES = 8

EMPTY = 0
WALL = 1
PLAYER = 2
TREASURE = 3
KEY = 4
DOOR = 5
GHOST = 6
APPLE = 7

CHARS = {
    EMPTY: ' ',
    WALL: 'w',
    PLAYER: 'p',
    TREASURE: 't',
    KEY: 'k',
    DOOR: 'd',
    GHOST: 'g',
    APPLE: 'a',
}
COLORS = {
    EMPTY: [255, 255, 255],
    WALL: [44, 42, 60],
    PLAYER: [105, 105, 105],
    KEY: [135, 206, 250],
    DOOR: [152, 251, 152],
    TREASURE: [255, 255, 0],
    APPLE: [250, 128, 114],
    GHOST: [25,25,112],
}
CHAR_LOOKUP = {c: idx for idx, c in CHARS.items()}

TRACK = {PLAYER, TREASURE, KEY, DOOR, GHOST, APPLE}
ACTIONS = 5
DIRS = [(0, 1), (-1, 0), (0, -1), (1, 0), (0, 0)]
DUMMY_ACTION = 4

def string_to_carray(ss):
    return np.array([list(row) for row in ss.strip().split('\n')])

class Maze(object):
    def __init__(self, grid, latent=None):
        self.grid = grid
        if not latent:
            self.latent = self.track_grid(self.grid)
        else:
            self.latent = latent

    def render(self, mode='human'):
        if mode == 'human':
            result = '\n'.join(''.join(CHARS[c] for c in row) for row in self.grid)
        elif mode == 'grid':
            H, W = self.grid.shape
            result = np.zeros((H, W, N_ENTITIES))
            flat = result.reshape(H * W, -1)
            flat[np.arange(H * W), self.grid.reshape(-1).view(dtype=int)] = 1
        elif mode == 'rgb':
            H, W = self.grid.shape
            result = np.zeros((H, W, 3))
            for i in range(H):
                for j in range(W):
                    result[i, j] = COLORS[self.grid[i, j]]
        return result

    def _elem(self, pos):
        return self.grid[pos[0], pos[1]]

    def _set_elem(self, pos, elem):
        self.grid[pos[0], pos[1]] = elem

    def _move(self, pos, direction):
        return (pos[0] + DIRS[direction][0], pos[1] + DIRS[direction][1])

    def _legal_pos(self, pos, has_key=False):
        if has_key:
            return self._elem(pos) != WALL
        else:
            return self._elem(pos) not in [WALL, DOOR]

    def _legal_neighbors(self, pos, has_key=False):
        for d in range(ACTIONS):
            new_pos = self._move(pos, d)
            if self._legal_pos(new_pos, has_key):
                yield new_pos, d

    def sketch_solution(self):
        sol = self._sketch_solution(TREASURE)
        if sol is not None:
            return sol

        key_sol = self._sketch_solution(KEY)
        if key_sol is not None:
            final_sol = self._sketch_solution(TREASURE, start_pos=self.latent[KEY], has_key=True)
            if final_sol is not None:
                return key_sol + final_sol

        return None

    def _sketch_solution(self, target, start_pos=None, has_key=False):
        if start_pos is None:
            pos = self.latent[PLAYER]
        else:
            pos = start_pos
        target_pos = self.latent[target]
        visited = {pos}

        def heuristic(new_p):
            return (target_pos[0] - new_p[0])**2 + (target_pos[1] - new_p[1])**2
        def neighbors(cur_state):
            _, (pos, history) = cur_state
            for new_pos, d in self._legal_neighbors(pos, has_key):
                yield heuristic(new_pos), (new_pos, history + [d])

        candidates = list(neighbors((None, (pos, []))))
        heapq.heapify(candidates)
        while len(candidates) > 0:
            state = heapq.heappop(candidates)
            for candidate in neighbors(state):
                _, (new_pos, history) = candidate
                if new_pos in visited:
                    continue
                if new_pos == target_pos:
                    return history

                visited.add(new_pos)
                heapq.heappush(candidates, candidate)

        return None

    @staticmethod
    def track_grid(grid):
        state = {elem: None for elem in TRACK}
        for i, row in enumerate(grid):
            for j, elem in enumerate(row):
                if elem in TRACK:
                    assert state[elem] is None
                    state[elem] = (i, j)
        return state

    @staticmethod
    def from_string(ss):
        c_grid = string_to_carray(ss)
        return Maze.from_carray(c_grid)

    @staticmethod
    def from_carray(c_grid):
        grid = np.zeros(c_grid.shape, dtype=np.int)
        special_map = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if c_grid[i, j].isdigit():
                    grid[i, j] = int(c_grid[i, j])
                else:
                    special_map.setdefault(c_grid[i, j], []).append((i, j))

        latent = {}
        for c, locations in special_map.items():
            loc = locations[np.random.randint(len(locations))]
            grid[loc[0], loc[1]] = CHAR_LOOKUP[c]
            latent[CHAR_LOOKUP[c]] = loc

        return Maze(grid, latent=latent)

class MazeEnv(object):
    def __init__(self, s_grid, max_t=50, render_mode='grid', ghost_movement='random'):
        self.c_grid = string_to_carray(s_grid)
        self.max_t = max_t
        self.render_mode = render_mode
        self.ghost_movement = ghost_movement
        assert self.ghost_movement in ['random', 'sway']
        self.ghost_state = 0
        self.terminated = True

    def render(self):
        return self.maze.render(mode=self.render_mode)

    def reset(self):
        self.t = 0
        self.episode_reward = 0
        self.has_key = False
        self.terminated = False

        self.maze = Maze.from_carray(self.c_grid)
        return self.render()

    def step(self, d):
        assert 0 <= d < ACTIONS, f'invalid action: {d}'
        assert not self.terminated, 'episode is terminated'

        cur_pos = self.maze.latent[PLAYER]
        candidate_pos = self.maze._move(cur_pos, d)
        new_elem = self.maze._elem(candidate_pos)

        reward = 0
        ghost_swap = False# edge case when collide with ghost
        if new_elem == WALL:
            pos = cur_pos
        elif new_elem == EMPTY:
            pos = candidate_pos
        elif new_elem == PLAYER:
            pos = cur_pos
        elif new_elem == TREASURE:
            reward = 4
            self.maze.latent[TREASURE] = None
            pos = candidate_pos
            self.terminated = True
        elif new_elem == KEY:
            self.maze.latent[KEY] = None
            pos = candidate_pos
            self.has_key = True
        elif new_elem == DOOR:
            if self.has_key:
                self.maze.latent[DOOR] = None
                pos = candidate_pos
            else:
                pos = cur_pos
        elif new_elem == GHOST:
            ghost_swap = True
            self.maze.latent[GHOST] = cur_pos
            pos = candidate_pos
        elif new_elem == APPLE:
            reward = 1
            self.maze.latent[APPLE] = None
            pos = candidate_pos

        self.maze.latent[PLAYER] = pos
        self.maze._set_elem(cur_pos, EMPTY)
        self.maze._set_elem(pos, PLAYER)

        # ghost
        if ghost_swap:
            self.maze._set_elem(cur_pos, GHOST)
        else:
            ghost_pos = self.maze.latent[GHOST]
            if ghost_pos is not None:
                if self.ghost_movement == 'random':
                    rand_d = np.random.randint(ACTIONS)
                else:
                    self.ghost_state = (self.ghost_state + 1) % 10
                    rand_d = 0 if self.ghost_state <= 4 else 2
                new_ghost_pos = self.maze._move(ghost_pos, rand_d)
                if self.maze._elem(new_ghost_pos) == EMPTY:
                    self.maze.latent[GHOST] = new_ghost_pos
                    self.maze._set_elem(ghost_pos, EMPTY)
                    self.maze._set_elem(new_ghost_pos, GHOST)

        self.episode_reward += reward

        self.t += 1
        if self.t >= self.max_t:
            self.terminated = True

        return self.render(), reward, self.terminated, dict(self.maze.latent)

def TEST(s):
    with open(s) as f:
        ss = f.read()
    maze = Maze.from_string(ss)
    print('TEST MAZE')
    print(maze.render())
    print(maze.render(mode='grid').argmax(2))
    print(maze.latent)
    print(maze.sketch_solution())
    print('TEST MAZE ENV 1')
    solution = None
    while solution is None:
        env = MazeEnv(ss, render_mode='human')
        env.reset()
        solution = env.maze.sketch_solution()
        if solution is not None and len(solution) < 15:
            solution = None

    done = False
    while not done:
        #action = np.random.randint(ACTIONS)
        action = solution.pop(0)
        ob, r, done, info = env.step(action)
        print(ob)

    print('TEST MAZE ENV 2')
    env = MazeEnv(ss, render_mode='human')
    print(env.reset())
    done = False
    while not done:
        #action = np.random.randint(ACTIONS)
        action = int(input('>> '))
        ob, r, done, info = env.step(action)
        print(ob)
        print(info)

if __name__ == '__main__':
    TEST('conf/4doors.txt')
