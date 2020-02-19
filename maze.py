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
    EMPTY: 'o',
    WALL: 'w',
    PLAYER: 'p',
    TREASURE: 't',
    KEY: 'k',
    DOOR: 'd',
    GHOST: 'g',
    APPLE: 'a',
}
CHAR_LOOKUP = {c: idx for idx, c in CHARS.items()}

TRACK = {PLAYER, TREASURE, KEY, DOOR, GHOST, APPLE}
ACTIONS = 5
DIRS = [(0, 1), (-1, 0), (0, -1), (1, 0), (0, 0)]

def string_to_carray(ss):
    return np.array([list(row) for row in ss.strip().split('\n')])

class Maze(object):
    def __init__(self, grid, latent=None):
        self.grid = grid
        if not latent:
            self.latent = self.track_grid(self.grid)
        else:
            self.latent = latent

    def _elem(self, pos):
        return self.grid[pos[0], pos[1]]

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
        return self._sketch_solution(TREASURE)

    def _sketch_solution(self, target, has_key=False):
        pos = self.latent[PLAYER]
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
        grid = np.zeros(c_grid.shape)
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

def TEST(s):
    with open(s) as f:
        ss = f.read()
    maze = Maze.from_string(ss)
    print(maze.grid)
    print(maze.latent)
    print(maze.sketch_solution())


if __name__ == '__main__':
    TEST('conf/4doors.txt')
