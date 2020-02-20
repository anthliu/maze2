import argparse
import numpy as np

import maze
from pseudo_policy import PseudoPolicy

def main(args):
    with open(args.map) as f:
        grid_s = f.read()
    env = maze.MazeEnv(grid_s, max_t=args.max_t, render_mode=args.render_mode)
    test_ob = env.reset()
    
    obs = np.zeros((args.n_episodes, args.max_t + 1) + test_ob.shape)
    rewards = np.zeros((args.n_episodes, args.max_t + 1))
    actions = np.zeros((args.n_episodes, args.max_t + 1))
    agent_locs = np.zeros((args.n_episodes, args.max_t + 1, 2))
    key_locs = np.zeros((args.n_episodes, args.max_t + 1, 2))
    ghost_locs = np.zeros((args.n_episodes, args.max_t + 1, 2))
    seq_lengths = np.zeros(args.n_episodes)

    successes = 0
    key_pickups = 0
    key_fail = 0
    for ep_idx in range(args.n_episodes):
        done = False
        success = False
        key_pickup = False
        t = 0
        while True:
            obs[ep_idx, t] = ob = env.reset()
            solution = env.maze.sketch_solution()
            if solution is not None and len(solution) >= 10:
                break
        policy = PseudoPolicy(solution)
        agent_locs[ep_idx, t] = env.maze.latent[maze.PLAYER]
        key_locs[ep_idx, t] = env.maze.latent[maze.KEY]
        ghost_locs[ep_idx, t] = env.maze.latent[maze.GHOST]
        t += 1

        while not done:
            action = policy(ob)
            ob, r, done, info = env.step(action)
            if r > 0:
                success = True

            obs[ep_idx, t] = ob
            rewards[ep_idx, t] = r
            agent_locs[ep_idx, t] = info[maze.PLAYER] if info[maze.PLAYER] is not None else (0, 0)
            key_locs[ep_idx, t] = info[maze.KEY] if info[maze.KEY] is not None else (0, 0)
            ghost_locs[ep_idx, t] = info[maze.GHOST] if info[maze.KEY] is not None else (0, 0)

            if info[maze.KEY] is None:
                key_pickup = True

            t += 1

        successes += int(success)
        key_pickups += int(key_pickup)
        key_fail += int(key_pickup and (not success))
        seq_lengths[ep_idx] = t

    print(f'success rate: {successes}/{args.n_episodes}')
    print(f'key pickups: {key_pickups}')
    print(f'key and fail: {key_fail}')

    np.savez(
        args.out,
        obs=obs,
        actions=actions,
        rewards=rewards,
        agent_locs=agent_locs,
        key_locs=key_locs,
        ghost_locs=ghost_locs,
        seq_lengths=seq_lengths
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate trajectories',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-n', '--n_episodes', type=int, default=5000, help='number of episodes to generate'
    )
    parser.add_argument(
        '-o', '--out', default='trajectories.npz', help='outfile'
    )
    parser.add_argument(
        '--render_mode', default='grid', help='grid | rgb'
    )
    parser.add_argument(
        '--map', default='conf/4doors.txt', help='map text file'
    )
    parser.add_argument(
        '--max_t', type=int, default=50, help='max length of episode'
    )

    args = parser.parse_args()
    main(args)
