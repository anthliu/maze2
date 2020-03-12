import argparse
import numpy as np

import maze
from pseudo_policy import PseudoPolicy

def main(args):
    with open(args.map) as f:
        grid_s = f.read()
    env = maze.MazeEnv(grid_s, max_t=args.max_t, render_mode=args.render_mode, ghost_movement=args.ghost_movement, generate_q_values=True)
    obs_shape = env.obs_shape

    assert obs_shape[0] == 13 and obs_shape[1] == 13
    encode_pos = lambda pos: pos[0] * 13 + pos[1] if pos is not None else 0
    
    obs = np.zeros((args.n_episodes, args.max_t + 1) + obs_shape)
    rewards = np.zeros((args.n_episodes, args.max_t + 1))
    actions = np.zeros((args.n_episodes, args.max_t + 1))
    infos = np.zeros((args.n_episodes, args.max_t + 1, 4))
    seq_lengths = np.zeros(args.n_episodes)

    successes = 0
    key_pickups = 0
    key_fail = 0
    for ep_idx in range(args.n_episodes):
        done = False
        success = False
        key_pickup = False
        t = 0
        obs[ep_idx, t] = ob = env.reset()
        solution = env.solution

        policy = PseudoPolicy(solution)
        infos[ep_idx, t, 0] = encode_pos(env.maze.latent[maze.PLAYER])
        infos[ep_idx, t, 1] = encode_pos(env.maze.latent[maze.KEY])
        infos[ep_idx, t, 2] = encode_pos(env.maze.latent[maze.GHOST])
        infos[ep_idx, t, 3] = env.q_values[env.get_id()]
        t += 1

        while not done:
            action = policy(ob)
            ob, r, done, info = env.step(action)
            if r > 0:
                success = True

            actions[ep_idx, t - 1] = action
            obs[ep_idx, t] = ob
            rewards[ep_idx, t] = r
            infos[ep_idx, t, 0] = encode_pos(env.maze.latent[maze.PLAYER])
            infos[ep_idx, t, 1] = encode_pos(env.maze.latent[maze.KEY])
            infos[ep_idx, t, 2] = encode_pos(env.maze.latent[maze.GHOST])
            infos[ep_idx, t, 3] = info['q']

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
        infos=infos,
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
        '--ghost_movement', default='random', help='random | sway'
    )
    parser.add_argument(
        '--map', default='conf/4doors.txt', help='map text file'
    )
    parser.add_argument(
        '--max_t', type=int, default=50, help='max length of episode'
    )

    args = parser.parse_args()
    main(args)
