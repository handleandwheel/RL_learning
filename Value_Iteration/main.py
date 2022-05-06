import gym
import numpy as np


def value_iter(env: gym.Env, gamma=1.0, iter=2000):
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-5
    for i in range(iter):
        old_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            reward = [np.sum([((reward_ + gamma * old_table[n_s]) * prob) \
                for prob, n_s, reward_, _ in env.P[state][action]]) \
                    for action in range(env.action_space.n)]
            value_table[state] = max(reward)
        if np.sum(np.fabs(value_table - old_table)) <= threshold:
            break
    return value_table

def test_table(env: gym.Env, value_table, gamma=1.0, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    done = False
    while not done:
        if render:
            env.render()
        action = int(policy[obs])
        obs, _, done, _ = env.step(action)
        
def get_policy(env: gym.Env, value_table, gamma):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        reward = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, next_state, reward_, _ in env.P[state][action]:
                reward[action] += prob * (reward_ + gamma * value_table[next_state])
        policy[state] = np.argmax(reward)
    return policy

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env.reset()
    value_table = value_iter(env)
    print(value_table)
    policy = get_policy(env, value_table, 1.0)
    print(policy)
    test_table(env, value_table, 1.0, True)



