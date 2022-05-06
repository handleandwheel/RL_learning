import gym
import numpy as np

def get_policy(env: gym.Env, value_table, gamma):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        reward = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, next_state, reward_, _ in env.P[state][action]:
                reward[action] += prob * (reward_ + gamma * value_table[next_state])
        policy[state] = np.argmax(reward)
    return policy

def policy_iter(env: gym.Env, gamma=1.0, iter=2000):
    policy_table = np.floor(np.random.random(env.observation_space.n)*3.99)
    threshold = 1e-5
    value_table = np.zeros(env.observation_space.n)
    for i in range(iter):
        old_value = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = policy_table[state]
            reward = 0.0
            for prob, next_state, reward_, _ in env.P[state][action]:
                reward += prob * (reward_ + gamma * value_table[next_state])
            value_table[state] = reward
        policy_table = get_policy(env, value_table, gamma)
        if np.sum(np.fabs(value_table - old_value)) <= threshold:
            break
    return policy_table

def test_policy(env: gym.Env, policy, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    done = False
    while not done:
        if render:
            env.render()
        action = int(policy[obs])
        obs, _, done, _ = env.step(action)

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env.reset()
    policy = policy_iter(env)
    print(policy)
    test_policy(env, policy, True)
    