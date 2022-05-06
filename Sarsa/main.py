import numpy as np
from cliffwalk import GridWorld

def e_greedy(Q: np.ndarray, state: int, epsilon=0.1):
    action_distribution = Q[state, :]
    action = np.argmax(action_distribution)
    if np.random.random() >= epsilon:
        return action
    else:
        return int(np.floor(np.random.random()*3.99))

def sarsa(env: GridWorld, alpha=0.05, gamma=1.0, episode=1000, epsilon=0.1, decay=0.999):
    Q = np.zeros((48, 4))
    epsilon_ = epsilon
    for i in range(episode):
        episode_ = epsilon_ * decay
        obs = env.reset()
        done = False
        action = e_greedy(Q, obs, epsilon=epsilon_)
        while not done:
            obs_, reward, done = env.step(action)
            action_ = e_greedy(Q, obs_, epsilon=epsilon_)
            Q[obs][action] += alpha * (reward + gamma * Q[obs_][action_] - Q[obs][action])
            obs = obs_
            action = action_
    return Q

def test_result(env: GridWorld, Q: np.ndarray, episode=10, render=True):
    for i in range(episode):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = np.argmax(Q[obs, :])
            obs, reward, done = env.step(action)


if __name__ == "__main__":
    env = GridWorld()
    Q = sarsa(env, episode=1000)
    print(Q)
    test_result(env, Q)

