import gym
from DQN_agent import DQN
import numpy as np
episode = 0

# def run_gym():
#     step = 0
#     env = gym.make('CartPole-v1')
#     state = env.reset()
#
#     for episode in range(300):
#         while True:
#             env.render()
#             action = RL.choose_action(state)
#             state_, reward, done = env.step(action)
#             RL.store_transition(state, action, reward, state_)





if __name__ == "__main__":
    # maze game
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    actions_size = env.action_space.n
    agent = DQN(state_size, actions_size)
    done = False
    batch_size = 32

    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state=next_state
            if done:
                print("OK")
                break
        if len(agent.memory) >batch_size:
            agent.replay(batch_size)

