import gym

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(state_size,action_size)
print(env.observation_space)
print(env.action_space)