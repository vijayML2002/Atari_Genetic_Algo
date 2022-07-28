import gym
env = gym.make('SpaceInvaders-v4') 
observation = env.reset()
for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Finished after {} timesteps".format(t+1))
            break