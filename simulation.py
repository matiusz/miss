import gym

import csv

from agents.Agents import RandomAgent, KeepingAgent, LoweringAgent, IncreasingAgent

from gym_bertrand.envs.bertrand_env import BertrandEnv

env = gym.make('bertrand-v0')

agents = []

marginal_cost = 0.1
reservation_price = 0.8

agents.append(RandomAgent(marginal_cost))
agents.append(RandomAgent(marginal_cost))
agents.append(KeepingAgent(marginal_cost))
agents.append(KeepingAgent(marginal_cost))
agents.append(LoweringAgent(marginal_cost))
agents.append(LoweringAgent(marginal_cost))
agents.append(IncreasingAgent(marginal_cost))
agents.append(IncreasingAgent(marginal_cost))

env.custom_init(len(agents), marginal_cost, reservation_price)

with open('actions.csv', 'a', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i_episode in range(100):
        observation = env.reset()
        action = [env.action_space.sample() for x in agents]
        for t in range(20):
            #env.render()
            observation, reward, done, info = env.step(action)
            obs = []
            for o in observation:
                if type(o) == float:
                    obs.append(o)
                else:
                    obs.append(o[0])
            writer.writerow(obs)
            action = []
            for i in range(len(agents)):
                action.append(agents[i].react(observation[i], reward[i]))
            if done:
                rews = [agent.total_reward for agent in agents]
                print("Episode fisnished after {} timesteps".format(t+1))
                env.reset()
                print(rews)
                #writer.writerow(rews)
                break
env.close()