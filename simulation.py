import gym

import csv

from agents.Agents import RandomAgent, KeepingAgent, LoweringAgent, IncreasingAgent, SimpleForgivingAgent, SimpleRetalitatingAgent, MimicAgent

from agents.TFAgent import TFAgent

from gym_bertrand.envs.bertrand_env import BertrandEnv

env = gym.make('bertrand-v0')

agents = []

marginal_cost = 0.1
reservation_price = 0.8

agents.append(RandomAgent(marginal_cost))
#agents.append(RandomAgent(marginal_cost))
agents.append(KeepingAgent(marginal_cost))
#agents.append(KeepingAgent(marginal_cost))
agents.append(LoweringAgent(marginal_cost))
#agents.append(LoweringAgent(marginal_cost))
agents.append(IncreasingAgent(marginal_cost))
#agents.append(IncreasingAgent(marginal_cost))
agents.append(SimpleForgivingAgent(marginal_cost))
agents.append(SimpleRetalitatingAgent(marginal_cost))
agents.append(MimicAgent(marginal_cost))

tfagent = TFAgent(marginal_cost)
agents.append(tfagent)

env.custom_init(len(agents), marginal_cost, reservation_price)

tfagent.set_env(env)




with open('actions.csv', 'a', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i_episode in range(1000):
        observation = env.reset()
        action = [env.action_space.sample() for x in agents]
        for t in range(100):
            #env.render()
            observation, reward, done, info = env.step(action)
            obs = []
            for o in observation:
                if type(o) == float:
                    obs.append(o)
                else:
                    obs.append(o[0])
            #writer.writerow(obs)
            action = []
            for i in range(len(agents)):
                action.append(agents[i].react(observation[i], reward[i]))
            if done:
                rews = [agent.total_reward*len(agents) for agent in agents]
                [agent.reset() for agent in agents]
                writer.writerow(rews)
                print("Episode {} fisnished after {} timesteps".format(i_episode, t+1))
                env.reset()
                print(rews)
                break
        if(i_episode==900):
            for agent in agents:
                agent.total_reward = 0
env.close()

