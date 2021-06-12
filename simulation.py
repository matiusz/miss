import gym

import csv

from agents.Agents import RandomAgent, KeepingAgent, LoweringAgent, \
    IncreasingAgent, SimpleForgivingAgent, SimpleRetalitatingAgent, MimicAgent

import utils.DisplayUtils as du

from agents.TFAgent import TFAgent

import numpy as np

from gym_bertrand.envs.bertrand_env import BertrandEnv

marginal_cost = 0.1
reservation_price = 0.8

actions_num = 1000
learning_rate = 0.02
reward_decay = 0.99
units_num = 7
repeats_num = 10

runs_num = 2
episodes_num = 200
time_steps_per_episode_num = 20
series_size = 100
series_counter_stop_value = episodes_num // series_size
total_program_steps = repeats_num * runs_num * episodes_num
cur_program_step = 0

results_sum = np.zeros((runs_num, series_size + 1, 8))
prices = np.zeros((runs_num, series_size + 1, 8))

for repeat_id in range(repeats_num):

    env = gym.make('bertrand-v0')

    agents = []
    agents.append(RandomAgent(marginal_cost))
    agents.append(KeepingAgent(marginal_cost))
    agents.append(LoweringAgent(marginal_cost))
    agents.append(IncreasingAgent(marginal_cost))
    agents.append(SimpleForgivingAgent(marginal_cost))
    agents.append(SimpleRetalitatingAgent(marginal_cost))
    agents.append(MimicAgent(marginal_cost))

    # agents.append(IncreasingAgent(marginal_cost))
    # agents.append(IncreasingAgent(marginal_cost))
    # agents.append(IncreasingAgent(marginal_cost))
    # agents.append(IncreasingAgent(marginal_cost))
    # agents.append(IncreasingAgent(marginal_cost))
    # agents.append(IncreasingAgent(marginal_cost))
    # agents.append(IncreasingAgent(marginal_cost))

    # agents.append(IncreasingAgent(marginal_cost))

    tfagent = TFAgent(marginal_cost)
    agents.append(tfagent)

    env.custom_init(len(agents), marginal_cost, reservation_price,
                    time_steps_per_episode_num)

    tfagent.set_env(env)
    display = du.GameStateDisplay(agents, episodes_num *
                                  time_steps_per_episode_num / 70)
    for run_id in range(runs_num):
        series_counter = 0
        series_part = 0
        for i_episode in range(episodes_num):
            observation = env.reset()
            action = [1.0 for x in agents]
            for t in range(time_steps_per_episode_num):
                # env.render()
                observation, reward, done, info = env.step(action)
                obs = []
                for o in observation:
                    if type(o) == float:
                        obs.append(o)
                    else:
                        obs.append(o[0])
                action = []
                for i in range(len(agents)):
                    action.append(agents[i].react(observation[i], reward[i]))
                if done:
                    rews = [agent.total_reward * len(agents) for agent in
                            agents]
                    series_counter += 1
                    if series_counter == series_counter_stop_value:
                        # writer.writerow(rews)
                        display.display_game_state(action, rews)
                        print('\rExecuting experiment... ' + str(
                            (cur_program_step * 100) //
                            total_program_steps) + '%',
                              end='')
                        for agent_Id in range(len(rews)):
                            results_sum[run_id, series_part, agent_Id] += \
                                rews[agent_Id]
                            prices[run_id, series_part, agent_Id] += \
                                action[agent_Id]
                        series_counter = 0
                        series_part += 1
                    [agent.reset() for agent in agents]
                    # print("Episode {} fisnished after {} timesteps".format(
                    #     i_episode, t + 1))
                    # print(rews)
                    env.reset()
                    cur_program_step += 1
                    break
        for agent in agents:
            agent.total_reward = 0
            display.reset()
    env.close()
print('\rExecuting experiment... 100% DONE')
for run_id in range(runs_num):
    with open('rewards' + str(run_id) + '.csv', 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for series_part in range(series_size):
            # print(f'{series_part}')
            for agent_Id in range(len(rews)):
                results_sum[run_id, series_part, agent_Id] /= repeats_num
            writer.writerow(results_sum[run_id, series_part])
with open('prices' + '.csv', 'a', newline='\n') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for run_id in range(runs_num):
        for series_part in range(series_size):
            # print(f'{series_part}')
            for agent_Id in range(len(rews)):
                prices[run_id, series_part, agent_Id] /= repeats_num
            writer.writerow(prices[run_id, series_part])

