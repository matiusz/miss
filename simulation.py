import gym
import csv
import sys
import getopt
import numpy as np
import utils.DisplayUtils as du
from agents.TFAgent import TFAgent

from agents.Agents import RandomAgent, KeepingAgent, LoweringAgent, \
    IncreasingAgent, SimpleForgivingAgent, SimpleRetalitatingAgent, MimicAgent

from gym_bertrand.envs.bertrand_env import BertrandEnv

def main(argv):

    marginal_cost = 0.1
    reservation_price = 0.8
    episodes_num = 500
    time_steps_per_episode_num = 20
    learning_rate = 0.02
    reward_decay = 0.99

    repeats_num = 10
    runs_num = 2
    series_size = 100
    prices_measure_times = 4

    csv_file_name_suffix = ""

    try:
        opts, args = getopt.getopt(argv, "", ["mc=", "rp=", "en=", "tspe=",
                                              "lr=", "rd=", "agents="])
        for opt, arg in opts:
            if opt == "--mc":
                marginal_cost = float(arg)
            elif opt == "--rp":
                reservation_price = float(arg)
            elif opt == "--en":
                episodes_num = int(arg)
            elif opt == "--tspe":
                time_steps_per_episode_num = int(arg)
            elif opt == "--lr":
                learning_rate = float(arg)
            elif opt == "--rd":
                reward_decay = float(arg)
            csv_file_name_suffix += "." + opt[2:] + "-" + arg
    except getopt.GetoptError:
        print("Error in parsing program arguments!")

    if episodes_num < series_size:
        series_size = episodes_num

    series_counter_stop_value = episodes_num // series_size
    total_program_steps = repeats_num * runs_num * episodes_num
    cur_program_step = 0

    prices_measure_id = 0
    prices_measure_counter = 0
    prices_measure_stop_value = episodes_num // prices_measure_times

    for repeat_id in range(repeats_num):

        env = gym.make('bertrand-v0')

        agents = []
        tf_agents = []

        try:
            opts, args = getopt.getopt(argv, "", ["mc=", "rp=", "en=", "tspe=",
                                              "lr=", "rd=", "agents="])
            for opt, arg in opts:
                if opt == "--agents":
                    for agent_char in arg.lower():
                        if agent_char == 'r':
                            agents.append(RandomAgent(marginal_cost))
                        elif agent_char == 'k':
                            agents.append(KeepingAgent(marginal_cost))
                        elif agent_char == 'l':
                            agents.append(LoweringAgent(marginal_cost))
                        elif agent_char == 'i':
                            agents.append(IncreasingAgent(marginal_cost))
                        elif agent_char == 'f':
                            agents.append(SimpleForgivingAgent(marginal_cost))
                        elif agent_char == 'e':
                            agents.append(SimpleRetalitatingAgent(marginal_cost))
                        elif agent_char == 'm':
                            agents.append(MimicAgent(marginal_cost))
                        elif agent_char == 't':
                            tf_agent = TFAgent(marginal_cost)
                            agents.append(tf_agent)
                            tf_agents.append(tf_agent)
        except getopt.GetoptError:
            pass

        if not agents:
            agents.append(RandomAgent(marginal_cost))
            agents.append(KeepingAgent(marginal_cost))
            agents.append(LoweringAgent(marginal_cost))
            agents.append(IncreasingAgent(marginal_cost))
            agents.append(SimpleForgivingAgent(marginal_cost))
            agents.append(SimpleRetalitatingAgent(marginal_cost))
            agents.append(MimicAgent(marginal_cost))
            tf_agent = TFAgent(marginal_cost)
            agents.append(tf_agent)
            tf_agents.append(tf_agent)

        if repeat_id == 0:
            results_sum = np.zeros((runs_num, series_size + 1, len(agents)))
            prices = np.zeros((prices_measure_times + 1,
                               time_steps_per_episode_num, len(agents)))

        env.custom_init(len(agents), marginal_cost, reservation_price,
                        time_steps_per_episode_num)

        for tf_agent in tf_agents:
            tf_agent.set_env(env, learning_rate=learning_rate,
                            reward_decay=reward_decay)
        display = du.GameStateDisplay(agents, episodes_num *
                                      time_steps_per_episode_num / 70)
        for run_id in range(runs_num):
            series_counter = 0
            series_part = 0
            for i_episode in range(episodes_num):
                observation = env.reset()
                action = [1.0 for x in agents]
                for t in range(time_steps_per_episode_num):
                    observation, reward, done, info = env.step(action)
                    obs = []
                    for o in observation:
                        if type(o) == float:
                            obs.append(o)
                        else:
                            obs.append(o[0])
                    action = []
                    for i in range(len(agents)):
                        action.append(
                            agents[i].react(observation[i], reward[i]))
                    if repeat_id == 0 and run_id == 0 and \
                            prices_measure_counter == 0:
                        for agent_Id in range(len(action)):
                            prices[prices_measure_id, t, agent_Id] = \
                                action[agent_Id]
                    if done:
                        rews = [agent.total_reward * len(agents) for agent in
                                agents]
                        series_counter += 1
                        if series_counter == series_counter_stop_value:
                            display.display_game_state(action, rews)
                            print('\rExecuting experiment... ' + str(
                                (cur_program_step * 100) //
                                total_program_steps) + '%',
                                  end='')
                            for agent_Id in range(len(rews)):
                                results_sum[run_id, series_part, agent_Id] += \
                                    rews[agent_Id]
                            series_counter = 0
                            series_part += 1
                        if repeat_id == 0:
                            prices_measure_counter += 1
                            if prices_measure_counter == \
                                    prices_measure_stop_value:
                                prices_measure_counter = 0
                                prices_measure_id += 1
                        [agent.reset() for agent in agents]
                        env.reset()
                        cur_program_step += 1
                        break
            for agent in agents:
                agent.total_reward = 0
                display.reset()
        env.close()
    print('\rExecuting experiment... 100% DONE')
    with open('rewards' + csv_file_name_suffix + '.csv', 'a',
              newline='\n') as csvfile:
        for run_id in range(runs_num):
            csvfile.truncate()
            writer = csv.writer(csvfile, delimiter=',')
            for series_part in range(series_size):
                for agent_Id in range(len(rews)):
                    results_sum[run_id, series_part, agent_Id] /= repeats_num
                writer.writerow(results_sum[run_id, series_part])
    with open('prices' + csv_file_name_suffix + '.csv', 'a',
              newline='\n') as csvfile:
        csvfile.truncate()
        writer = csv.writer(csvfile, delimiter=',')
        for meas_id in range(prices_measure_times):
            for series_part in range(time_steps_per_episode_num):
                writer.writerow(prices[meas_id, series_part])


if __name__ == "__main__":
    main(sys.argv[1:])
