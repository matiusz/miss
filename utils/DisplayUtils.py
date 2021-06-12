
class GameStateDisplay:

    def __init__(self, agents, max_reward_bar=0.0):
        self.agents = agents
        self.first_print = True
        self.last_total_rewards = None
        self.max_reward_bar = max_reward_bar

    def display_game_state(self, actions, total_rewards):
        if not self.first_print:
            print("\r\033[%dA" % (len(self.agents) + 1))
        else:
            self.first_print = False
        max_reward = self.max_reward_bar
        max_reward_delta = 0.0
        for agent_id in range(len(self.agents)):
            if total_rewards[agent_id] > max_reward:
                max_reward = total_rewards[agent_id]
            reward_delta = \
                total_rewards[agent_id] - self.last_total_rewards[agent_id] \
                    if self.last_total_rewards is not None else 0.0
            if reward_delta > max_reward_delta:
                max_reward_delta = reward_delta
        for agent_id in range(len(self.agents)):
            agent = self.agents[agent_id]
            reward_delta = \
                total_rewards[agent_id] - self.last_total_rewards[agent_id] \
                if self.last_total_rewards is not None else 0.0
            total_reward_bar_len = (int(total_rewards[agent_id] * 50 //
                                      max_reward))
            total_reward_bar = '>' * total_reward_bar_len +\
                               ' ' * (50 - total_reward_bar_len)
            reward_delta_bar_len = (int(reward_delta * 10 //
                                      max_reward_delta
                                      if max_reward_delta != 0.0 else 1.0))
            reward_delta_bar_len = min(reward_delta_bar_len, 10)
            reward_delta_bar = '+' * reward_delta_bar_len +\
                               ' ' * (10 - reward_delta_bar_len)
            print("%1d | %20s | %20s | %1.3f | %6.2f | +%5.2f | %50s | %10s" %
                  (agent_id + 1, agent.get_name(), agent.get_status_str(),
                   actions[agent_id], total_rewards[agent_id],
                   reward_delta, total_reward_bar, reward_delta_bar))
        self.last_total_rewards = total_rewards

    def reset(self):
        self.last_total_rewards = None