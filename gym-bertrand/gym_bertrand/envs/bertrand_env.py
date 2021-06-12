import gym
from gym import error, spaces, utils
from gym.utils import seeding


class BertrandEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BertrandEnv, self).__init__()

    def custom_init(self, agent_count, marginal_cost, reservation_price,
                    epochs_num=20):
        self.epoch = 0
        self.agent_count = agent_count
        self.marginal_cost = marginal_cost
        self.reservation_price  = reservation_price
        self.action_space = spaces.Box(low=self.marginal_cost, high=1, shape=[])
        self.observation_space = spaces.Box(low=0, high=1, shape=[agent_count-1])
        self.laststep = [1 for x in range(self.agent_count)]
        self.epochs_num = epochs_num

    def step(self, action):
        bestOffer = None
        bestCompetitors = []
        for i in range(len(action)):
            if bestOffer == None or bestOffer > action[i]:
                bestOffer = action[i]
                bestCompetitors = [i]
            elif bestOffer == action[i]:
                bestCompetitors.append(i)
        reward_n = [0 for x in range(len(action))]
        if bestOffer<=self.reservation_price:
            for comp in bestCompetitors:
                reward_n[comp] = (bestOffer - self.marginal_cost) / len(bestCompetitors)

        self.laststep = action
        obs_n = []
        for i in range(self.agent_count):
            obs_n.append(action[:i] + action[(i + 1):])

        self.epoch += 1

        done_n = (self.epoch>=self.epochs_num)

        info_n = {'n': []}
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.laststep = [1 for x in range(self.agent_count)]
        self.epoch = 0
        return self.laststep

    def render(self, mode='human'):
        print(self.laststep)

    def close(self):
        pass
