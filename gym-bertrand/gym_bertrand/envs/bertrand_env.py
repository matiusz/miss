import gym
import tensorflow as tf
from gym import spaces
from collections import namedtuple
from tf_agents.specs import BoundedArraySpec
from tf_agents.specs import ArraySpec
from tf_agents.trajectories.time_step import TimeStep, StepType
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from gym.utils import seeding

tf.compat.v1.enable_v2_behavior()

class BertrandEnv(gym.Env, py_environment.PyEnvironment):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BertrandEnv, self).__init__()
        self.epoch = 0
        self.agent_count = 0
        self.marginal_cost = 0
        self.reservation_price = 0
        self.action_space = None
        self.laststep = None
        self.act_spec = None
        self.obs_spec = None

    def custom_init(self, agent_count, marginal_cost, reservation_price):
        self.epoch = 0
        self.agent_count = agent_count
        self.marginal_cost = marginal_cost
        self.reservation_price = reservation_price
        self.action_space = spaces.Box(low=self.marginal_cost, high=1, shape=[])
        self.laststep = [1 for x in range(self.agent_count)]
        min_obs = [[self.marginal_cost] * (
                self.agent_count - 1)] * self.agent_count
        max_obs = [[1.0] * (self.agent_count - 1)] * self.agent_count
        self.act_spec = BoundedArraySpec((), float, 0.0, 1.0,
                                            'action')
        self.obs_spec = BoundedArraySpec(
            (self.agent_count, self.agent_count - 1),
            float, min_obs, max_obs, 'observation')

    # def step(self, action):
    #     return self._step(action)

    def _step(self, action):
        bestOffer = None
        bestCompetitors = []
        for i in range(len(action)):
            if bestOffer == None or bestOffer > action[i]:
                bestOffer = action[i]
                bestCompetitors = [i]
            elif bestOffer == action[i]:
                bestCompetitors.append(i)
        reward_n = [0 for x in range(len(action))]
        if bestOffer <= self.reservation_price:
            for comp in bestCompetitors:
                reward_n[comp] = (bestOffer - self.marginal_cost) / len(
                    bestCompetitors)

        self.laststep = action
        obs_n = []
        for i in range(self.agent_count):
            obs_n.append(action[:i] + action[(i + 1):])

        self.epoch += 1

        done_n = (self.epoch >= 20)

        info_n = {'n': []}
        step_type = StepType.LAST if done_n else StepType.MID

        time_step = TimeStep(step_type, reward_n, 1.0, obs_n)

        return time_step

    def time_step_spec(self):
        reward_spec = ArraySpec((self.agent_count, ), float, 'reward')
        TimeStepSpec = namedtuple('TimeStepSpec', ['observation', 'reward'])

        time_step_spec = TimeStepSpec(self.obs_spec, reward_spec)
        return time_step_spec

    def action_spec(self):
        return self.act_spec

    def observation_spec(self):
        return self.obs_spec

    # def reset(self):
    #     return self._reset()

    def _reset(self):
        self.laststep = [0 for x in range(self.agent_count)]
        self.epoch = 0
        reward = [0.0] * self.agent_count
        observation = []
        for i in range(self.agent_count):
            observation.append(self.laststep[:i] + self.laststep[(i + 1):])
        time_step = TimeStep(StepType.FIRST, reward, 1.0, observation)
        return time_step

    def render(self, mode='human'):
        print(self.laststep)

    def _seed(self):
        return self.seed()

    def close(self):
        pass
