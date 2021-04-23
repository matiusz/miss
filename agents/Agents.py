import random
#import tensorflow

class AbstractAgent():
    def __init__(self, mc):
        self.total_reward = 0
        self.mc = mc
    def react(self, observation, reward):
        self.total_reward += reward

class RandomAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        return random.uniform(self.mc, 1)

class LoweringAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = 1
        for obs in observation:
            if obs<min_obs:
                min_obs = obs
        return max(min_obs - (min_obs - self.mc)*0.1, self.mc)
1
class KeepingAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = 1
        for obs in observation:
            if obs < min_obs:
                min_obs = obs
        return max(min_obs, self.mc)
1
class IncreasingAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = 1
        for obs in observation:
            if obs < min_obs:
                min_obs = obs
        return max(min_obs * 1.1, self.mc)

class LearningAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        raise NotImplementedError