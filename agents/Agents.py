import random
import statistics as st

class AbstractAgent():
    def __init__(self, mc):
        self.total_reward = 0
        self.mc = mc
    def reset(self):
        pass
    def react(self, observation, reward):
        self.total_reward += reward
    def get_name(self):
        return "Abstract"
    def get_status_str(self):
        return ""

class MimicAgent(AbstractAgent):
    def __init__(self, mc):
        super().__init__(mc)
        self.lastObs = None
        self.mode = "coop"
        self.min_diffs = 0.0
    def reset(self):
        self.mode = "coop"
        self.lastObs = None
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = min(observation)
        if self.lastObs is None:
            self.lastObs = observation
            return min_obs
        else:
            diffs = [observation[i] - self.lastObs[i] for i in range(len(observation))]
            self.lastObs = observation
            min_diffs = min(diffs)
            self.min_diffs = min_diffs
            if min_diffs < 0:
                self.mode ="defect"
            if min_diffs >= 0:
                self.mode = "coop"
            if self.mode == "defect":
                return max(min_obs - (min_obs - self.mc) * 0.1, self.mc)
            else:
                return max(min_obs, self.mc)
    def get_name(self):
        return " Mimic"
    def get_status_str(self):
        return "{:4.2f} {:6s}".format(self.min_diffs, self.mode)



class SimpleRetalitatingAgent(AbstractAgent):
    def __init__(self, mc):
        super().__init__(mc)
        self.lastObs = None
        self.mode = "coop"
        self.min_diffs = 0.0
    def reset(self):
        self.mode = "coop"
        self.lastObs = None
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = min(observation)
        if self.lastObs is None:
            self.lastObs = observation
            return min_obs
        else:
            diffs = [observation[i] - self.lastObs[i] for i in range(len(observation))]
            self.lastObs = observation
            self.min_diffs = min(diffs)
            if self.mode == "defect":
                return max(min_obs - (min_obs - self.mc) * 0.1, self.mc)
            else:
                if min(diffs)<0:
                    self.mode = "defect"
                    return max(min_obs - (min_obs - self.mc) * 0.1, self.mc)
                else:
                    return max(min_obs, self.mc)
    def get_name(self):
        return "SimpleRetalitating"
    def get_status_str(self):
        return "{:4.2f} {:6s}".format(self.min_diffs, self.mode)

class SimpleForgivingAgent(AbstractAgent):
    def __init__(self, mc):
        super().__init__(mc)
        self.lastObs = None
        self.trustLevel = 2
        self.mode = "coop"
        self.min_diffs = 0.0
    def reset(self):
        self.mode = "coop"
        self.trustLevel = 2
        self.lastObs = None
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = min(observation)
        if self.lastObs is None:
            self.lastObs = observation
            return min_obs
        else:
            diffs = [observation[i] - self.lastObs[i] for i in range(len(observation))]
            self.lastObs = observation
            min_diffs = min(diffs)
            self.min_diffs = min_diffs
            if min_diffs < 0 and self.trustLevel > 0:
                self.trustLevel -= 1
            if min_diffs >= 0 and self.trustLevel < 2:
                self.trustLevel += 1
            if self.trustLevel == 0:
                self.mode = "defect"
            if self.trustLevel == 2:
                self.mode = "coop"
            if self.mode == "defect":
                return max(min_obs - (min_obs - self.mc) * 0.1, self.mc)
            else:
                return max(min_obs, self.mc)
    def get_name(self):
        return "SimpleForgiving"
    def get_status_str(self):
        return "{:4.2f} ({:d}) {:6s}".format(self.min_diffs, self.trustLevel,
                                             self.mode)

class RandomAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        return random.uniform(self.mc, 1)
    def get_name(self):
        return "Random"

class LoweringAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = min(observation)
        return max(min_obs - (min_obs - self.mc)*0.1, self.mc)
    def get_name(self):
        return "Lowering"

class KeepingAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = min(observation)
        return max(min_obs, self.mc)
    def get_name(self):
        return "Keeping"

class IncreasingAgent(AbstractAgent):
    def react(self, observation, reward):
        super().react(observation, reward)
        min_obs = min(observation)
        return max(min_obs * 1.1, self.mc)
    def get_name(self):
        return "Increasing"
