from gym.envs.registration import register

register(
    id='bertrand-v0',
    entry_point='gym_bertrand.envs:BertrandEnv',
)