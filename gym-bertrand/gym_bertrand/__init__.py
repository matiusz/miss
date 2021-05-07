from gym.envs.registration import register

register(
    id='bertrand-v0',
    entry_point='gym_bertrand.envs.bertrand_env:BertrandEnv',
    reward_threshold=1.0,
    nondeterministic=False,
)