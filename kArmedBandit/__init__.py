from gym.envs.registration import register

register(
    id='TenArmedBanditGaussian-v0',
    entry_point='kArmedBandit.envs:TenArmedBanditGaussianRewardEnv',
)
register(
    id='TenArmedBanditFixed-v0',
    entry_point='kArmedBandit.envs:TenArmedBanditFixedRewardEnv',
)
