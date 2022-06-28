from gym.envs.registration import register

register(id='Pendulum-v10',
         entry_point='rlcomposer.rl.components.environments:Pendulum',
         max_episode_steps=300)

register(id='MountainCarEnv-v10',
         entry_point='rlcomposer.rl.components.environments:MountainCarEnv',
         max_episode_steps=300)

register(id='Continuous_MountainCarEnv-v10',
         entry_point='rlcomposer.rl.components.environments:Continuous_MountainCarEnv',
         max_episode_steps=300)

register(id='CartPoleEnv-v10',
         entry_point='rlcomposer.rl.components.environments:CartPoleEnv',
         max_episode_steps=300)

register(id='AcrobotEnv-v10',
         entry_point='rlcomposer.rl.components.environments:AcrobotEnv',
         max_episode_steps=300)
