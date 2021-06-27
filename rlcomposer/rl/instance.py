from stable_baselines3 import SAC

def disable_view_window():
  from gym.envs.classic_control import rendering
  org_constructor = rendering.Viewer.__init__

  def constructor(self, *args, **kwargs):
    org_constructor(self, *args, **kwargs)
    self.window.set_visible(visible=False)

  rendering.Viewer.__init__ = constructor



class Instance():
    def __init__(self, scene):
        self.scene = scene
        self.buildInstance()


    def buildInstance(self):
        disable_view_window()
        self.env_wrapper, self.reward_wrapper = None, None
        print("Build Instance 1")
        for item in self.scene.nodes:
            if item.title == "Environment":
                print("Build Instance 2")
                self.env_wrapper = item.wrapper
            elif item.title == "Reward":
                print("Build Instance 3")
                self.reward_wrapper = item.wrapper
        print("Build Instance 4")
        self.reward_func = self.reward_wrapper.reward
        print("Build Instance 5")
        self.env_wrapper.setReward(self.reward_func)
        print("Build Instance 6")
        self.env = self.env_wrapper.env
        print("Build Instance 7")
        # self.model = SAC("MlpPolicy", self.env, verbose=1)
        # self.model.learn(total_timesteps=20000)
        # self.model.save("sac_pendulum")
        # Load the trained model
        self.model = SAC.load("sac_pendulum")


    def step(self):
        action, _ = self.model.predict(self.state)
        # action_probabilities = self.model.action_probability(self.state)
        # action = self.env.action_space.sample()
        action_probabilities = 0
        print(self.env_wrapper.param)
        self.state, reward, done, _ = self.env.step(action)
        img = self.env.render(mode="rgb_array")

        return img, reward, done, action_probabilities

    def prep(self):
        print(self.env)
        self.state = self.env.reset()
        img = self.env.render(mode="rgb_array")
        print(type(img))
        return img

    def closeInstance(self):
        pass

    # def train(self, breakpoints=None, progress_callback=None, **kwargs):
    #     self.model.set_env(self.env)
    #     print('setting training environment', self.env)
    #     # if breakpoints is None:
    #     #     breakpoints = settings.STEPS
    #     n_steps = self.config.main.steps_to_train
    #     n_checkpoints = n_steps//breakpoints
    #
    #     train_config = dict(
    #         total_timesteps=breakpoints,
    #         tb_log_name='log_1',
    #         reset_num_timesteps=False)
    #
    #     # Train the model and save a checkpoint every n steps
    #     for i in range(n_checkpoints):
    #         if not self.stop_training:
    #             self.model = self.model.learn(
    #                 **train_config)
    #             if progress_callback:
    #                 progress_callback.emit(breakpoints)
    #
    #             self.config.main.steps_trained += breakpoints
    #     self.save_instance()
    #     self.model.set_env(self.env)