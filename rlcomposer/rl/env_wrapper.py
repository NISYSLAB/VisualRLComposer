import sys
import importlib

def excludeVariables(obj):
    dic = vars(obj)
    res_dic = {}
    for (key, val) in dic.items():
        if (type(val) == float or type(val) == int):
            res_dic[key] = val
    return res_dic

class EnvWrapper():
    def __init__(self, env_name=""):
        self.env_name = env_name
        self.setEnv()
        # self.setParameters(self.param)


    def setEnv(self):
        from components.environments import Pendulum
        module = importlib.import_module("components.environments")
        print(sys.modules[__name__])
        self.env = getattr(module, self.env_name)()
        self.param = excludeVariables(self.env)


    # def setParam(self):
    #     print("set param")
    #     if self.env_name == "Pendulum":
    #         self.param = {"max_speed": 8,
    #                     "max_torque": 2.,
    #                     "dt": .05,
    #                     "g": 10.0,
    #                     "m": 1.,
    #                     "l": 1., }
    #     elif self.env_name == "MountainCarEnv":
    #         self.param = {
    #         "min_position": -1.2,
    #         "max_position": 0.6,
    #         "max_speed": 0.07,
    #         "goal_position": 0.5,
    #         "goal_velocity": 0,
    #         "force": 0.001,
    #         "gravity": 0.0025
    #         }

    def setReward(self, reward):
        setattr(self.env, "reward_fn", reward)

    def setParameters(self, param):
        for key,value in param.items():
          setattr(self.env, key, value)


