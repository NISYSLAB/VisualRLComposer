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
        module = importlib.import_module(".environments", "rlcomposer.rl.components")
        print(sys.modules[__name__])
        self.env = getattr(module, self.env_name)()
        self.param = excludeVariables(self.env)


    def setReward(self, reward):
        setattr(self.env, "reward_fn", reward)

    def setParameters(self, param):
        for key,value in param.items():
          setattr(self.env, key, value)


