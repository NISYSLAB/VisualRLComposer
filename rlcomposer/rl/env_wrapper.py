import sys
import importlib
import gym
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

def excludeVariables(obj):
    dic = vars(obj)
    res_dic = {}
    print(dic)
    parameter_box = obj.get_attr('parameter_box')
    for key in parameter_box[0]:
        res_dic[key] = obj.get_attr(key)[0]
    # for (key, val) in dic.items():
    #     if (type(val) == float or type(val) == int):
    #         res_dic[key] = val
    return res_dic

class EnvWrapper():
    def __init__(self, env_name=""):
        self.env_name = env_name
        self.env = None
        self.setEnv()
        # self.setParameters(self.param)


    def setEnv(self):
        module = importlib.import_module(".environments", "rlcomposer.rl.components")
        print(sys.modules[__name__])
        # self.env = getattr(module, self.env_name)()

        try:
            #self.env = SubprocVecEnv([make_env(self.env_name+'-v10', i) for i in range(1)])
            self.env = make_vec_env(self.env_name+'-v10', n_envs=1, seed=set_random_seed(0), vec_env_cls=SubprocVecEnv)
            pass
        except Exception as e:
            print("ERROR", e)
        self.param = excludeVariables(self.env)


    def setReward(self, reward):
        # self.env.set_attr('reward_fn', reward)
        self.env.env_method("update_reward", reward)

    def setParameters(self, param):
        if self.env.get_attr('n_envs')[0] != param['n_envs']:
            self.env = make_vec_env(self.env_name + '-v10', n_envs=param['n_envs'], seed=set_random_seed(0), vec_env_cls=SubprocVecEnv)

        self.env.env_method("update_values", param)
        for key, value in param.items():
            self.env.set_attr(key, value)

