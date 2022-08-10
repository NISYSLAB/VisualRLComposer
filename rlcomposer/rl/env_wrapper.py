import sys
import importlib
import os
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import gym

def excludeVariables(obj):
    dic = vars(obj.env.env)
    res_dic = {}
    parameter_box = dic['parameter_box']
    for key in parameter_box:
        res_dic[key] = dic[key]
    # for (key, val) in dic.items():
    #     if (type(val) == float or type(val) == int):
    #         res_dic[key] = val

    return res_dic

def make_env(env_id, rank, seed=None):
    if isinstance(env_id, str):
        env = gym.make(env_id)
    else:
        env = env_id()
    if seed is not None:
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
    env = Monitor(env, filename=None)
    return env

class EnvWrapper():
    def __init__(self, env_name=""):
        self.env_name = env_name
        self.env = None
        self.setEnv(0)
        self.setParameters(self.param)


    def setEnv(self, rank):
        module = importlib.import_module(".environments", "rlcomposer.rl.components")
        print(sys.modules[__name__])
        #self.env = make_vec_env(self.env_name+'-v10', n_envs=1, seed=set_random_seed(0), vec_env_cls=SubprocVecEnv)
        self.env = make_env(self.env_name+'-v10', rank)
        self.param = excludeVariables(self.env)

    def callable_env(self):
        def _init():
            return self.env
        return _init

    def setReward(self, reward):
        setattr(self.env.env.env, 'reward_fn', reward)

    def setParameters(self, param):
        for key, value in param.items():
            setattr(self.env.env.env, key, value)
