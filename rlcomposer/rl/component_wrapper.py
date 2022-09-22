import sys
import importlib
import os
import pprint
import ppo.ppo_components as ppo_components
import ppo.sac_components as sac_components


def excludeVariables(component):
    dic = vars(component)
    res_dic = {}
    parameter_box = dic['parameter_box']
    for key in parameter_box:
        res_dic[key] = dic[key]
    # for (key, val) in dic.items():
    #     if (type(val) == float or type(val) == int):
    #         res_dic[key] = val
    return res_dic


def get_shape(name):
    shape_dict = {"policy_forward_output": '3',
                  "observations": '250',
                  "observation_updates": "250"}
    return shape_dict.get(name, '1')


def to_bool(value):
    """
       Converts 'something' to boolean. Raises exception for invalid formats
           Possible True  values: 1, True, "1", "TRue", "yes", "y", "t"
           Possible False values: 0, False, None, [], {}, "", "0", "faLse", "no", "n", "f", 0.0, ...
    """
    if str(value).lower() in ("yes", "y", "true", "t", "1"): return True
    if str(value).lower() in ("no", "n", "false", "f", "0", "0.0", "", "none", "[]", "{}"): return False
    raise Exception('Invalid value for boolean conversion: ' + str(value))


class ComponentWrapper():
    def __init__(self, component_name="", component_title=""):
        self.component_name = component_name
        self.component_title = component_title
        self.component = None
        self.param = {"Outputs": [], "Inputs": [], "States": [], "Arguments": [self.component_argument()]}
        self.setComponent()

    def setComponent(self):
        print(sys.modules[__name__])
        if self.component_title == 'PPO_Components':
            self.component = getattr(ppo_components, self.component_name)(**self.component_argument())
        elif self.component_title == 'SAC_Components':
            self.component = getattr(sac_components, self.component_name)(**self.component_argument())
        print(self.component)

        for outputs in self.component.output_names:
            property_dict = {"Name": outputs,
                             "Shape": get_shape(outputs),
                             "Is_Process_Parallel": str(True)}
            self.param['Outputs'].append(property_dict)

        for inputs in self.component.input_names:
            property_dict = {"Name": inputs,
                             "Shape": get_shape(inputs),
                             "Is_Process_Parallel": str(True)}
            self.param['Inputs'].append(property_dict)

        for states in self.component.state_names:
            property_dict = {"Name": states,
                             "Shape": get_shape(states),
                             "Is_Process_Parallel": str(True)}
            self.param['States'].append(property_dict)

    def setParameters(self, param):
        if self.component_argument() != param['Arguments'][0]:
            print(sys.modules[__name__])
            if self.component_title == 'PPO_Components':
                self.component = getattr(ppo_components, self.component_name)(**param['Arguments'][0])
            elif self.component_title == 'SAC_Components':
                self.component = getattr(sac_components, self.component_name)(**param['Arguments'][0])
            print(self.component)

        self.param = param

    def component_argument(self):
        if self.component_name == 'RolloutCollector':
            return {"rollout_steps": 2048}
        elif self.component_name == 'RLTrainer':
            return {"total_training_timesteps": 10}
        else:
            return {}

    def bindParameters(self):
        for outputs in self.param["Outputs"]:
            self.component.set_output(outputs["Name"], (int(outputs["Shape"]),),
                                      to_bool(outputs["Is_Process_Parallel"]))
        for inputs in self.param["Inputs"]:
            self.component.set_input(inputs["Name"], (int(inputs["Shape"]),), to_bool(inputs["Is_Process_Parallel"]))
        for states in self.param["States"]:
            self.component.set_state(states["Name"], (int(states["Shape"]),), to_bool(states["Is_Process_Parallel"]))
        if self.component_name == 'RLTrainer':
            print('1')
            self.component.iterations = 1

    def print_info(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.component.name)
        pp.pprint(self.component.__dict__)