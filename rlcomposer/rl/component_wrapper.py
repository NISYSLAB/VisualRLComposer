import sys
import importlib
import os
import pprint
#from .components.testing_components import *


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

class ComponentWrapper():
    def __init__(self, component_name=""):
        print("A")
        self.component_name = component_name
        self.component = None
        self.param = {}
        self.setComponent()
        self.setParameters(self.param)

    def setComponent(self):
        import test.testing_components as testing_components
        # import ppo.
        #module = importlib.import_module("testing_components", "test")
        print(sys.modules[__name__])

        self.component = getattr(testing_components, self.component_name)(**self.component_argument())
        print(self.component.output_names, self.component.input_names, self.component.state_names)

        if len(self.component.output_names) > 0:
            self.param['Output Name'] = self.component.output_names
            self.param['Output Shape'] = (1,1)
        if len(self.component.input_names) > 0:
            self.param['Input Name'] = self.component.input_names
            self.param['Input Shape'] = (1,1)
        if len(self.component.state_names) > 0:
            self.param['State Name'] = self.component.state_names
            self.param['State Shape'] = (1,1)

    def setParameters(self, param):
        pass

    def component_argument(self):
        if self.component_name == 'Inference':
            return {'filepath': './weights.txt'}
        elif self.component_name == 'SignalToStimulation':
            return {'component_parameter': '1'}
        elif self.component_name == 'SignalToStimulationOnGPU':
            return {'component_parameter': '1'}
        elif self.component_name == 'Trainer':
            return {'lp_parameter': '1'}
        else:
            return {}

    def bindProperties(self):
        if len(self.component.output_names) > 0:
            self.component.set_output(*self.param['Output Name'], self.param['Output Shape'])
        if len(self.component.input_names) > 0:
            self.component.set_input(*self.param['Input Name'], self.param['Input Shape'])
        if len(self.component.state_names) > 0:
            self.component.set_state(*self.param['State Name'], self.param['State Shape'])

    def print_info(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.component.name)
        pp.pprint(self.component.__dict__)