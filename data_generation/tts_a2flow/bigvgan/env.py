# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import os
import shutil
import json
from typing import List, Dict
import ast

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def str_to_bool(value):
    """Converts string to boolean if applicable."""
    if value.lower() in ('true'):
        return True
    elif value.lower() in ('false'):
        return False
    return None

def update_params(
    config: Dict,
    params: List
) -> Dict:
    for param in params:
        print(param)
        k, v = param.split("=")
        boolean_value = str_to_bool(v)
        if boolean_value is None:  # str_to_bool did not return a boolean, try other conversions
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):  # Catch SyntaxError as well for malformed literals
                pass  # v remains a string if it cannot be evaluated to a Python literal
        else:
            v = boolean_value  # Use the boolean value returned by str_to_bool
            
        k_split = k.split('.')
        if len(k_split) > 1:
            parent_k = k_split[0]
            cur_param = ['.'.join(k_split[1:])+"="+str(v)]
            update_params(config[parent_k], cur_param)
        elif k in config and len(k_split) == 1:
            print(f"overriding {k} with {v}")
            config[k] = v
        elif len(k_split) == 1:
            print(f"new params {k} with {v}")
            config[k] = v
        else:
            print("{}, {} params not updated".format(k, v))
    
    return config

def build_env(config, config_name, path):
    os.makedirs(path, exist_ok=True)
    t_path = os.path.join(path, config_name)
    with open(t_path, 'w') as f:
        json.dump(config, f, indent=4)