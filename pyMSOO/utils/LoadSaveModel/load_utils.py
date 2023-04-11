from ...MFEA.model import * 
from ...MFEA.competitionModel import * 
from pyMSOO.utils.Crossover import *
from pyMSOO.utils.Mutation import *
from pyMSOO.utils.Selection import *
from pyMSOO.utils.Search import * 
from pyMSOO.MFEA.benchmark.continous import *
from pyMSOO.utils.MultiRun.RunMultiTime import * 

from pyMSOO.utils.EA import * 
from pyMSOO.MFEA.benchmark.continous.funcs import * 

import inspect
import sys 
import numpy as np 


PRINT_ERROR = True 

ls_error = [] 
primary_type= [int,np.int32,np.uint8, float, np.float32, np.float64, np.ndarray,tuple, bool, str, list] 


def classname_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def module_class_str_to_class(module, classname, old_name_module= None, new_name_module= None):
    '''
    Target: 
        convert string to 'class variables'
    Optional: 
        old_name_module, new_name_module: only apply for the model such as SM_MFEA, MFEA_base, ... 
    '''
    # print(module, classname)
    if module == old_name_module: 
        module= new_name_module
    elif module == "SMP_MFEA":
        module= "SM_MFEA"
    return getattr(sys.modules[__name__].__dict__[module], classname)

def restore_list_attr(ls):
    global primary_type 
    # primary_type= [int, np.uint8, float, np.float32, np.float64, np.ndarray,tuple, bool, str, type(None), list]
    new_ls = [] 
    for element in ls:
        if type(element) in primary_type: 
            if type(element) is list:
                new_ls.append(restore_list_attr(element))
            else: new_ls.append(element) 
        else: 
            if type(element) is dict: 
                if element['name_type_object'].lower() == 'dict': 
                    new_ls.append(restore_dict_attr(element))
                else:
                    new_ls.append(restore_object(element))
    return new_ls

def restore_dict_attr(diction: dict): 
    new_dict = {}
    global primary_type
    # primary_type= [int, np.uint8, float, np.float32, np.float64, np.ndarray,tuple, bool, str, type(None), list]
    for key, value in diction.items():
        if type(value) is list:
            new_dict[key]= restore_list_attr(value) 
            pass 
        elif type(value) is dict: 
            if value['name_type_object'].lower() == 'dict': 
                new_dict[key] = restore_dict_attr(value)
            else: 
                new_dict[key] = restore_object(value) 
        elif type(value) in primary_type: 
            new_dict[key] = value 
        else: 
            error = f"{key}: {value} cannot assigned"
            if error not in ls_error:
                if PRINT_ERROR: print(error)
                ls_error.append(error)

def assign_attribute(model, attri_diction: dict):
    global primary_type
    for key, value in attri_diction.items():
        if type(value) is list: 
            # model.key = process_list_attr(value)
            setattr(model, key, restore_list_attr(value))
        elif type(value) is dict: 
            if value['name_type_object'].lower() == 'dict': 
                setattr(model, key, restore_dict_attr(value))
            else:
                setattr(model, key, restore_object(value))
        elif type(value) in primary_type: 
            setattr(model, key, (value))
            # model.key = value 
        else: 
            error = f"{key}: {value} ||| {type(value)} cannot assigned"
            if error not in ls_error:
                if PRINT_ERROR: print(error)
                ls_error.append(error)
        
    return model 

def restore_object(diction):
    if type(diction) is not dict: 
        return None 
    if diction['name_type_object'].lower() == 'dict':
        return restore_dict_attr(diction)
    # print(diction)
    if diction['name_type_object'] == 'function':
        return None 
    try:
        if len(diction['name_type_object'].split(".")) == 1: 
            if '__module__' not in diction.keys():
                model = classname_to_class(diction['name_type_object'])
            else: 
                try: 
                    model = classname_to_class(diction['name_type_object'])
                except:
                    try:
                        model = module_class_str_to_class(diction['__module__'].split(".")[-1], diction['name_type_object'])
                    except:
                        error = (f"Cannot restore {diction['name_type_object']}")
                        if error not in ls_error:
                            if PRINT_ERROR: print(error)
                            ls_error.append(error)
                        return None 
        else: 
            module, classname = diction['name_type_object'].split(".")[-2:]

            model = module_class_str_to_class(module, classname) 
    except :
        class Temp: 
            def __init__(self) -> None:
                pass
        model = Temp 
        error = f"Error when create object {diction['name_type_object']}. Using Temp class instead"
        if error not in ls_error:
            if PRINT_ERROR: print(error)
            ls_error.append(error)

        
    if '__module__' in diction.keys():
        return model 

    # return model 
    init_parameters = {} 
    signature = inspect.signature(model.__init__).parameters 
    for name, parameter in signature.items(): 
        if name == 'self': continue 
        if name in diction.keys(): 
            # print(name)
            if type(diction[name]) is dict: 
                if diction[name]['name_type_object'].lower() == 'dict':
                    init_parameters[name] = restore_dict_attr(diction[name])
                else: 
                    init_parameters[name] = restore_object(diction[name])
            elif type(diction[name]) is list: 
                init_parameters[name] = restore_list_attr(diction[name])
            elif type(diction[name]) in primary_type:
                init_parameters[name] = diction[name] 
            else: 
                error = (f'Cannot convert {diction[name]} to create init parameters')
                if error not in ls_error:
                    if PRINT_ERROR: print(error)
                    ls_error.append(error)
                
                init_parameters[name]= None 
            # print(init_parameters[name])
            # print()
        else: 
            # print(parameter.default)
            init_parameters[name] = parameter.default
    try:
        instance_model = model(**init_parameters)
    except :
        # if e not in ls_error:
        #     print(e) 
        #     ls_error.append(e)
        # print()
        error = (f"Error while create instance model for {model}. Using Temp class instead")
        if error not in ls_error:
            print(error)
            ls_error.append(error)
        class Temp: 
            def __init__(self) -> None:
                pass 
        instance_model = Temp() 

    new_model = assign_attribute(instance_model, diction)
    return new_model 
