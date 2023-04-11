import pickle 
from pathlib import Path 

from ...MFEA.model import * 
from ...MFEA.competitionModel import * 
from pyMSOO.utils.Crossover import *
from pyMSOO.utils.Mutation import *
from pyMSOO.utils.Selection import *
from pyMSOO.utils.Search import * 
from pyMSOO.MFEA.benchmark.continous import *
from pyMSOO.utils.MultiRun.RunMultiTime import * 

from pyMSOO.utils.EA import * 
from pyMSOO.MFEA.benchmark.continous.CEC17 import CEC17_benchmark 
from pyMSOO.MFEA.benchmark.continous.WCCI22 import WCCI22_benchmark
from pyMSOO.MFEA.benchmark.continous.utils import Individual_func 
from pyMSOO.MFEA.benchmark.continous.funcs import * 

import inspect
import sys 
import numpy as np 
import pandas as pd 


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

def loadModel(PATH: str, ls_tasks=None, set_attribute=False, mso_orginal= False) -> AbstractModel:
    '''
    `.mso`
    '''
    if mso_orginal: 
        assert type(PATH) == str

        # check tail
        path_tmp = Path(PATH)
        index_dot = None
        for i in range(len(path_tmp.name)):
            if path_tmp.name[i] == '.':
                index_dot = i
                break

        if index_dot is None:
            PATH += '.mso'
        else:
            assert path_tmp.name[i:] == '.mso', 'Only load model with .mso, not ' + \
                path_tmp.name[i:]

        f = open(PATH, 'rb')
        model = pickle.load(f)
        f.close()

        cls = model.__class__
        model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

        if model.tasks is None:
            model.tasks = ls_tasks
            if set_attribute is True:
                assert ls_tasks is not None, 'Pass ls_tasks plz!'
                model.compile_kwargs['tasks'] = ls_tasks
                for submodel in model.ls_model:
                    submodel.tasks = ls_tasks
                    submodel.last_pop.ls_tasks = ls_tasks
                    for idx, subpop in enumerate(submodel.last_pop):
                        subpop.task = ls_tasks[idx]
                    if 'attr_tasks' in submodel.kwargs.keys():
                        for attribute in submodel.kwargs['attr_tasks']:
                            # setattr(submodel, getattr(subm, name), None)
                            setattr(getattr(submodel, attribute),
                                    'tasks', ls_tasks)
                            pass
                    else:
                        submodel.crossover.tasks = ls_tasks
                        submodel.mutation.tasks = ls_tasks

                    # submodel.search.tasks = ls_tasks
                    # submodel.crossover.tasks = ls_tasks
                    # submodel.mutation.tasks = ls_tasks

        if model.name.split('.')[-1] == 'AbstractModel':
            model.name = path_tmp.name.split('.')[0]
        return model

    else: 
        with open(PATH, 'rb') as file: 
            model_dict = pickle.load(file) 
        
        return restore_object(model_dict) 



def loadModelFromTxt(source_path, model, target_path = "./", remove_tasks: bool = False, history_cost_shape= (1000, 2), nb_runs = 1, ls_tasks = [], name_model = None, total_time= None ):
    '''
    File txt has the format of MTO competition.

    Load result from file txt and save it to .mso file
    
    Args: 
        remove_tasks: Do remove tasks when save model or not ? 
        history_cost_shape: The history cost shape in one run. 
        nb_runs: the number run of model 
        ls_tasks: list of tasks
        name_model (optional): that will be use as name to save the model 
    
    Results: 
        Save multitime model file 
    '''

    assert len(ls_tasks) > 0 
    data = pd.read_csv(source_path, header= None) 
    # data = pd.read_csv(source_path, header= None, delim_whitespace= True).astype("float") 
    history_cost_shape = (data.shape[0], (data.shape[1] - 1) // nb_runs)
    
    history_cost = np.zeros(shape = (nb_runs,history_cost_shape[0], history_cost_shape[1]))
    data_transpose = data.transpose() 
    count_row = 1
    for i_run in range(nb_runs): 
        for idx_task in range(len(ls_tasks)):
            history_cost[i_run, :,idx_task] = data_transpose.iloc[count_row, :]
            count_row += 1  
    
    avg_history_cost = np.average(history_cost, axis = 0) 
    assert avg_history_cost.shape == history_cost_shape 


    mutiltime_model = MultiTimeModel(model) 
    mutiltime_model.compile()
    mutiltime_model.tasks = None 
    mutiltime_model.history_cost = avg_history_cost
    mutiltime_model.nb_run = nb_runs 

    for run in range(nb_runs): 
        new_model = model.model() 
        new_model.history_cost = history_cost[run] 
        mutiltime_model.ls_model.append(new_model)
    if name_model is None:
        name_model = source_path.split("/")[-1].split(".")[0]
    
    if os.path.isdir(target_path) is False: 
        os.makedirs(target_path) 
    
    return mutiltime_model
    # return saveModel(model= mutiltime_model, PATH= f"{target_path}/{name_model}.mso", remove_tasks= remove_tasks, total_time= total_time) 
