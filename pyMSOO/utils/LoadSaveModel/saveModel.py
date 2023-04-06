import pickle 
from pathlib import Path 

from ...MFEA.model import * 

import inspect
import sys 
import numpy as np 
import os 
# from ..MultiRun.RunMultiTime import MultiTimeModel 

primary_type= [int,np.int32,np.uint8, float, np.float32, np.float64, np.ndarray,tuple, bool, str, list] 

# def saveModel(model, PATH: str, remove_tasks=True, total_time = None ):
#     '''
#     `.mso`
#     '''
#     assert model.__class__.__name__ == 'MultiTimeModel'
#     assert type(PATH) == str

#     # check tail
#     path_tmp = Path(PATH)
#     index_dot = None
#     for i in range(len(path_tmp.name) - 1, -1, -1):
#         if path_tmp.name[i] == '.':
#             index_dot = i
#             break

#     if index_dot is None:
#         PATH += '.mso'
#     else:
#         assert path_tmp.name[i:] == '.mso', 'Only save model with .mso, not ' + \
#             path_tmp.name[i:]

#     # model.__class__ = MultiTimeModel

#     tasks = model.tasks 

#     if total_time is not None: 
#         model.total_time = total_time 

#     if remove_tasks is True:
#         if hasattr(model, "tasks"):
#             model.tasks = None
#         if hasattr(model, "compile_kwargs"):
#             model.compile_kwargs['tasks'] = None
#         for submodel in model.ls_model:
#             if hasattr(submodel, "tasks"):
#                 submodel.tasks = None
#             if hasattr(submodel, "last_pop"):
#                 submodel.last_pop.ls_tasks = None
#             if hasattr(submodel, "last_pop"):
#                 for subpop in submodel.last_pop:
#                     subpop.task = None
#             if hasattr(submodel, "kwargs"):
#                 if 'attr_tasks' in submodel.kwargs.keys():
#                     for attribute in submodel.kwargs['attr_tasks']:
#                         # setattr(submodel, getattr(subm, name), None)
#                         setattr(getattr(submodel, attribute), 'tasks', None)
#                         pass
#                 else:
#                     submodel.crossover.tasks = None
#                     submodel.mutation.tasks = None

#     f = open(PATH, 'wb')
#     pickle.dump(model, f)
#     f.close()
#     try:
#         f = open(PATH, 'wb')
#         pickle.dump(model, f)
#         f.close()

#     except:
#         cls = model.__class__
#         model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

#         if remove_tasks is True:
#             if hasattr(model, "tasks"):
#                 model.tasks = tasks
#             if hasattr(model, "compile_kwargs"): 
#                 model.compile_kwargs['tasks'] = None
#             for submodel in model.ls_model:
#                 if hasattr(submodel, "tasks"):
#                     submodel.tasks = tasks
#                 if hasattr(submodel, "last_pop"):
#                     if hasattr(submodel.last_pop, "ls_tasks"):
#                         submodel.last_pop.ls_tasks = tasks 
#                     for idx, subpop in enumerate(submodel.last_pop):
#                         if hasattr(subpop, "task"):
#                             subpop.task = tasks[idx]
#                 if hasattr(submodel, "kwargs"):
#                     if 'attr_tasks' in submodel.kwargs.keys():
#                         for attribute in submodel.kwargs['attr_tasks']:
#                             # setattr(submodel, getattr(subm, name), None)
#                             setattr(getattr(submodel, attribute), 'tasks', tasks)
#                             pass
#                     else:
#                         submodel.crossover.tasks = tasks
#                         submodel.mutation.tasks = tasks 
#         return 'Cannot Saved'

    
#     if remove_tasks is True:
#         if hasattr(model, "tasks"):
#             model.tasks = tasks
#         if hasattr(model, "compile_kwargs"): 
#             model.compile_kwargs['tasks'] = None
#         for submodel in model.ls_model:
#             if hasattr(submodel, "tasks"):
#                 submodel.tasks = tasks
#             if hasattr(submodel, "last_pop"):
#                 if hasattr(submodel.last_pop, "ls_tasks"):
#                     submodel.last_pop.ls_tasks = tasks 
                
#                 for idx, subpop in enumerate(submodel.last_pop):
#                     if hasattr(subpop, "task"):
#                         subpop.task = tasks[idx]
#             if hasattr(submodel, "kwargs"):
#                 if 'attr_tasks' in submodel.kwargs.keys():
#                     for attribute in submodel.kwargs['attr_tasks']:
#                         # setattr(submodel, getattr(subm, name), None)
#                         setattr(getattr(submodel, attribute), 'tasks', tasks)
#                         pass
#                 else:
#                     submodel.crossover.tasks = tasks
#                     submodel.mutation.tasks = tasks 

#     cls = model.__class__
#     model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

#     return 'Saved'

def saveModel(model, PATH, remove_tasks= True, total_time= None):
    new_model = process_save_object(model, dict())
    
    if total_time is not None:
        new_model['total_time'] = total_time 

    index = -1 
    while PATH[index] != '/':
        index -= 1 
    if os.path.isdir(PATH[0:index]) is False:
        os.makedirs(PATH[0:index])
    
    with open(PATH, 'wb') as file: 
        pickle.dump(new_model, file)
    
    return "Done" 

    

def process_save_list(ls): 
    '''
    Convert list elements
    '''
    global primary_type 
    saved_ls = []
    for element in ls:
        if type(element) not in primary_type: 
            if type(element) is dict or hasattr(element, '__dict__'): 
               saved_ls.append(process_save_object(element, dict())) 

        else:
            if type(element) is list:
                saved_ls.append(process_save_list(element))
            else: saved_ls.append(element) 
    return saved_ls

def process_save_dict(diction: dict, saved_dict = dict()): 
    '''
    Return a dict 
    '''
    saved_dict['name_type_object'] = 'dict' 
    # primary_type= [int, np.uint8, float, np.float32, np.float64, np.ndarray,tuple, bool, str, type(None), list] 
    global primary_type 

    for key, value in diction.items(): 
        if hasattr(value, "__dict__"):
            saved_dict[key] = process_save_object(value, dict()) 
        else:
            if type(value) is list: 
                saved_dict[key] = process_save_list(value)
            elif type(value) is dict: 
                saved_dict[key] = process_save_dict(value, dict())
            elif type(value) in primary_type: 
                saved_dict[key] = value 
            else: 
                if "method" in str(type(value)) or \
                   "function" in str(type(value)) or \
                   "attribute" in str(type(value)): 
                    continue
                if value is None: 
                    saved_dict[key] = None
                print(f"{key} : {value}, {type(value)} cannot saved in process_save_dict function")
                # exit(0) 
                try:
                    saved_dict[key] = str(value)
                except:
                    print(f"Cannot save {key} : {value}; type: {type(value)}")
    return saved_dict 

def process_save_object(model, saved_object = dict()):
    '''
    Return a new dict object  
    '''
    if type(model).__name__ != 'type':
        if type(model).__name__ == 'model':
            saved_object['name_type_object'] = type(model).__module__ + "." + type(model).__name__
            # module_class_str_to_class(type(model).__module__.split(".")[-1],type(model).__name__)
        else:    
            saved_object['name_type_object'] = type(model).__name__
            # classname_to_class(type(model).__name__)
    else:
        if model.__name__ == 'model': 
            saved_object['name_type_object'] = model.__module__ + "." + model.__name__ 
            # module_class_str_to_class((model).__module__.split(".")[-1],(model).__name__)
        else: 
            saved_object['name_type_object'] = model.__name__ 
            # classname_to_class((model).__name__)
    # saved_object['name_type_object'] = type(model).__name__ if type(model).__name__ != "type" else model.__name__
    
    for key, value in model.__dict__.items(): 
        if hasattr(value, "__dict__"):
            saved_object[key] = process_save_object(value, dict())
        else: 
            if type(value) is list: 
                saved_object[key] = process_save_list(value)
            elif type(value) is dict: 
                saved_object[key] = process_save_dict(value, dict())
            elif type(value) in primary_type: 
                saved_object[key] = value 
            else: 
                if  "method" in str(type(value)) or \
                    "function" in str(type(value)) or \
                    "attribute" in str((value)) or \
                    "object" in str((value)): 
                    continue 
                if value is None: 
                    saved_object[key] = None
                 
                # print(f"=={key} : {value}, {type(value)} == cannot saved in function process_save_object")
                try:
                    saved_object[key] = str(value)
                except:
                    print(f"Cannot save {key} : {value}; type: {type(value)} because it cannot convert to string.")
    return saved_object


