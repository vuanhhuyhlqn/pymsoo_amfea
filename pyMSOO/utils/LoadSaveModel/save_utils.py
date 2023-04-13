from ...MFEA.model import * 

import numpy as np 

primary_type= [int,np.int32,np.uint8, float, np.float32, np.float64, np.ndarray,tuple, bool, str, list] 

    

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


