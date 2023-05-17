from pathlib import Path 
import pickle 
import os 
import numpy as np 
import pandas as pd 

from .load_utils import loadModel
from pyMSOO.MFEA.benchmark.continous import * 

def saveModel(model, PATH: str, remove_tasks=True, total_time = None ):
    '''
    `.mso`
    '''
    assert model.__class__.__name__ == 'MultiTimeModel'
    assert type(PATH) == str

    # check tail
    path_tmp = Path(PATH)
    index_dot = None
    for i in range(len(path_tmp.name) - 1, -1, -1):
        if path_tmp.name[i] == '.':
            index_dot = i
            break

    if index_dot is None:
        PATH += '.mso'
    else:
        assert path_tmp.name[i:] == '.mso', 'Only save model with .mso, not ' + \
            path_tmp.name[i:]

    tasks = model.tasks 

    if total_time is not None: 
        model.total_time = total_time 

    if remove_tasks is True:
        if hasattr(model, "tasks"):
            model.tasks = None
        if hasattr(model, "compile_kwargs"):
            model.compile_kwargs['tasks'] = None
            for key, value in model.compile_kwargs.items():
                if hasattr(model.compile_kwargs[key], 'tasks'):
                    setattr(model.compile_kwargs[key], 'tasks', None)
        for submodel in model.ls_model:
            if hasattr(submodel, "tasks"):
                submodel.tasks = None
            if hasattr(submodel, "last_pop"):
                submodel.last_pop.ls_tasks = None
            if hasattr(submodel, "last_pop"):
                for subpop in submodel.last_pop:
                    subpop.task = None
            if hasattr(submodel, "kwargs"):
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        if hasattr(submodel, attribute): 
                            setattr(getattr(submodel, attribute), 'tasks', None)
                        pass
                else:
                    submodel.crossover.tasks = None
                    submodel.mutation.tasks = None

    try:
        f = open(PATH, 'wb')
        pickle.dump(model, f)
        f.close()

    except:
        cls = model.__class__
        model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

        if remove_tasks is True:
            if hasattr(model, "tasks"):
                model.tasks = tasks
            if hasattr(model, "compile_kwargs"): 
                model.compile_kwargs['tasks'] = None
            for submodel in model.ls_model:
                if hasattr(submodel, "tasks"):
                    submodel.tasks = tasks
                if hasattr(submodel, "last_pop"):
                    if hasattr(submodel.last_pop, "ls_tasks"):
                        submodel.last_pop.ls_tasks = tasks 
                    for idx, subpop in enumerate(submodel.last_pop):
                        if hasattr(subpop, "task"):
                            subpop.task = tasks[idx]
                if hasattr(submodel, "kwargs"):
                    if 'attr_tasks' in submodel.kwargs.keys():
                        for attribute in submodel.kwargs['attr_tasks']:
                            # setattr(submodel, getattr(subm, name), None)
                            setattr(getattr(submodel, attribute), 'tasks', tasks)
                            pass
                    else:
                        submodel.crossover.tasks = tasks
                        submodel.mutation.tasks = tasks 
        return 'Cannot Saved'

    
    if remove_tasks is True:
        if hasattr(model, "tasks"):
            model.tasks = tasks
        if hasattr(model, "compile_kwargs"): 
            model.compile_kwargs['tasks'] = None
        for submodel in model.ls_model:
            if hasattr(submodel, "tasks"):
                submodel.tasks = tasks
            if hasattr(submodel, "last_pop"):
                if hasattr(submodel.last_pop, "ls_tasks"):
                    submodel.last_pop.ls_tasks = tasks 
                
                for idx, subpop in enumerate(submodel.last_pop):
                    if hasattr(subpop, "task"):
                        subpop.task = tasks[idx]
            if hasattr(submodel, "kwargs"):
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        if hasattr(submodel, attribute): 
                            setattr(getattr(submodel, attribute), 'tasks', tasks)
                        pass
                else:
                    submodel.crossover.tasks = tasks
                    submodel.mutation.tasks = tasks 

    cls = model.__class__
    model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

    return 'Saved'

def export_history2cec_format(save_path= "./Complex/", \
                source_path = "./RESULTS/COMPLEX/SMP_v2/", \
                prefix_name = "MTOSOO_P",\
                ls_tasks =  WCCI22_benchmark.get_complex_benchmark(1)[0],
                total_evals = int(100000 * 2), 
                steps = 1000, 
                load_func = loadModel
                ):
    '''
    Export history cost of a all files in a folder to .txt files that has format of CEC competition
    '''
    for each_name_model in os.listdir(os.path.join(source_path)):
        model = load_func(os.path.join(source_path, each_name_model), ls_tasks) 
        
        tmp = np.concatenate([model.ls_model[i].history_cost for i in range(len(model.ls_model))], axis=1)
        stt = np.arange(total_evals // steps, total_evals + total_evals // steps, total_evals // steps)
        # stt = stt.reshape(1,int((200000)/ 2000))
        stt = stt.reshape(1, -1) 
        tmp = tmp.T
        if tmp.shape[0] != stt.shape[1]:
            tmp = tmp[:, -stt.shape[1]:]
        
        assert tmp.shape[1] == stt.shape[1], print(tmp.shape)
        tmp4= np.concatenate([stt, tmp], axis=0).T

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        f = open(save_path + prefix_name+ each_name_model.split(".")[0] + ".txt", "w")
        for line in tmp4: 
            f.write(", ".join([str(int(line[0]))] + [str('{:.6f}'.format(e)) for e in line[1:]]) + "\n")
        f.close()
    

def export_solution2csv(folder_path,save_name_csv= "history_cost_summaries.csv", name= [str(i) + ".mso" for i in range(1, 11)], min_value= 1e-6): 
    '''
    export solution of algorithms to csv file with each column correspond an algorithm
    
    Arg: 
        folder_path: folder contain algorithms folder
                    Ex: Result/ contain three algorithms MFEA, SM, LSA 
                    Ex: Result/ contains MFEA/, SM/, LSA/ 
                    Ex:  MFEA/, SM/, LSA/ contain file .mso 
        name: name file mso in each algorithms folder. They must be same. 
        min_value: if "result" is smaller min_value => change to 0 
    '''
    history_cost_summaries = []
    for model_lib in os.listdir(folder_path): 
        print(model_lib)
        tmp = []
        # for id in os.listdir("RESULTS/SM_MFEA_DaS/WCCI22/SM_MFEA_DaS/"):
        for id in name:
            print("id: ", id)
            model = loadModel(os.path.join(os.path.join(folder_path, model_lib), id), ls_tasks= WCCI22_benchmark.get_50tasks_benchmark(1)[0])
            tmp += np.array(model.history_cost[-1]).tolist() 
        print(len(tmp))
        history_cost_summaries.append(tmp.copy())
        

    ls_col_name = os.listdir(folder_path)
    history_cost_summaries = np.array(history_cost_summaries)
    history_cost_summaries = np.where(history_cost_summaries < min_value, 0, history_cost_summaries)
    print(history_cost_summaries.T.shape)
    df = pd.DataFrame(history_cost_summaries.T,columns= ls_col_name)
    pd.DataFrame.to_csv(df, save_name_csv)
    pass 