from pathlib import Path 
import pickle 



def loadModel(PATH: str, ls_tasks=None, set_attribute=False):
    '''
    `.mso`
    '''
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
