# -------Abstract Crossover----------
import numpy as np
from typing import Tuple, Type, List
from ..EA import AbstractTask, Individual, Population

class AbstractDaS():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, ind: Individual, knwl_skf: int, original_ind: Individual, *args, **kwargs) -> Individual:
        pass

    def update(self, population: Population, **kwargs) -> None:
        pass

    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])
        self.nb_tasks = len(tasks)
        self.tasks = tasks
        self.IndClass = IndClass
        #seed
        np.random.seed(seed)
        pass
    
    def update(self, *args, **kwargs) -> None:
        pass

class NoDaS(AbstractDaS):
    def __call__(self, ind: Individual, knwl_skf: int, original_ind: Individual, *args, **kwargs) -> Individual:
        return ind