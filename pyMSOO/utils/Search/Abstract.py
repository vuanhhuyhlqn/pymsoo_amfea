from typing import Tuple, Type, List
import numpy as np
from ..EA import AbstractTask, Individual, Population

class AbstractSearch():
    def __init__(self) -> None:
        pass
    def __call__(self, *args, **kwargs) -> Individual:
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
