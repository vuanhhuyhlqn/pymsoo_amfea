import numpy as np
from typing import Tuple, Type, List

from ..EA import AbstractTask, Individual, Population
from numba import jit
from .utils import AbstractCrossover

class SBX_LSA21(AbstractCrossover): 
    def __init__(self, nc=2, default_rmp = 0.5, *args, **kwargs): 
        self.nc = nc
        self.best_partner = None 
        self.default_rmp = default_rmp 
        self.C =  0.02 

        pass 

    def getInforTasks(self,  IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)

        self.rmp = np.zeros(shape=(self.nb_tasks, self.nb_tasks)) + self.default_rmp
        self.best_partner = np.zeros(shape= (self.nb_tasks), dtype= int) - 1 

        self.s_rmp = np.empty(shape= (self.nb_tasks, self.nb_tasks,0)).tolist()
        self.diff_f_inter_x = np.empty(shape=(self.nb_tasks, self.nb_tasks,0)).tolist() 

    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        u = np.random.rand(self.dim_uss)

        # ~1 TODO
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))

        #like pa
        oa = self.IndClass(np.clip(0.5*((1 + beta) * pa.genes + (1 - beta) * pb.genes), 0, 1))
        #like pb
        ob = self.IndClass(np.clip(0.5*((1 - beta) * pa.genes + (1 + beta) * pb.genes), 0, 1))


        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob
    
    def update(self, *args, **kwargs) -> None:
        for task in range(self.nb_tasks):
            maxRmp = 0 
            self.best_partner[task] = -1 
            for task2 in range(self.nb_tasks): 
                if task2 == task: 
                    continue 
                
                good_mean = 0 
                if len(self.s_rmp[task][task2]) > 0: 
                    sum = np.sum(np.array(self.diff_f_inter_x[task][task2])) 

                    w = np.array(self.diff_f_inter_x[task][task2]) / sum 

                    val1 =  np.sum(w * np.array(self.s_rmp[task][task2]) ** 2) 
                    val2 = np.sum(w * np.array(self.s_rmp[task][task2])) 

                    good_mean = val1 / val2 

                    if (good_mean > self.rmp[task][task2] and good_mean > maxRmp): 
                        maxRmp = good_mean 
                        self.best_partner[task] = task2 
                    
                
                if good_mean > 0: 
                    c1 = 1.0 
                else: 
                    c1 = 1.0 - self.C 
                self.rmp[task][task2] = c1 * self.rmp[task][task2] + self.C * good_mean 
                self.rmp[task][task2] = np.max([0.01, np.min([1, self.rmp[task][task2]])])

        self.s_rmp = np.empty(shape= (self.nb_tasks, self.nb_tasks,0)).tolist()
        self.diff_f_inter_x = np.empty(shape=(self.nb_tasks, self.nb_tasks,0)).tolist() 
