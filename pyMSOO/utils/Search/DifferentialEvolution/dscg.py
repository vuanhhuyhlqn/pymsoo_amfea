from ..Abstract import AbstractSearch
from ...EA import AbstractTask, Individual, Population

from typing import Type, List
import numpy as np

from ...numba_utils import * 

class LocalSearch_DSCG(AbstractSearch):
    def __init__(self) -> None:
        super().__init__() 
        self.INIT_STEP_SIZE= 0.02
        self.EPSILON= 1e-8 
        self.EVALS_PER_LINE_SEARCH= 50 
    
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        return super().getInforTasks(IndClass, tasks, seed)
    
    def update(self, *args, **kwargs) -> None:
        return super().update(*args, **kwargs)
    
    def search(self, start_point: Individual, fes: int, *args, **kwargs) -> Individual:
        s: float = self.INIT_STEP_SIZE 
        evals_per_linesarch= self.EVALS_PER_LINE_SEARCH 

        result = self.IndClass(start_point.genes, dim= self.dim_uss) 
        result.fcost = start_point.fcost 

        x: List[Individual] = [self.IndClass(genes = None, dim= self.dim_uss) for i in range(self.dim_uss + 2)]

        # x.append(self.IndClass(start_point.genes))
        x[0] = self.IndClass(start_point.genes)
        x[0].fcost = start_point.fcost 

        direct = 1
        evals= 0
        v = np.eye(N= self.dim_uss + 1, M = self.dim_uss)

        while True: 
            a = np.zeros(shape= (self.dim_uss, self.dim_uss)) 

            while (evals < fes - evals_per_linesarch) : 
                evals, x[direct] = self.lineSearch(x[direct-1], evals,  self.EVALS_PER_LINE_SEARCH, self.tasks[start_point.skill_factor], s, v[direct-1]) 
                for i in range(1, direct + 1, 1) : 
                    a[i-1] += x[direct].genes - x[direct-1].genes 
                
                if result.fcost > x[direct].fcost : 
                    result.genes = np.copy(x[direct].genes) 
                    result.fcost = np.copy(x[direct].fcost) 
                
                if (direct < self.dim_uss): 
                    direct += 1 
                else: 
                    break 
                pass 

            if evals >= fes or direct < self.dim_uss : 
                break 
            
            z = np.zeros(shape= (self.dim_uss,)) 
            norm_z = 0 

            z = x[self.dim_uss].genes - x[0].genes 
            norm_z = np.sum(z ** 2) 

            norm_z = np.sqrt(norm_z) 

            if (norm_z == 0):
                x[self.dim_uss + 1].genes = np.copy(x[self.dim_uss].genes )
                x[self.dim_uss + 1].fcost = np.copy(x[self.dim_uss].fcost)

                s *= 0.1 
                if (s <= self.EPSILON):
                    break 
                else: 
                    direct = 1 

                    x[0].genes = np.copy(x[self.dim_uss + 1].genes) 
                    x[0].fcost = np.copy(x[self.dim_uss + 1].fcost) 

                    continue 
            else: 
                v[self.dim_uss] = z / norm_z 

                direct = self.dim_uss + 1 

                rest_eval = fes - evals 
                overall_ls_eval =0 
                
                if rest_eval < evals_per_linesarch: 
                    overall_ls_eval = rest_eval 
                else: 
                    overall_ls_eval = evals_per_linesarch 
                evals, x[direct] = self.lineSearch(x[direct-1], evals, overall_ls_eval, self.tasks[start_point.skill_factor], s, v[direct-1])
                if result.fcost > x[direct].fcost: 
                    result.fcost = np.copy(x[direct].fcost) 
                    result.genes = np.copy(x[direct].genes) 
                
                norm_z = 0 
                norm_z = np.sum((x[direct].genes - x[0].genes) ** 2) 
                norm_z = np.sqrt(norm_z)

                if norm_z < s: 
                    s *= 0.1 
                    if s <= self.EPSILON: 
                        break 
                    else: 
                        direct= 1 

                        x[0].genes = np.copy(x[self.dim_uss + 1].genes) 
                        x[0].fcost = np.copy(x[self.dim_uss + 1].fcost) 

                        continue 
                else: 
                    direct = 2 
                    x[0].genes = np.copy(x[self.dim_uss].genes) 
                    x[1].genes = np.copy(x[self.dim_uss + 1].genes)

                    x[0].fcost = np.copy(x[self.dim_uss].fcost) 
                    x[1].fcost = np.copy(x[self.dim_uss + 1].fcost)

                    continue 


        if result.fcost > x[self.dim_uss +1].fcost: 
            return evals, x[self.dim_uss + 1]
        
        return evals, result
    
    def lineSearch(self, start_point: Individual, eval: int,  fes: int, task: AbstractTask, step_size: int, v: np.array) :

        result: Individual = self.IndClass(genes = None, dim = self.dim_uss)

        evals= eval 
        s = step_size 
        change: bool = False 
        interpolation_flag = False 

        x0 = self.IndClass(start_point.genes)
        x0.fcost = np.copy(start_point.fcost)

        x = self.IndClass(x0.genes + s * v) 
        x.fcost = x.eval(task) 
        evals += 1 

        F = np.zeros(shape=(3,))
        interpolation_points = np.zeros(shape=(3, self.dim_uss))


        interpolation_points[0] = np.copy(x0.genes) 
        interpolation_points[1] = np.copy(x.genes)

        F[0] = x0.fcost 
        F[1] = x.fcost 

        if x.fcost > x0.fcost: 
            x.genes = x.genes - 2 * s * v 
            s = -s 
            x.fcost = x.eval(task) 
            evals += 1 

            if x.fcost <= x0.fcost: 
                change= True 
                interpolation_points[0] = np.copy(x0.genes) 
                interpolation_points[1] = np.copy(x.genes) 

                F[0] = x0.fcost 
                F[1] = x.fcost 
            else: 
                change= False 
                interpolation_flag = True 

                interpolation_points[2] = np.copy(interpolation_points[1]) 
                interpolation_points[1] = np.copy(interpolation_points[0]) 
                interpolation_points[0] = np.copy(x.genes) 

                F[2] = F[1] 
                F[1] = F[0] 
                F[0] = x.fcost 
        else: 
            change= True 
        
        while change: 
            s *= 2 

            x0.genes = np.copy(x.genes) 
            x0.fcost = np.copy(x.fcost) 

            x.genes = x0.genes + s * v
            x.fcost = x.eval(task) 
            evals +=1 

            if x.fcost < x0.fcost : 
                interpolation_points[0] = np.copy(x0.genes) 
                interpolation_points[1] = np.copy(x.genes) 

                F[0] = x0.fcost 
                F[1] = x.fcost 

            else: 
                change= False 
                interpolation_flag = True 

                interpolation_points[2] = np.copy(x.genes) 
                F[2] = x.fcost 

                # generate x = x0 + 0.5s 
                s *= 0.5 
                x.genes = x0.genes + s * v 
                x.fcost = x.eval(task) 
                evals += 1 

                if x.fcost > F[1] : 
                    interpolation_points[2] = np.copy(x.genes) 
                    F[2]= x.fcost 
                else: 
                    interpolation_points[0] = np.copy(interpolation_points[1])
                    interpolation_points[1] = np.copy(x.genes)
                    
                    F[0] = F[1] 
                    F[1] = x.fcost 

            if (evals >= fes -2): 
                change = False 


        if (interpolation_flag and ((F[0] - 2 * F[1] + F[2]) != 0)) : 
        
            x.genes = interpolation_points[1] + s * (F[0] - F[2]) / ( 2.0 * (F[0] - 2 * F[1] + F[2])) 
            x.fcost = x.eval(task) 
            evals += 1 

            if x.fcost < F[1] : 
                result.genes = np.copy(x.genes) 
                result.fcost = np.copy(x.fcost) 
            else: 
                result.genes = np.copy(interpolation_points[1]) 
                result.fcost = F[1] 
        else : 
            result.genes = np.copy(interpolation_points[1]) 
            result.fcost = F[1] 
        
        return evals, result 
