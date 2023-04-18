import numpy as np
from typing import Type, List
from numba import jit

from ..EA import AbstractTask, Individual, Population
from .utils import AbstractDaS

class DaS_strategy(AbstractDaS):
    def __init__(self, eta = 3):
        self.eta = eta

    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        # self.prob = 1 - KL_divergence
        self.prob = np.ones((self.nb_tasks, self.nb_tasks, self.dim_uss))

    @staticmethod
    @jit(nopython= True)
    def _updateProb(prob, u, dim_uss, nb_tasks, mean, std):
        for i in range(nb_tasks):
            for j in range(nb_tasks):
                kl = np.log((std[i] + 1e-50)/(std[j] + 1e-50)) + (std[j] ** 2 + (mean[j] - mean[i]) ** 2)/(2 * std[i] ** 2 + 1e-50) - 1/2
                prob[i][j] = np.exp(-kl * u)

        return np.clip(prob, 1/dim_uss, 1)

    def update(self, population: Population, **kwargs) -> None:
        mean = np.zeros((self.nb_tasks, self.dim_uss))
        std = np.zeros((self.nb_tasks, self.dim_uss))
        for idx_subPop in range(self.nb_tasks):
            mean[idx_subPop] = population[idx_subPop].__meanInds__
            std[idx_subPop] = population[idx_subPop].__stdInds__
        self.prob = self.__class__._updateProb(self.prob, 10**(-self.eta), self.dim_uss, self.nb_tasks, mean, std)

    @staticmethod
    @jit(nopython= True)
    def _dimensionSelection(ind, original_ind, dim_uss, pcd):
        idx_transfer = np.random.rand(dim_uss) < pcd
        if np.all(idx_transfer == 0) or np.all(ind[idx_transfer] == original_ind[idx_transfer]):
            # alway crossover -> new individual
            idx_notsame = np.where(ind != original_ind)[0]
            if len(idx_notsame) == 0:
                idx_transfer = np.ones((dim_uss, ), dtype= np.bool_)
            else:
                idx_transfer[np.random.choice(idx_notsame)] = True
        
        return np.where(idx_transfer, ind, original_ind)

    def __call__(self, ind: Individual, knwl_skf: int, original_ind: Individual):
        tmp = self.__class__._dimensionSelection(ind.genes.copy(), original_ind.genes, self.dim_uss, self.prob[original_ind.skill_factor][knwl_skf])
        ind.genes = tmp
        return ind