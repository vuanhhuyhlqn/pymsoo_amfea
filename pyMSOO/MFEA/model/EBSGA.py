import numpy as np
import random
from functools import reduce
import time

from . import AbstractModel
from ...utils import Crossover, Mutation, Selection, DimensionAwareStrategy
from ...utils.EA import *
from ...utils.numba_utils import numba_randomchoice

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, 
        mutation: Mutation.PolynomialMutation, 
        selection: Selection.ElitismSelection, 
        dimension_strategy: DimensionAwareStrategy.AbstractDaS = DimensionAwareStrategy.NoDaS(),
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation,dimension_strategy, selection, *args, **kwargs)
    
    def fit(self, nb_generations, rmp = 0.3, nb_inds_each_task = 100, evaluate_initial_skillFactor = True, *args, **kwargs) -> List[Individual]:
        super().fit(*args, **kwargs)

        # initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        self.R_o = np.random.rand((len(self.tasks))) 
        self.R_s = np.random.rand((len(self.tasks)))
        self.E_o = np.random.rand((len(self.tasks)))
        self.E_s = np.random.rand((len(self.tasks)))

        # self.R_o = np.zeros((len(self.tasks))) 
        # self.R_s = np.zeros((len(self.tasks)))
        # self.E_o = np.zeros((len(self.tasks)))
        # self.E_s = np.zeros((len(self.tasks)))

        # save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys= True)

        for epoch in range(nb_generations):
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )

            lambda_i = (self.R_o/(self.E_o + 1e-6))/(self.R_s/(self.E_s + 1e-6) + self.R_o/(self.E_o + 1e-6) + 1e-6)
            idx_other = np.where(np.random.rand(len(self.tasks)) < lambda_i)[0]
            idx_same = list(set(range(len(self.tasks))) - set(idx_other))

            self.E_o[idx_other] += nb_inds_each_task
            self.E_s[idx_same] += nb_inds_each_task
            
            # start = time.time()

            for i in idx_other:
                offs = population.__getRandomInds__(nb_inds_each_task)
                current_best = population[i].__getBestIndividual__.fcost
                for o in offs:
                    j = o.skill_factor
                    off = self.dimension_strategy(o, j, population[i].__getRandomItems__())
                    fcost = self.tasks[i](off)
                    self.R_o[i] += fcost < current_best
                    offsprings.__addIndividual__(self.IndClass(genes = off.genes, skill_factor = i))

            for i in idx_same:
                current_best = population[i].__getBestIndividual__.fcost
                offs = []
                while len(offs) < nb_inds_each_task:
                    # choose parent 
                    pa, pb = population[i].__getRandomItems__(2)
                    # intra / inter crossover
                    oa, ob = self.crossover(pa, pb, i, i)
                    # mutate
                    oa = self.mutation(oa, return_newInd= True)
                    oa.skill_factor = pa.skill_factor

                    ob = self.mutation(ob, return_newInd= True)    
                    ob.skill_factor = pb.skill_factor
                    
                    offsprings.__addIndividual__(oa)
                    offsprings.__addIndividual__(ob)

                    # self.R_s[i] += self.tasks[i](oa) > current_best
                    # self.R_s[i] += self.tasks[i](ob) > current_best

                    # cnt += 1
                    offs.append(oa)
                    offs.append(ob)

                res = np.fromiter(map(self.tasks[i].__call__, offs), float)
                self.R_s[i] += sum(res < population[i].__getBestIndividual__.fcost)      
            
            # end = time.time()
            # print("B: ", end - start)

            # merge and update rank
            population = population + offsprings
            population.update_rank()

            # selection
            self.selection(population, [nb_inds_each_task] * len(self.tasks))

            # update operators
            self.crossover.update(population = population)
            self.mutation.update(population = population)
            self.dimension_strategy.update(population = population)

            # save history
            self.history_cost.append([ind.fcost for ind in population.get_solves()])

            #print
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        
        print('\nEND!')

        #solve 
        self.last_pop = population
        return self.last_pop.get_solves() 