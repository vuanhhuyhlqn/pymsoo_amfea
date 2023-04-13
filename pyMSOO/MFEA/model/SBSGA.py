import numpy as np
import random

from . import AbstractModel
from ...utils import Crossover, Mutation, Selection, DimensionAwareStrategy
from ...utils.EA import *

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
    
    def fit(self, nb_generations, B = 0.25, H = 0.5, nb_inds_each_task = 100, evaluate_initial_skillFactor = True, *args, **kwargs) -> List[Individual]:
        super().fit(*args, **kwargs)

        # initialize population
        self.population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )
        self.B = B
        self.H = H

        self.M = np.ones((len(self.tasks), len(self.tasks)))
        self.N = np.ones((len(self.tasks), len(self.tasks)))
        self.C = np.ones((len(self.tasks), len(self.tasks)))
        self.O = np.ones((len(self.tasks), len(self.tasks)))
        self.P = np.ones((len(self.tasks), len(self.tasks)))
        self.A = np.ones((len(self.tasks), len(self.tasks)))
        self.R = np.ones((len(self.tasks), len(self.tasks)))
        
        # save history
        self.history_cost.append([ind.fcost for ind in self.population.get_solves()])
        
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys= True)

        for epoch in range(nb_generations):

            offsprings = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )

            copy_offsprings = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )

            subPop = [0] * len(self.tasks)
            other_task = [0] * len(self.tasks)

            # Generate offsprings
            for i in range(len(self.tasks)):
                offs = Population(
                    self.IndClass,
                    nb_inds_tasks = [0] * len(self.tasks), 
                    dim = self.dim_uss,
                    list_tasks= self.tasks,
                )
                while len(offs) < nb_inds_each_task:
                    # choose parent 
                    pa, pb = self.population[i].__getRandomItems__(2)
                    # intra crossover
                    oa, ob = self.crossover(pa, pb, i, i)
                    # mutate
                    oa = self.mutation(oa, return_newInd= True)
                    oa.skill_factor = pa.skill_factor

                    ob = self.mutation(ob, return_newInd= True)    
                    ob.skill_factor = pb.skill_factor
                    
                    offs.__addIndividual__(oa)
                    offs.__addIndividual__(ob)

                offsprings += offs
                copy_offsprings += offs
            
            # Inter task

            # np.fill_diagonal(self.R, -1)
            # R_i = np.max(self.R, axis = 1)
            # task_i = list(np.where(np.random.rand() < R_i)[0])
            # task_j = list(np.argmax(self.R[task_i], axis = 1))
            # # S_i = np.min(np.max((R_i[t] * nb_inds_each_task).astype(int), 0), nb_inds_each_task)
            # S_i = np.clip((R_i[task_i] * nb_inds_each_task).astype(int), 0, nb_inds_each_task)
            # # print(task_j)
            # # print(S_i)
            # S_i = np.clip((R_i[task_i] * nb_inds_each_task).astype(int), 0, nb_inds_each_task)
            # # subPop[t] = [self.IndClass(off.genes, skill_factor=i, fcost=t(off.genes)) for off in copy_offsprings[task_j][0: S_i]]
            # # subPop = [[self.IndClass(off.genes, skill_factor=i, fcost=self.tasks[i](off.genes)) for off in copy_offsprings[i][0: S_i[i]]] for i in task_j]
            # # offsprings[t][nb_inds_each_task - S_i: nb_inds_each_task] = subPop[t]
            # for i, j in zip(task_i, task_j):
            #     subPop[i] = [self.IndClass(off.genes, skill_factor=i, fcost=self.tasks[i](off.genes)) for off in copy_offsprings[j][0: S_i[i]]]
            #     offsprings[i][nb_inds_each_task - S_i[i]: nb_inds_each_task] = subPop[i]

            for i, t in enumerate(self.tasks):
                self.R[i][i] = -1
                R_i = max(self.R[i])
                if random.random() < R_i:
                    current= self.population[i].__getRandomItems__()
                    task_j = np.argmax(self.R[i])
                    S_i = min(max(int(R_i * nb_inds_each_task), 0), nb_inds_each_task)
                    other_task[i] = task_j
                    
                    subPop[i] = [self.dimension_strategy(self.IndClass(off.genes, skill_factor=i, fcost=t(off.genes)), task_j, current) 
                                 for off in copy_offsprings[task_j][0: S_i]]
                    # subPop[i] = []
                    offsprings[i][nb_inds_each_task - S_i: nb_inds_each_task] = subPop[i]

            # merge and update rank
            self.population = self.population + offsprings
            self.population.update_rank()

            # selection
            self.selection(self.population, [nb_inds_each_task] * len(self.tasks))

            # update symbiosis and rate
            self.update_symbiosis(subPop, other_task)
            # self.update_rate()
            self.R = self.__class__.update_rate(self.M, self.O, self.P, self.A, self.C, len(self.tasks))

            # update operators
            self.crossover.update(population = self.population)
            self.mutation.update(population = self.population)
            self.dimension_strategy.update(population = self.population)

            # save history
            self.history_cost.append([ind.fcost for ind in self.population.get_solves()])

            #print
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        
        print('\nEND!')

        #solve 
        self.last_pop = self.population
        return self.last_pop.get_solves()

    # def update_rate(self):
    #     for t in range(len(self.tasks)):
    #         T_pos = self.M[t] + self.O[t] + self.P[t]
    #         T_neg = self.A[t] + self.C[t]
    #         T_neu = self.M[t]
    #         self.R[t] = T_pos/(T_pos + T_neg + T_neu)
    #         self.R[t][t] = -1

    @jit(nopython = True)
    def update_rate(M, O, P, A, C, nb_tasks):
        R = np.zeros((nb_tasks, nb_tasks))
        for t in range(nb_tasks):
            T_pos = M[t] + O[t] + P[t]
            T_neg = A[t] + C[t]
            T_neu = M[t]
            R[t] = T_pos/(T_pos + T_neg + T_neu)
            R[t][t] = -1
        return R

    def update_symbiosis(self, subpop: List[Individual], other_task):
        for i in range(len(self.tasks)):
            if isinstance(subpop[i], list):
                for o in subpop[i]:
                    j = other_task[i]
                    if i != j:
                        r_i = self.rank(o, i)
                        r_j = self.rank(o, j)
                        # r_i = self.__class__.rank(self.tasks[i](o), nb.typed.List(self.population[i].getFitness()))
                        # r_j = self.__class__.rank(self.tasks[j](o), nb.typed.List(self.population[j].getFitness()))
                        if (self.isBenefit(r_i) and self.isBenefit(r_j)):
                            self.M[i][j] += 1
                        elif (self.isNeural(r_i) and self.isNeural(r_j)):
                            self.N[i][j] += 1
                        elif (self.isHarmful(r_i) and self.isHarmful(r_j)):
                            self.C[i][j] += 1
                        elif (self.isBenefit(r_i) and self.isNeural(r_j)):
                            self.O[i][j] += 1
                        elif (self.isBenefit(r_i) and self.isHarmful(r_j)): 
                            self.P[i][j] += 1
                        elif (self.isNeural(r_i) and self.isHarmful(r_j)):
                            self.A[i][j] += 1

    def rank(self, off: Individual, task):
        fitness = self.tasks[task](off)
        rank = np.searchsorted(self.population[task].getFitness(), fitness)
        return rank/len(self.population[task])
    
    # @jit(nopython = True)
    # def rank(off_fitness, subPop_fitness):
    #     # fitness = self.tasks[task](off)
    #     # print(self.population[task].getFitness())
    #     rank = np.searchsorted(subPop_fitness, off_fitness)
    #     return rank/len(subPop_fitness)
    
    def isBenefit(self, r):
        if (r <= self.B):
            return True
        return False

    def isHarmful(self, r) :
        if (r > self.H) :
            return True
        return False

    def isNeural(self, r) :
        return (r > self.B and r <= self.H)