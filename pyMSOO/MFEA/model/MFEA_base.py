import numpy as np
import random
from . import AbstractModel
from ...utils import Crossover, Mutation, Selection
from ...utils.EA import *
from ...utils.numba_utils import numba_randomchoice

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.PolynomialMutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        super().compile(
            IndClass, tasks, 
            crossover= crossover,
            mutation= mutation,
            selection= selection,
            *args, **kwargs
        )
    
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

        # save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys= True)

        for epoch in range(nb_generations):
            
            # initial offspring_population of generation
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )

            # create offspring pop
            while len(offsprings) < len(population):
                # choose parent 
                pa, pb = population.__getRandomInds__(2)

                if pa.skill_factor == pb.skill_factor or random.random() < rmp:
                    # intra / inter crossover
                    skf_oa, skf_ob = numba_randomchoice(np.array([pa.skill_factor, pb.skill_factor]), size= 2, replace= True)
                    oa, ob = self.crossover(pa, pb, skf_oa, skf_ob)

                    # dimension strategy
                    p_of_oa, knwl_oa = (pa, pb.skill_factor) if pa.skill_factor == skf_oa else (pb, pa.skill_factor)
                    p_of_ob, knwl_ob = (pa, pb.skill_factor) if pa.skill_factor == skf_oa else (pb, pa.skill_factor)

                    oa = self.dimension_strategy(oa, knwl_oa, p_of_oa)
                    ob = self.dimension_strategy(ob, knwl_ob, p_of_ob)
                
                else:
                    # mutate
                    oa = self.mutation(pa, return_newInd= True)
                    oa.skill_factor = pa.skill_factor

                    ob = self.mutation(pb, return_newInd= True)    
                    ob.skill_factor = pb.skill_factor
                
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob)
            # print("len off : {0}".format(len(offsprings)))
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
