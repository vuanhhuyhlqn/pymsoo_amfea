import numpy as np
import random
from . import AbstractModel
from ...utils import Crossover, Mutation, Selection, Rmp
from ...utils.EA import *
from ...utils.numba_utils import numba_randomchoice

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover,
        mutation: Mutation.PolynomialMutation,  
        selection: Selection.ElitismSelection, 
        *args, **kwargs):
        super().compile(
            IndClass, tasks, 
            crossover= crossover,
            mutation= mutation,
            selection= selection,
            *args, **kwargs
        )
        self.rmp = Rmp.AdaptiveRMP(5, 5, 0.8, 0.1, crossover, mutation)

    
    def get_pop_distribution(self, population: Population):
        mean = np.zeros((population.nb_tasks, population.dim_uss))
        var = np.zeros((population.nb_tasks, population.dim_uss))
        for sub_pop in population:
            mean[sub_pop.skill_factor] = np.mean([ind.genes for ind in sub_pop], axis= 0)
            var[sub_pop.skill_factor] = np.var([ind.genes for ind in sub_pop], axis=0)
        return mean, var 

    def get_llm_rmp(self, population: Population, gen):
        mean, variance = self.get_pop_distribution(population)
        collect_state = {
            "task_count": population.nb_tasks,
            "pop_mean": mean,
            "pop_variance": variance
        }
        p1_inds = population.__getRandomInds__(population.nb_tasks * 50)
        p2_inds = population.__getRandomInds__(population.nb_tasks * 50)
        p1_genes = [ind.genes for ind in p1_inds]
        p2_genes = [ind.genes for ind in p2_inds]
        p1_skill_factor = [ind.skill_factor for ind in p1_inds]
        p2_skill_factor = [ind.skill_factor for ind in p2_inds]
        p1_fitness = [ind.fcost for ind in p1_inds]
        p2_fitness = [ind.fcost for ind in p2_inds]
        return self.rmp(collect_state,
                        p1_genes,
                        p2_genes,
                        p1_skill_factor,
                        p2_skill_factor,
                        p1_fitness,
                        p2_fitness,
                        gen,
                        self.llm_rate,
                        self.tasks
                        )
        return np.random.choice([0.2, 0.3, 0.4, 0.5], size=(len(self.tasks), len(self.tasks)))

    def fit(self, nb_generations, nb_inds_each_task = 100, llm_rate=100, evaluate_initial_skillFactor = True, *args, **kwargs) -> List[Individual]:
        super().fit(*args, **kwargs)
        self.llm_rate = llm_rate
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
            rmp = self.get_llm_rmp(population, epoch)
            # print("RMP")
            # print(rmp)

            # initial offspring_population of generation
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks = [0] * len(self.tasks), 
                dim = self.dim_uss,
                list_tasks= self.tasks,
            )

            # create offspring pop
            while len(offsprings) < int(len(population) * 1.5):
                # choose parent 
                pa, pb = population.__getRandomInds__(2)
                # print("{0} {1}".format(pa.skill_factor, pb.skill_factor))
                # print("-> {0}".format(rmp[pa.skill_factor][pb.skill_factor]))

                if pa.skill_factor == pb.skill_factor or random.random() < rmp[pa.skill_factor][pb.skill_factor]:
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
