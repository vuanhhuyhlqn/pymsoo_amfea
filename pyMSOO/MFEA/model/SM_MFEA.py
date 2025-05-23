import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import random
import matplotlib 
from ...utils.EA import *
from ...utils import Crossover, Mutation, DimensionAwareStrategy, Selection
from ...utils.Search.DifferentialEvolution.shade import * 
from . import AbstractModel
from ...utils.numba_utils import numba_randomchoice_w_prob

class model(AbstractModel.model):
    class battle_smp:
        def __init__(self, idx_host: int, nb_tasks: int, lr, mu=0.1) -> None:
            assert idx_host < nb_tasks
            self.idx_host = idx_host
            self.nb_tasks = nb_tasks
            self.lr = lr
            self.mu = mu

            #value const for intra
            self.lowerBound_smp = mu/(self.nb_tasks + 1)

            # smp without lowerbound
            self.smp_wo_mu: np.ndarray = np.zeros((nb_tasks + 1, )) + (1 - self.mu)/(nb_tasks + 1)
            self.smp_wo_mu[self.idx_host] += (1 - self.mu) - np.sum(self.smp_wo_mu)

            self.smp_vector = self.smp_wo_mu + self.lowerBound_smp
            
        def get_smp(self) -> np.ndarray:
            return np.copy(self.smp_vector)
        
        def update_SMP(self, Delta_task, count_Delta_tasks):
            '''
            Delta_task > 0 
            '''

            if np.sum(Delta_task) != 0:         
                new_smp = np.array(Delta_task) / (np.array(count_Delta_tasks) + 1e-50)
                new_smp = new_smp * (1 - self.mu) / (np.sum(new_smp) + 1e-50)

                self.smp_wo_mu = self.smp_wo_mu * (1 - self.lr) + new_smp * self.lr
                self.smp_wo_mu[self.idx_host] += (1 - self.mu) - np.sum(self.smp_wo_mu)
                
                self.smp_vector = self.smp_wo_mu + self.lowerBound_smp
            return self.smp_vector
    
    def __init__(self, seed=None, percent_print=2) -> None:
        super().__init__(seed, percent_print)
        self.ls_attr_avg.append('history_smp')

    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, 
        mutation: Mutation.PolynomialMutation, 
        dimension_strategy: DimensionAwareStrategy.AbstractDaS = DimensionAwareStrategy.NoDaS(),
        selection: Selection.AbstractSelection= Selection.ElitismSelection(), 
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation, dimension_strategy, selection, *args, **kwargs)

    # def render_smp(self,  shape = None, title = None, figsize = None, dpi = 100, step = 1, re_fig = False, label_shape= None, label_loc= None):
        
    #     if title is None:
    #         title = self.__class__.__name__
    #     if shape is None:
    #         shape = (int(np.ceil(len(self.tasks) / 3)), 3)
    #     else:
    #         assert shape[0] * shape[1] >= len(self.tasks)

    #     if label_shape is None:
    #         label_shape = (1, len(self.tasks))
    #     else:
    #         assert label_shape[0] * label_shape[1] >= len(self.tasks)

    #     if label_loc is None:
    #         label_loc = 'lower center'

    #     if figsize is None:
    #         figsize = (shape[1]* 6, shape[0] * 5)

    #     fig = plt.figure(figsize= figsize, dpi = dpi)
    #     fig.suptitle(title, size = 15)
    #     fig.set_facecolor("white")
    #     fig.subplots(shape[0], shape[1])

    #     his_smp:np.ndarray = np.copy(self.history_smp)
    #     y_lim = (-0.1, 1.1)

    #     for idx_task, task in enumerate(self.tasks):
    #         fig.axes[idx_task].stackplot(
    #             np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])),
    #             [his_smp[
    #                 np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])), 
    #                 idx_task, t] for t in range(len(self.tasks) + 1)],
    #             labels = ['Task' + str(i + 1) for i in range(len(self.tasks))] + ["mutation"]
    #         )
    #         # plt.legend()
    #         fig.axes[idx_task].set_title('Task ' + str(idx_task + 1) +": " + task.name)
    #         fig.axes[idx_task].set_xlabel('Generations')
    #         fig.axes[idx_task].set_ylabel("SMP")
    #         fig.axes[idx_task].set_ylim(bottom = y_lim[0], top = y_lim[1])


    #     lines, labels = fig.axes[0].get_legend_handles_labels()
    #     fig.tight_layout()
    #     fig.legend(lines, labels, loc = label_loc, ncol = label_shape[1])
    #     plt.show()
    #     if re_fig:
    #         return fig

    
    def render_smp(self,  shape = None, title = None, figsize = None, dpi = 100, step = 1, re_fig = False, 
                label_shape= None, label_loc= None,grid = True, name_tasks= False,
                title_size= None, label_size_x=None, label_size_y= None, pad=None, x_tick_size=None, y_tick_size=None,
                bbox_to_anchor= None, loc_legend= None,borderaxespad= None,handletextpad= 0.8, legend_size=14, 
                ncol= 11
                ):
        self.colors = ["#0000CD","#FF0000","#FF8C00","#FFFF00","#7CFC00","#228B22","#00CED1","#8A2BE2","#FF00FF","#FFF8DC","#B0C4DE","#800000"]
        print(matplotlib.rcParams['pdf.fonttype'])
        if title is None:
            title = self.__class__.__name__
        if shape is None:
            shape = (int(np.ceil(len(self.tasks) / 3)), 3)
        else:
            assert shape[0] * shape[1] >= len(self.tasks)

        if label_shape is None:
            label_shape = (1, len(self.tasks))
        else:
            assert label_shape[0] * label_shape[1] >= len(self.tasks)

        if label_loc is None:
            label_loc = 'lower center'

        if figsize is None:
            figsize = (shape[1]* 6, shape[0] * 6)

        fig = plt.figure(figsize= figsize, dpi = dpi)
        fig.suptitle(title, size = 15)
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        his_smp:np.ndarray = np.copy(self.history_smp)
        y_lim = (-0.1, 1.1)

        for idx_task, task in enumerate(self.tasks):

            stacks = fig.axes[idx_task].stackplot(
                np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])),
                [his_smp[
                    np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])), 
                    idx_task, t] for t in range(len(self.tasks) + 1)],
                labels = ['Task' + str(i + 1) for i in range(len(self.tasks))] + ["mutation"],
                colors = [self.colors[i] for i in range(len(self.tasks)+1)],
            )
            # if idx_task == 9 and idx_task +1 <= shape[0] * shape[1] - 1: 
            #     # idx_task += 1 

            #     stacks = fig.axes[idx_task+1].stackplot(
            #         np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])),
            #         [his_smp[
            #             np.append(np.arange(0, len(his_smp), step), np.array([len(his_smp) - 1])), 
            #             idx_task, t] for t in range(len(self.tasks) + 1)],
            #         labels = ['Task' + str(i + 1) for i in range(len(self.tasks))] + ["mutation"],
            #         colors = [self.colors[i] for i in range(len(self.tasks)+1)],
            #     )
            #     if name_tasks : 
            #         fig.axes[idx_task+1].set_title('Task ' + str(idx_task + 1) +": " + task.name)
            #     else: 
            #         fig.axes[idx_task+1].set_title('Task ' + str(idx_task + 1))

            #     fig.axes[idx_task+1].set_xlabel('Generations')
            #     fig.axes[idx_task+1].set_ylabel("SMP")
            #     fig.axes[idx_task+1].set_ylim(bottom = y_lim[0], top = y_lim[1])
            #     if grid: 
            #         fig.axes[idx_task+1].grid() 
            # hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*','/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
            # for index, stack in enumerate(stacks):
            #     if index >= len(hatches):
            #         break 
            #     # stack.set_edgecolor("b") 
            #     stack.set_hatch(hatches[index]) 
            #     # print(stack.get_hatch())

            # plt.legend()
            if name_tasks : 
                if title_size is not None: 
                    fig.axes[idx_task].set_title('Task ' + str(idx_task + 1) +": " + task.name, fontsize= title_size)
                else: 
                    fig.axes[idx_task].set_title('Task ' + str(idx_task + 1) +": " + task.name)

            else:
                if title_size is not None: 
                    fig.axes[idx_task].set_title('Task ' + str(idx_task + 1), fontsize= title_size)
                else: 
                    fig.axes[idx_task].set_title('Task ' + str(idx_task + 1))

            if label_size_x is not None:
                fig.axes[idx_task].set_xlabel('Generations', fontsize= label_size_x)
            else:
                fig.axes[idx_task].set_xlabel('Generations')
            
            if label_size_y is not None: 
                fig.axes[idx_task].set_ylabel("SMP", fontsize= label_size_y)
            else: 
                fig.axes[idx_task].set_ylabel("SMP")

            if x_tick_size is not None: 
                for tick in fig.axes[idx_task].xaxis.get_major_ticks(): 
                    tick.label.set_fontsize(x_tick_size)
            if y_tick_size is not None: 
                for tick in fig.axes[idx_task].yaxis.get_major_ticks():
                    tick.label.set_fontsize(y_tick_size) 
                
            fig.axes[idx_task].set_ylim(bottom = y_lim[0], top = y_lim[1])
            if grid: 
                fig.axes[idx_task].grid() 
        # fig.delaxes(fig.axes[-1])
        # fig.delaxes(fig.axes[-2])
        # fig.axes.insert(-2, fig.axes[-1])
        # fig.delaxes(fig.axes[-2])
        # fig.delaxes(fig.axes[-3])
        # fig.delaxes(fig.axes[-2])
        
        # for i in range(shape[0] * shape[1]  - len(self.tasks)):
        #     # if fig.axes[i].line
        #     fig.delaxes(fig.axes[-1])
    
        print(len(fig.axes))

        lines, labels = fig.axes[0].get_legend_handles_labels()
        if bbox_to_anchor is not None: 
            fig.legend(lines, labels, loc = label_loc, ncol = ncol,fontsize= legend_size, handletextpad= handletextpad, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad)
            # fig.legend(lines, labels,facecolor=(208/255, 206/255, 206/255, 1), loc = label_loc, ncol = ncol,fontsize= legend_size, handletextpad= handletextpad, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad)
        else: 
            fig.legend(lines, labels, loc = label_loc, ncol = ncol,fontsize= legend_size)
            # fig.legend(lines, labels,facecolor=(208/255, 206/255, 206/255, 1), loc = label_loc, ncol = ncol,fontsize= legend_size)
        if pad is not None: 
            fig.tight_layout(pad)
        else:
            fig.tight_layout()
        fig.savefig("xinchao2.png", transparent= True, bbox_inches='tight')
        plt.show()
        if re_fig:
            return fig



    def fit(self, nb_generations: int, nb_inds_each_task: int, nb_inds_min = None,
        lr = 0.1, mu= 0.1,
        evaluate_initial_skillFactor = False,
        *args, **kwargs):
        super().fit(*args, **kwargs)
        
        # nb_inds_min
        if nb_inds_min is not None:
            assert nb_inds_each_task >= nb_inds_min
        else: 
            nb_inds_min = nb_inds_each_task

        # initial history of smp -> for render
        self.history_smp = []

        #initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        nb_inds_tasks = [nb_inds_each_task] * len(self.tasks)
        
        # SA params:
        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        eval_k = [0] * len(self.tasks)
        epoch = 0

        '''
        ------
        per params
        ------
        '''
        # prob choose first parent
        p_choose_father = np.ones((len(self.tasks), ))/ len(self.tasks)

        # Initialize memory M_smp
        M_smp = [self.battle_smp(i, len(self.tasks), lr, mu) for i in range(len(self.tasks))]

        #save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])
        self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])
        epoch = 1

        while sum(eval_k) <= MAXEVALS:
            turn_eval = 0

            # Delta epoch
            Delta:List[List[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()
            count_Delta: List[List[float]] = np.zeros((len(self.tasks), len(self.tasks) + 1)).tolist()

            # initial offspring_population of generation
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                list_tasks= self.tasks,
            )

            while turn_eval < sum(nb_inds_tasks):
                if sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history
                    self.history_cost.append([ind.fcost for ind in population.get_solves()])
                    self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])

                    self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
                    epoch += 1

                # choose subpop of father pa
                skf_pa = numba_randomchoice_w_prob(p_choose_father)

                # get smp 
                smp = M_smp[skf_pa].get_smp()

                # choose subpop of mother pb
                skf_pb = numba_randomchoice_w_prob(smp)

                if skf_pb != len(self.tasks):
                    if skf_pa == skf_pb:
                        pa, pb = population[skf_pa].__getRandomItems__(size= 2, replace=False)
                    else:
                        pa = population[skf_pa].__getRandomItems__()
                        pb = population[skf_pb].__getRandomItems__()

                    if np.all(pa.genes == pb.genes):
                        pb = population[skf_pb].__getWorstIndividual__
                    
                    oa, ob = self.crossover(pa, pb, skf_pa, skf_pa, population)

                    # dimension strategy
                    oa = self.dimension_strategy(oa, pb.skill_factor, pa)
                    ob = self.dimension_strategy(ob, pb.skill_factor, pb if skf_pa == skf_pb else pa)

                else:
                    pa, pb = population.__getIndsTask__(skf_pa, type= 'random', size= 2)

                    oa = self.mutation(pa, return_newInd= True)
                    oa.skill_factor = skf_pa

                    ob = self.mutation(pb, return_newInd= True)
                    ob.skill_factor = skf_pa

                
                # add oa, ob to offsprings population and eval fcost
                offsprings.__addIndividual__(oa)
                offsprings.__addIndividual__(ob)

                count_Delta[skf_pa][skf_pb] += 2
                eval_k[skf_pa] += 2
                turn_eval += 2

                # Calculate the maximum improvement percetage
                Delta1 = (pa.fcost - oa.fcost) / (pa.fcost ** 2 + 1e-50)
                Delta2 = (pa.fcost - ob.fcost) / (pa.fcost ** 2 + 1e-50)

                Delta[skf_pa][skf_pb] += max([Delta1, 0])**2
                Delta[skf_pa][skf_pb] += max([Delta2, 0])**2

            # merge
            population = population + offsprings
            population.update_rank()

            # selection
            nb_inds_tasks = [int(
                int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* (epoch - 1) + nb_inds_each_task, nb_inds_each_task))
            )] * len(self.tasks)
            self.selection(population, nb_inds_tasks)

            # update operators
            self.crossover.update(population = population)
            self.mutation.update(population = population)
            self.dimension_strategy.update(population = population)

            # update smp
            for skf in range(len(self.tasks)):
                M_smp[skf].update_SMP(Delta[skf], count_Delta[skf])

        #solve
        self.last_pop = population
        self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True)
        print()
        print(p_choose_father)
        print(eval_k)
        print('END!')
        return self.last_pop.get_solves()
    