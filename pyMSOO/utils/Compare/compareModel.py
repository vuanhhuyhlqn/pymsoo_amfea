import matplotlib.pyplot as plt
import numpy as np
from ...MFEA.model import AbstractModel
import os
import pandas as pd
from typing import List, Tuple
from ..LoadSaveModel.load_utils import loadModel
import matplotlib 


def function_to_split(hand,labl,dividor):

    Hand_L=[]
    Hand_M=[]
    Labl_L=[]
    Labl_M=[]

    for h,l in zip(hand,labl):
        co=h.get_color()
        ls=h.get_linestyle()
        lw=h.get_linewidth()
        mk=h.get_marker()
        mew=h.get_markeredgewidth()
        ms=h.get_markersize()

        LABS=l.split(dividor)

        if len(LABS) != 2:
            print('Split Legends Error: Only exactly 1 Dividor is accepted.')
            print('                     Currently ' + str(len(LABS)-1) + ' dividors were given')
            return hand,labl

        #Line and Color
        LICO = plt.Line2D((0,1),(0,0), color=co, marker='', linestyle=ls,linewidth=lw)
        #Marker
        MARK = plt.Line2D((0,1),(0,0), color='k', marker=mk, markeredgewidth=mew, markersize=ms, linestyle='')

        if LABS[0] not in Labl_L:
            Hand_L.append(LICO)
            Labl_L.append(LABS[0])

        if LABS[1] not in Labl_M:
            Hand_M.append(MARK)
            Labl_M.append(LABS[1])

    return Hand_L+Hand_M,Labl_L+Labl_M



class CompareModel():
    # TODO so sánh
    def __init__(self, models: List[AbstractModel.model] = [], label: List[str] = None) -> None:
        self.models = models
        if label is None:
            label = [m.name for m in self.models]
        else:
            assert len(self.models) == len(label)
            for idx in range(len(label)):
                if label[idx] == Ellipsis:
                    label[idx] = self.models[idx].name

        self.label = label
        self.ls_marker = [  "X","D","o", "*", "^",">", "v", "<", "s", "P", "h", "1", "2", "3", "4", ]
        # self.ls_color = ["#76BACB","#A6DDD7", "#A7D6AF", "#00FF00", "#6495ED","#00FFFF", "#FFF7D6", "#FFFF00","#EFDAB1","#800000", "#FFAFA3", "#FFB8DE", "#BCB8E3", "#FF0000"]
        self.ls_color = ["#5CB3FF", "#007C80", "#ff0000",  "#CD7F32", "#64E986", "#FEA3AA", 
                         "#01F9C6", "#5CB3FF", "#728C00","#000080", "#FF8C00", 
                         "#F70D1A", "#B666D2"]
        # self.ls_color = ["#0000cd", "#ff0000", "#ff8c00", "#ffff00", "#00ced1", "#228b22"]

    def render(self, shape: tuple = None, min_cost=0, nb_generations: int = None, step=1, figsize: Tuple[int, int] = None, dpi=200, 
               yscale: str = None, re=False, label_shape=None, label_loc=None, grid = True, title= None, showname= True, 
               title_size= None, label_size_x=None, label_size_y= None, pad=None, x_tick_size=None, y_tick_size=None,
               bbox_to_anchor= None, loc_legend= None,borderaxespad= None,handletextpad= 0.8, legend_size=14, 
               scatter_size=200):
        print(matplotlib.rcParams['pdf.fonttype']);
        assert np.all([len(self.models[0].tasks) == len(m.tasks)
                      for m in self.models])
        nb_tasks = len(self.models[0].tasks)
        for i in range(nb_tasks):
            assert np.all([self.models[0].tasks[i] == m.tasks[i]
                          for m in self.models])

        if label_shape is None:
            label_shape = (1, len(self.label))
        else:
            assert label_shape[0] * label_shape[1] >= len(self.label)

        if label_loc is None:
            label_loc = 'lower center'

        if shape is None:
            shape = (nb_tasks // 3 + np.sign(nb_tasks % 3), 3)
        else:
            assert shape[0] * shape[1] >= nb_tasks

        if nb_generations is None:
            nb_generations = min([len(m.history_cost) for m in self.models])
        else:
            nb_generations = min(nb_generations, min(
                [len(m.history_cost) for m in self.models]))

        if figsize is None:
            print(shape)
            figsize = (shape[1] * 6, shape[0] * 5)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        if title is None:
            fig.suptitle("Compare Models\n", size=15)
        else: 
            # fig.suptitle("\n", size=15)
            pass
        # fig.set_facecolor((208/255, 206/255, 206/255, 1))
        fig.subplots(shape[0], shape[1])

        if step >= 10:
            marker = 'o'
        else:
            marker = None
        for idx_task, task in enumerate(self.models[0].tasks):
            count_color = 0 
            for idx_model, model in enumerate(self.models):
                fig.axes[idx_task].plot(
                    np.append(np.arange(0, nb_generations, step),
                              np.array([nb_generations - 1])),
                    np.where(
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task] >= min_cost,
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task],
                        0
                    ),
                    label=self.label[idx_model],
                    markersize= scatter_size,
                    marker=marker if marker is None else self.ls_marker[idx_model%2],
                    color= self.ls_color[count_color]
                )
                count_color += idx_model % 2 

                # plt.legend()
                if yscale is not None:
                    fig.axes[idx_task].set_yscale(yscale)
            if showname:
                if title_size is not None:
                    
                    # fig.axes[idx_task].set_title(task.name, fontdict= {'fontsize': title_size})
                    fig.axes[idx_task].set_title(task.name, fontsize=title_size)
                else:
                    fig.axes[idx_task].set_title(task.name)
            else: 
                if title_size is not None:
                    # fig.axes[idx_task].set_title("Task " + str(idx_task + 1), fontdict= {'fontsize': title_size})
                    fig.axes[idx_task].set_title("Task " + str(idx_task + 1), fontsize= title_size)
                else: 
                    fig.axes[idx_task].set_title("Task " + str(idx_task + 1))
            # fig.axes[idx_task].set_facecolor((208/255, 206/255, 206/255, 1))
            if label_size_x is not None: 
                fig.axes[idx_task].set_xlabel("Generations", fontsize=label_size_x)
            else:
                fig.axes[idx_task].set_xlabel("Generations")
            if label_size_y is not None: 
                fig.axes[idx_task].set_ylabel("Log scale objective value", fontsize= label_size_y)
            else:
                fig.axes[idx_task].set_ylabel("Log scale objective value")
            if grid: 
                fig.axes[idx_task].grid()
            if x_tick_size is not None:
                for tick in fig.axes[idx_task].xaxis.get_major_ticks():
                    tick.label.set_fontsize(x_tick_size)
            if y_tick_size is not None: 
                for tick in fig.axes[idx_task].yaxis.get_major_ticks():
                    tick.label.set_fontsize(y_tick_size)
        for idx_blank_fig in range(idx_task + 1, shape[0] * shape[1]):
            fig.delaxes(fig.axes[idx_task + 1])

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        hand, labl = lines, labels
        hand, labl = function_to_split(hand,labl,'*')
        lines, labels = hand, labl 
        print(hand)
        print(labels)
        if bbox_to_anchor is not None: 
            # leg = fig.legend(lines, labels,facecolor=(208/255, 206/255, 206/255, 0), fontsize= legend_size, handletextpad=handletextpad, loc=label_loc, ncol=label_shape[1], bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad)
            # leg = fig.legend(lines, labels,facecolor=(208/255, 206/255, 206/255, 1), fontsize= legend_size, handletextpad=handletextpad, loc=label_loc, ncol=label_shape[1], bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad)
            # leg = fig.legend(lines, labels, fontsize= legend_size, handletextpad=handletextpad, loc=label_loc, ncol=label_shape[1], bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad)
            
            line1, label1 = lines[:6], labels[:6]
            line2, label2 = lines[6:], labels[6:]
            leg = fig.legend(line1, label1, fontsize= legend_size,
                              handletextpad=handletextpad, loc=label_loc, ncol= len(line1), bbox_to_anchor=(0.5, -0.16), borderaxespad=borderaxespad)
            for line in leg.legendHandles:
                line.set_linewidth(8)

            leg = fig.legend(line2, label2, fontsize= legend_size, markerscale=1.5, handletextpad=handletextpad, loc=label_loc, ncol = len(line2), bbox_to_anchor=(0.5, -0.09), borderaxespad=borderaxespad)
        else:
            leg = fig.legend(lines, labels, fontsize=legend_size, handletextpad= handletextpad, loc=label_loc, ncol=label_shape[1])
        if pad is not None: 
            fig.tight_layout(pad) 
        else:
            fig.tight_layout()
        # fig.legend(facecolor= (208/255, 206/255, 206/255, 1.0))
        # plt.savefig("xinchao2.png", transparent= True)
        # fig.savefig("xinchao2.png", bbox_inches='tight', facecolor= (208/255, 206/255, 206/255, 1))
        plt.show()
        
        # plt.savefig("xinchao.png", transparent= True)

        # if re:
        #     return fig
        return fig, leg 

    def summarizing_compare_result(self, path=None, idx_main_algo=0, min_value=0, combine=True, nb_task=50, ls_benchmark=None):
        #
        if path is None:
            result_table = np.zeros(shape=(len(self.label)-1, 3), dtype=int)
            name_row = []
            name_column = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(self.label)):
                if idx_algo != idx_main_algo:
                    name_row.append(
                        str(self.label[idx_main_algo] + " vs " + self.label[idx_algo]))
                    result = np.where(self.models[idx_main_algo].history_cost[-1] > min_value, self.models[idx_main_algo].history_cost[-1], min_value)\
                        - np.where(self.models[idx_algo].history_cost[-1] > min_value,
                                   self.models[idx_algo].history_cost[-1], min_value)
                    # Better
                    result_table[count_row][0] += len(np.where(result < 0)[0])
                    # Equal
                    result_table[count_row][1] += len(np.where(result == 0)[0])
                    # Worse
                    result_table[count_row][2] += len(np.where(result > 0)[0])
                    count_row += 1

            result_table = pd.DataFrame(
                result_table, columns=name_column, index=name_row)
        else:
            list_algo = os.listdir(path)
            print(list_algo)
            idx_main_algo = input("Type index main algo: ")
            idx_main_algo = int(idx_main_algo) 
            benchmarks = [name_ben.split(
                "_")[-1].split(".")[0] for name_ben in os.listdir(os.path.join(path, list_algo[0]))]
            ls_model_cost = [np.zeros(
                shape=(len(benchmarks), nb_task)).tolist() for i in range(len(list_algo))]
            # print(ls_model_cost)
            for idx_algo in range(len(list_algo)):
                path_algo = os.path.join(path, list_algo[idx_algo])
                # count_benchmark = 0
                print(path_algo)
                for benchmark_mso in sorted(os.listdir(path_algo)):
                    count_benchmark = benchmark_mso.split(".")[0]
                    count_benchmark = count_benchmark.split("_")[-1]
                    count_benchmark = int(count_benchmark) - 1
                    print(benchmark_mso)
                    model = loadModel(os.path.join(
                        path_algo, benchmark_mso), ls_benchmark[count_benchmark])
                    
                    ls_model_cost[idx_algo][count_benchmark] = model.history_cost[-1]
                    # count_benchmark += 1
                    del model 
            result_table = np.zeros(
                shape=(len(benchmarks), len(list_algo)-1, 3), dtype=int)
            name_row = []
            name_col = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(list_algo)):
                if idx_main_algo != idx_algo:
                    name_row.append(
                        list_algo[idx_main_algo] + " vs " + list_algo[idx_algo])
                    for idx_benchmark in range(len(benchmarks)):
                        result = np.where(ls_model_cost[idx_main_algo][idx_benchmark] > min_value, ls_model_cost[idx_main_algo][idx_benchmark], min_value) \
                            - np.where(ls_model_cost[idx_algo][idx_benchmark] > min_value,
                                       ls_model_cost[idx_algo][idx_benchmark], min_value)

                        result_table[idx_benchmark][count_row][0] += len(
                            np.where(result < 0)[0])
                        result_table[idx_benchmark][count_row][1] += len(
                            np.where(result == 0)[0])
                        result_table[idx_benchmark][count_row][2] += len(
                            np.where(result > 0)[0])
                    count_row += 1
            if combine is True:
                result_table = pd.DataFrame(
                    np.sum(result_table, axis=0), columns=name_col, index=name_row)
        return result_table

    def detail_compare_result(self, min_value=0, round = 100):
        name_row = [str("Task" + str(i + 1))
                    for i in range(len(self.models[0].tasks))]
        name_col = self.label
        data = []
        for model in self.models:
            data.append(model.history_cost[-1])

        data = np.array(data).T
        data = np.round(data, round)
        pre_data = pd.DataFrame(data)
        end_data = pd.DataFrame(data).astype(str)

        result_compare = np.zeros(shape=(len(name_col)), dtype=int).tolist()
        for task in range(len(name_row)):
            argmin = np.argmin(data[task])
            min_value_ = max(data[task][argmin], min_value)
            # for col in range(len(name_col)):
            #     if data[task][col] == data[task][argmin]:
            #         result_compare[col] += 1
            #         end_data.iloc[task][col]= str("(≈)") + pre_data.iloc[task][col].astype(str)
            #     elif data[task][col] > data[task][argmin]:
            #         end_data.iloc[task][col]= str("(-)") + pre_data.iloc[task][col].astype(str)
            #     else:

            for col in range(len(name_col)):
                if data[task][col] <= min_value_:
                    result_compare[col] += 1
                    end_data.iloc[task][col] = str(
                        "(+)") + end_data.iloc[task][col]

        for col in range(len(name_col)):
            result_compare[col] = str(
                result_compare[col]) + "/" + str(len(name_row))

        result_compare = pd.DataFrame([result_compare], index=[
                                      "Compare"], columns=name_col)
        end_data.columns = name_col
        end_data.index = name_row
        end_data = pd.concat([end_data, result_compare])

        # assert data.shape == (len(name_row), len(name_col))

        return end_data



class CompareModel_Original():
    # TODO so sánh
    def __init__(self, models: List[AbstractModel.model] = [], label: List[str] = None) -> None:
        self.models = models
        if label is None:
            label = [m.name for m in self.models]
        else:
            assert len(self.models) == len(label)
            for idx in range(len(label)):
                if label[idx] == Ellipsis:
                    label[idx] = self.models[idx].name

        self.label = label
        self.ls_marker = ["X", "D", "*", "o", "^", "v", "<", ">", "s", "P"]

    def render(self, shape: tuple = None, min_cost=0, nb_generations: int = None, step=1, figsize: Tuple[int, int] = None, dpi=200, 
               yscale: str = None, re=False, label_shape=None, label_loc=None, grid = True, title= None, showname= True, 
               title_size= None, label_size_x=None, label_size_y= None, pad=None, x_tick_size=None, y_tick_size=None,
               bbox_to_anchor= None, loc_legend= None,borderaxespad= None,handletextpad= 0.8, legend_size=14, 
               scatter_size=200):
        print(matplotlib.rcParams['pdf.fonttype']);
        assert np.all([len(self.models[0].tasks) == len(m.tasks)
                      for m in self.models])
        nb_tasks = len(self.models[0].tasks)
        for i in range(nb_tasks):
            assert np.all([self.models[0].tasks[i] == m.tasks[i]
                          for m in self.models])

        if label_shape is None:
            label_shape = (1, len(self.label))
        else:
            assert label_shape[0] * label_shape[1] >= len(self.label)

        if label_loc is None:
            label_loc = 'lower center'

        if shape is None:
            shape = (nb_tasks // 3 + np.sign(nb_tasks % 3), 3)
        else:
            assert shape[0] * shape[1] >= nb_tasks

        if nb_generations is None:
            nb_generations = min([len(m.history_cost) for m in self.models])
        else:
            nb_generations = min(nb_generations, min(
                [len(m.history_cost) for m in self.models]))

        if figsize is None:
            figsize = (shape[1] * 6, shape[0] * 5)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        if title is None:
            fig.suptitle("Compare Models\n", size=15)
        else: 
            # fig.suptitle("\n", size=15)
            pass
        fig.set_facecolor("white")
        fig.subplots(shape[0], shape[1])

        if step >= 10:
            marker = 'o'
        else:
            marker = None

        for idx_task, task in enumerate(self.models[0].tasks):
            for idx_model, model in enumerate(self.models):
                fig.axes[idx_task].plot(
                    np.append(np.arange(0, nb_generations, step),
                              np.array([nb_generations - 1])),
                    np.where(
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task] >= min_cost,
                        model.history_cost[np.append(np.arange(0, nb_generations, step), np.array([
                                                     nb_generations - 1])), idx_task],
                        0
                    ),
                    label=self.label[idx_model],
                    markersize= scatter_size,
                    marker=marker if marker is None else self.ls_marker[idx_model],
                )
                # plt.legend()
                if yscale is not None:
                    fig.axes[idx_task].set_yscale(yscale)
            if showname:
                if title_size is not None:
                    
                    # fig.axes[idx_task].set_title(task.name, fontdict= {'fontsize': title_size})
                    fig.axes[idx_task].set_title(task.name, fontsize=title_size)
                else:
                    fig.axes[idx_task].set_title(task.name)
            else: 
                if title_size is not None:
                    # fig.axes[idx_task].set_title("Task " + str(idx_task + 1), fontdict= {'fontsize': title_size})
                    fig.axes[idx_task].set_title("Task " + str(idx_task + 1), fontsize= title_size)
                else: 
                    fig.axes[idx_task].set_title("Task " + str(idx_task + 1))
            if label_size_x is not None: 
                fig.axes[idx_task].set_xlabel("Generations", fontsize=label_size_x)
            else:
                fig.axes[idx_task].set_xlabel("Generations")
            if label_size_y is not None: 
                fig.axes[idx_task].set_ylabel("Log scale objective value", fontsize= label_size_y)
            else:
                fig.axes[idx_task].set_ylabel("Log scale objective value")
            if grid: 
                fig.axes[idx_task].grid()
            if x_tick_size is not None:
                for tick in fig.axes[idx_task].xaxis.get_major_ticks():
                    tick.label.set_fontsize(x_tick_size)
            if y_tick_size is not None: 
                for tick in fig.axes[idx_task].yaxis.get_major_ticks():
                    tick.label.set_fontsize(y_tick_size)
        for idx_blank_fig in range(idx_task + 1, shape[0] * shape[1]):
            fig.delaxes(fig.axes[idx_task + 1])

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        if bbox_to_anchor is not None: 
            fig.legend(lines, labels,fontsize= legend_size, handletextpad=handletextpad, loc=label_loc, ncol=label_shape[1], bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad)
        else:
            fig.legend(lines, labels,fontsize=legend_size, handletextpad= handletextpad, loc=label_loc, ncol=label_shape[1])
        if pad is not None: 
            fig.tight_layout(pad) 
        else:
            fig.tight_layout()
        plt.show()
        if re:
            return fig

    def summarizing_compare_result(self, path=None, idx_main_algo=0, min_value=0, combine=True, nb_task=50, ls_benchmark=None):
        #
        if path is None:
            result_table = np.zeros(shape=(len(self.label)-1, 3), dtype=int)
            name_row = []
            name_column = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(self.label)):
                if idx_algo != idx_main_algo:
                    name_row.append(
                        str(self.label[idx_main_algo] + " vs " + self.label[idx_algo]))
                    result = np.where(self.models[idx_main_algo].history_cost[-1] > min_value, self.models[idx_main_algo].history_cost[-1], min_value)\
                        - np.where(self.models[idx_algo].history_cost[-1] > min_value,
                                   self.models[idx_algo].history_cost[-1], min_value)
                    # Better
                    result_table[count_row][0] += len(np.where(result < 0)[0])
                    # Equal
                    result_table[count_row][1] += len(np.where(result == 0)[0])
                    # Worse
                    result_table[count_row][2] += len(np.where(result > 0)[0])
                    count_row += 1

            result_table = pd.DataFrame(
                result_table, columns=name_column, index=name_row)
        else:
            list_algo = os.listdir(path)
            print(list_algo)
            benchmarks = [name_ben.split(
                "_")[-1].split(".")[0] for name_ben in os.listdir(os.path.join(path, list_algo[0]))]
            ls_model_cost = [np.zeros(
                shape=(len(benchmarks), nb_task)).tolist() for i in range(len(list_algo))]
            # print(ls_model_cost)
            for idx_algo in range(len(list_algo)):
                path_algo = os.path.join(path, list_algo[idx_algo])
                # count_benchmark = 0

                for benchmark_mso in sorted(os.listdir(path_algo)):
                    count_benchmark = benchmark_mso.split(".")[0]
                    count_benchmark = count_benchmark.split("_")[-1]
                    count_benchmark = int(count_benchmark) - 1
                    print(path_algo, benchmark_mso)
                    model = loadModel(os.path.join(
                        path_algo, benchmark_mso), ls_benchmark[count_benchmark])

                    ls_model_cost[idx_algo][count_benchmark] = model.history_cost[-1]
                    # count_benchmark += 1

            result_table = np.zeros(
                shape=(len(benchmarks), len(list_algo)-1, 3), dtype=int)
            name_row = []
            name_col = ["Better", "Equal", "Worse"]
            count_row = 0
            for idx_algo in range(len(list_algo)):
                if idx_main_algo != idx_algo:
                    name_row.append(
                        list_algo[idx_main_algo] + " vs " + list_algo[idx_algo])
                    for idx_benchmark in range(len(benchmarks)):
                        result = np.where(ls_model_cost[idx_main_algo][idx_benchmark] > min_value, ls_model_cost[idx_main_algo][idx_benchmark], min_value) \
                            - np.where(ls_model_cost[idx_algo][idx_benchmark] > min_value,
                                       ls_model_cost[idx_algo][idx_benchmark], min_value)

                        result_table[idx_benchmark][count_row][0] += len(
                            np.where(result < 0)[0])
                        result_table[idx_benchmark][count_row][1] += len(
                            np.where(result == 0)[0])
                        result_table[idx_benchmark][count_row][2] += len(
                            np.where(result > 0)[0])
                    count_row += 1
            if combine is True:
                result_table = pd.DataFrame(
                    np.sum(result_table, axis=0), columns=name_col, index=name_row)
        return result_table

    def detail_compare_result(self, min_value=0, round = 100):
        name_row = [str("Task" + str(i + 1))
                    for i in range(len(self.models[0].tasks))]
        name_col = self.label
        data = []
        for model in self.models:
            data.append(model.history_cost[-1])

        data = np.array(data).T
        data = np.round(data, round)
        pre_data = pd.DataFrame(data)
        end_data = pd.DataFrame(data).astype(str)

        result_compare = np.zeros(shape=(len(name_col)), dtype=int).tolist()
        for task in range(len(name_row)):
            argmin = np.argmin(data[task])
            min_value_ = max(data[task][argmin], min_value)
            # for col in range(len(name_col)):
            #     if data[task][col] == data[task][argmin]:
            #         result_compare[col] += 1
            #         end_data.iloc[task][col]= str("(≈)") + pre_data.iloc[task][col].astype(str)
            #     elif data[task][col] > data[task][argmin]:
            #         end_data.iloc[task][col]= str("(-)") + pre_data.iloc[task][col].astype(str)
            #     else:

            for col in range(len(name_col)):
                if data[task][col] <= min_value_:
                    result_compare[col] += 1
                    end_data.iloc[task][col] = str(
                        "(+)") + end_data.iloc[task][col]

        for col in range(len(name_col)):
            result_compare[col] = str(
                result_compare[col]) + "/" + str(len(name_row))

        result_compare = pd.DataFrame([result_compare], index=[
                                      "Compare"], columns=name_col)
        end_data.columns = name_col
        end_data.index = name_row
        end_data = pd.concat([end_data, result_compare])

        # assert data.shape == (len(name_row), len(name_col))

        return end_data

