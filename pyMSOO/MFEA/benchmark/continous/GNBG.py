import numpy as np
from typing import Tuple, Type, List
from scipy.io import loadmat
from .utils import Individual_func, AbstractFunc
from .funcs import GNBGFunc
import os

path = os.path.dirname(os.path.realpath(__file__))

class GNBG_benchmark:
    dim = 30 
    task_size = 2

    @staticmethod
    def get_all_tasks_benchmark(fix: bool = False) -> Tuple[List[AbstractFunc], Type[Individual_func]]:
        tasks = []
        for problem_index in range(1, 25):  
            tasks.append(
                GNBGFunc(
                    problem_index=problem_index,
                    bound=None, 
                    shift=None,  
                    rotation_matrix=None 
                )
            )
        return tasks, Individual_func

    def get_multitask_benchmark(ID: int)-> Tuple[List[AbstractFunc], Type[Individual_func]]:
        tasks = []
        if ID == 1:
            for problem_index in range(1, 7):
                tasks.append(
                    GNBGFunc(
                        problem_index=problem_index,
                        bound=None, 
                        shift=None,  
                        rotation_matrix=None 
                    )
                )
        if ID == 2:
            for problem_index in range(7, 16):
                tasks.append(
                    GNBGFunc(
                        problem_index=problem_index,
                        bound=None, 
                        shift=None,  
                        rotation_matrix=None 
                    )
                )        
        if ID == 3:
            for problem_index in range(16, 25):
                tasks.append(
                    GNBGFunc(
                        problem_index=problem_index,
                        bound=None, 
                        shift=None,  
                        rotation_matrix=None 
                    )
                )     
        return tasks, Individual_func
    @staticmethod
    def GNBG_get_tasks_benchmark(ID: int) -> Tuple[List[AbstractFunc], Type[Individual_func]]:
        tasks = []
        task_pairs = [
            (1, 2),   
            (3, 4),  
            (5, 6),   
            (7, 8),  
            (9, 10),  
            (11, 12),
            (13, 14), 
            (15, 16), 
            (17, 18), 
            (19, 20), 
            (21, 22), 
            (23, 24), 
        ]
        
    
        if ID < 1 or ID > 12:
            raise ValueError(f'ID must be an integer from 1 to 12, not {ID}')
        
        for i in range(len(task_pairs[ID - 1])):
            tasks.append(
                GNBGFunc(
                    problem_index=task_pairs[ID - 1][i],
                    bound=None,  # Use GNBG's default bounds
                    shift=None,  # Use GNBG's OptimumPosition
                    rotation_matrix=None  # Use GNBG's RotationMatrix
                )
            )
        return tasks, Individual_func
