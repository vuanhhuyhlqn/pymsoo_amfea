from pyMSOO.MFEA.benchmark.continous.CEC17 import CEC17_benchmark 
from pyMSOO.MFEA.benchmark.continous.WCCI22 import WCCI22_benchmark
from pyMSOO.MFEA.benchmark.continous.funcs import * 

class config: 
    '''Benchmarks'''
    WCCI22_benchmark_50_tasks = [WCCI22_benchmark.get_50tasks_benchmark(i)[0] for i in range(1,11)] 
    WCCI22_benchmark_50_IndClass = [WCCI22_benchmark.get_50tasks_benchmark(i)[1] for i in range(1,11)]

    CEC17_benchmark_10_task, CEC17_benchmark_10_IndClass = CEC17_benchmark.get_10tasks_benchmark() 


print(getattr(config, 'WCCI22_benchmark_50_tasks'))