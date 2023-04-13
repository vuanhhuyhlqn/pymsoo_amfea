
from ....utils.EA import Individual, AbstractTask
import numpy as np
from numba import jit


class Individual_func(Individual):
    def __init__(self, genes, skill_factor = None, fcost = None, parent= None, dim= None, *args, **kwargs) -> None:
        super().__init__(genes, skill_factor, fcost, parent, dim)
        if genes is None:
            self.genes: np.ndarray = np.random.rand(dim)

class AbstractFunc(AbstractTask):
    limited_space = False
    bound = (None, None)
    global_optimal = 0

    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None, *args, **kwargs):
        self.dim = dim

        if rotation_matrix is not None:
            assert np.all(np.array(rotation_matrix.shape) == dim)
            self.rotation_matrix = rotation_matrix
            self.inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)
        else:
            self.rotation_matrix = np.identity(dim)
            self.inv_rotation_matrix = np.identity(dim)
        
        tmp = np.array(shift).reshape(-1, )
        assert dim % len(tmp) == 0
        self.shift = np.array([[i] * int(dim / len(tmp)) for i in tmp ]).reshape(-1, )

        self.global_optimal = self.encode(self.inv_rotation_matrix @ self.global_optimal + self.shift)
        
        if bound is not None:
            self.limited_space = True
            self.bound = bound
            self.name = self.__class__.__name__ + ': [' + str(self.bound[0]) + ', ' + str(self.bound[1]) + ']^' + str(dim)
        else:
            self.name = self.__class__.__name__ + ': R^' + str(dim)

    def __eq__(self, other: object) -> bool:
        if self.__repr__() == other.__repr__():
            return True
        return self.dim == other.dim and np.all(self.shift == other.shift) and self.bound == other.bound

    def encode(self, x):
        '''
        encode x to [0, 1]
        '''
        x_encode = x
        # x_encode = self.inv_rotation_matrix @ x_encode + self.shift
        if self.limited_space == True:
            x_encode = (x_encode - self.bound[0])/(self.bound[1] - self.bound[0])
        return x_encode 

    @staticmethod
    # @jit(nopython = True)
    def decode(x, dim, limited_space, bound, rotation_matrix, shift):
        '''
        decode x from [0, 1] to bound
        '''
        x_decode = x[:dim]
        if limited_space == True:
            x_decode = x_decode * (bound[1] - bound[0]) + bound[0]
        x_decode = rotation_matrix @ (x_decode - shift) 
        return x_decode 

    def __call__(self, x):
        x = self.__class__.decode(x, self.dim, self.limited_space, self.bound, self.rotation_matrix, self.shift)
        return self.__class__._func(x)

    @staticmethod
    @jit(nopython = True)
    def _func(x):
        pass
