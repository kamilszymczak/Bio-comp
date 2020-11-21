from abc import ABC, abstractmethod


class Optimisable(ABC):
    """Implementing these methods will allow a class to be optimised by PSO
    """
    @abstractmethod
    def evaluate_fitness(self, vec):
        """Evaluate the fitness of self given the parameters as a vector

        :param vec: A vector describing a subset of the parameters being searched
        :type vec: list(float)
        """
        pass

    @abstractmethod
    def dimension_vec(self):
        """A method to return a vector that defines the search space
        """
        pass

    @abstractmethod
    def decode_vec(self, vec):
        """A convienience method for evaluate_fitness

        :param vec: A vector describing a subset of the parameters being searched
        :type vec: list(float)
        """
        pass
