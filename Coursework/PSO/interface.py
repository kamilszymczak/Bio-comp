from abc import ABC, abstractmethod


class Optimisable(ABC):
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
