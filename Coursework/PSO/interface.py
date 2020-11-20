from abc import ABC, abstractmethod


class Optimisable(ABC):
    @abstractmethod
    def evaluate_fitness(self, vec):
        pass

    @abstractmethod
    def dimension_vec(self):
        pass

    @abstractmethod
    def decode_vec(self, vec):
        pass
