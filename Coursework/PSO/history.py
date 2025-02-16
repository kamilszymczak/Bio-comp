from .interface import Optimisable
import numpy as np
import matplotlib.pyplot as plt


class PSOFittest(Optimisable):
    def __init__(self, model):
        """A wrapper class to store only the fittest location and fitness

        :param model: The model being optimised
        :type model: [type]
        """
        self.model = model
        self.run_count = 0
        self.fitness = 0
        self.vec = None

    def evaluate_fitness(self, vec):
        """Wrapper around the models assess fitness function to store all vectors passed into the history

        :param vec: The vector to use to build the model
        :type vec: numpy.array
        :return: a fitness value for PSO
        :rtype: float
        """
        self.run_count += 1
        fitness = self.model.evaluate_fitness(vec)
        if fitness > self.fitness:
            self.fitness = fitness
            self.vec = vec
        return fitness

    def dimension_vec(self):
        return self.model.dimension_vec()

    def decode_vec(self, vec):
        pass

class PSOHistory(Optimisable):
    """A wrapper class around classes implementing the Optimisable interface to store and plot metrics for every vector produced when performing PSO

    :param model: The model being optimised, must have implemented Optimisable
    :type model: model
    :param num_particles: The number of particles the model is using, defaults to 10
    :type num_particles: int, optional
    :param num_iterations: The number of iterations the model uses, defaults to 50
    :type num_iterations: int, optional
    """
        
    def __init__(self, model, num_particles=10, num_iterations=50):
        self.model = model
        self.vec_history = []
        self.vec_fitness = []
        self.particle_fitness = {}
        self.particle_location = {}
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.run_count = 0
        


    def historical_particle_fitness(self):
        """Build a dictionary recoring the fitness values returned by PSO
        """
        for i in range(self.num_particles):
            offset_vec = self.vec_fitness[i:]
            self.particle_fitness[i] = offset_vec[::self.num_particles]
        return self.particle_fitness


    def historical_particle_location(self):
        """Build a dictionary recoring the vectors returned by PSO
        """
        for i in range(self.num_particles):
            offset_vec = self.vec_history[i:]
            self.particle_location[i] = offset_vec[::self.num_particles]
        return self.particle_location

    def best_particle(self):
        """The most fit particle discovered

        :return: index and value pair
        :rtype: int, list(float)
        """
        index = np.argmax(self.vec_fitness)
        return index, self.vec_history[index]


    def best_iter_per_particle(self):
        """The indices of the most fit vector that every particle discovers

        :return: list of index values where each element is a new particle
        :rtype: list(int)
        """
        max_indices = []
        for i in range(len(self.particle_fitness)):
            max_indices.append(np.argmax(self.particle_fitness[i]))
            #print('Particle ', i, ': ', max(self.particle_fitness[i]))
        self.best_indices = max_indices
        return max_indices


    def evaluate_fitness(self, vec):
        """Wrapper around the models assess fitness function to store all vectors passed into the history

        :param vec: The vector to use to build the model
        :type vec: numpy.array
        :return: a fitness value for PSO
        :rtype: float
        """
        self.run_count = 0
        self.vec_history.append(vec)
        fitness = self.model.evaluate_fitness(vec)
        self.vec_fitness.append(fitness)
        return fitness

    def dimension_vec(self):
        """Return the encapsulated models dimension vector

        :return: A vector describing the dimensions of the search space
        :rtype: list(tuple(float))
        """
        return self.model.dimension_vec()

    def decode_vec(self, vec):
        # No implementation necessary
        pass

    def reshape_lists(self, num_runs=10):
        for i in range(len(self.particle_fitness)):
            self.particle_fitness[i] = np.mean(
                np.array(self.particle_fitness[i]).reshape(num_runs, self.num_iterations), axis=0)


    def plot_fitness(self, particles=(0, 10)):
        """Plot the fitness of each particle

        :param particles: The range of particles to plot, defaults to (0, 10)
        :type particles: tuple, optional
        """
        iterations = range(len(self.particle_fitness[particles[0]]))
        for p in range(particles[0], particles[1]):
            plt.plot(
                iterations, self.particle_fitness[p], label="particle " + str(p))

        plt.title('Particles fitness change over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()


    def plot_mean_fitness(self):
        """Plot the fitness of each particle

        :param particles: The range of particles to plot, defaults to (0, 10)
        :type particles: tuple, optional
        """
        
        line = np.mean(list(self.particle_fitness.values()), axis=0)
        iterations = list(range(line.shape[0]))
        plt.plot(iterations, line, label='mean of particles')

        plt.title('Particles fitness change over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()

    def plot_curve(self):
        plt.title('Function approximation curve')
        plt.xlabel('input')
        plt.ylabel('output')
        plt.scatter(self.model.input, self.model.y, alpha=0.4, label='truth')
        plt.scatter(self.model.input, self.model.y_hat, alpha=0.4, label='prediction')
        plt.legend()
        plt.show()