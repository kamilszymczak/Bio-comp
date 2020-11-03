

# Need a vector of all weights in a decodable order in the network
# Need a vector of all activation functions in the network

# Need to define the boundaries (-1, 1)



# need to define the behaviour when you hit a boundary (random re-intit, bounce, reject dimensions out of bounds)

# define movement behaviour

# information sharing 

# sub boundary groups: define the inner boundaries for subsets of particles

class pso:
    def __init__(self, swarm_size=10, bound=(1, -1), alpha=0.1, beta=0.2, gamma=0.2, delta=0.2, epsilon=0.1, max_iter=int(1e4)):
        """PSO constructor

        :param swarm_size: desired swarm size, defaults to 10
        :type swarm_size: int, optional
        :param bound: limits of dimensionality, defaults to (1, -1)
        :type bound: tuple, optional
        :param alpha: proportion of velocity to be retained, defaults to 0.1
        :type alpha: float, optional
        :param beta: proportion of personal best to be retained, defaults to 0.2
        :type beta: float, optional
        :param gamma: proportion of the informants’ best to be retained, defaults to 0.2
        :type gamma: float, optional
        :param delta: proportion of global best to be retained, defaults to 0.2
        :type delta: float, optional
        :param epsilon: jump size of a particle, defaults to 0.1
        :type epsilon: float, optional
        :param max_iter: The maximum number of iterations before termination, defaults to 0.1
        :type max_iter: int, optional
        """
        self.swarm_size = swarm_size
        self.boundary = bound
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter

        # search_dimension is a list of tuples || an integer -- eg. search_dimension = 3 => [(-1, 1), (-1, 1), (-1, 1)]
        self.search_dimesion = None
        self.best = None
        self.particles = None


    def set_search_dimentions(self):
        # should be first method call after instantiation
        raise NotImplementedError()

    def instantiate_particles(self):
        #depends on set_search_dimensions
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def calculate_fitness(self):
        raise NotImplementedError()



class particle:
    def __init__(self):
        raise NotImplementedError()
        self.search_space = [0.1, 0.5, -0.3]


