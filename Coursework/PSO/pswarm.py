from functools import total_ordering
import numpy as np
from datetime import datetime, timedelta
import enum
import random

# Need a vector of all weights in a decodable order in the network
# Need a vector of all activation functions in the network

# Need to define the boundaries (-1, 1)



# need to define the behaviour when you hit a boundary (random re-intit, bounce, reject dimensions out of bounds)

# define movement behaviour

# information sharing 

# sub boundary groups: define the inner boundaries for subsets of particles


TerminationPolicy = enum.Enum(
    'TerminationPolicy', 'ITERATIONS DURATION CONVERGENCE')

BoundaryPolicy = enum.Enum(
    'BoundaryPolicy', 'RANDOMREINIT REFUSE BOUNCE')

class PSO:
    def __init__(self, swarm_size=10, bound=(1, -1), alpha=0.1, beta=1.3, gamma=1.4, delta=1.3, epsilon=0.1, max_iter=int(1e6)):
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
        self.delta = delta
        self.epsilon = epsilon
        self.max_iter = max_iter


        self.search_dimension = None
        self.search_dimension_set = False
        self.best = None
        self.previous_best = None
        self.particles = None

        self.fitness_fn = None # The arg to this is the shape of the ANN (wieghts + activation)



    def set_search_dimentions(self, dimensions):
        """Specify the dimensionality of the search

        :param dimensions: define dimensionality. Either with an int (using default boundaries), or a list/np.array of tuples descriping the boundaries for each dimension
        :type dimensions: int / list / numpy.array
        :raises ValueError: When dimension parameter does not meet specified requirements
        """
        # set by a list of tuples || an integer -- eg. search_dimension = 3 => [(-1, 1), (-1, 1), (-1, 1)]
        if type(dimensions) is int:
            self.search_dimension = np.array([self.boundary for i in range(dimensions)])

        elif type(dimensions) is list:
            #TODO check list is valid
            self.search_dimension = np.array(dimensions)

        elif isinstance(dimensions, np.ndarray):
            #TODO check numpy array validity
            self.search_dimension = dimensions

        else:
            self.search_dimension_set = False
            raise ValueError("Invalid dimensions parameter")

        self.search_dimension_set = True

    def run(self):
        self._instantiate_particles()
        self.best = None
        previous_best = self.best
        controller = IterationController(TerminationPolicy.ITERATIONS, max_iter=1000)

        while controller.terminate:
            # Update best and personal fitness values based on the current positions
            # only iterate through particles that will move, thus continue if all velocities of a given particle are 0
            self._assess_fitness()

            # Update the informant fitness and velocity of all particles
            self._update_particle()

            # Move the particles based on their velocity
            self._move_particles()

            controller.next_iteration(self.best)




    def _assess_fitness(self):
        # evaluate and update fitness for each particle at current location 
        # Particle class: self.fitness should be updated here
        # update best
        for particle in self.particles:
            
            if not any(particle.velocity != 0):
                continue

            particle.assess_fitness(self.fitness_fn)

            if self.best is None or particle.fitness > self.best:
                self.best = FitnessLoc(particle.position, particle.fitness)



    def _update_particle(self):
        #TODO set the informant fitness and new velocity of all particles
        #! Doesnt move yet (this is important because the position of each particle affect how they all get a new velocity)
        # Its ok if the particles velocity would take it out of bounds, handle that in _move_particles()
        raise NotImplementedError()

    def _move_particles(self):
        #TODO change each particle position based on its velocity value
        #TODO handle out of bounds based on policies (see BoundaryPolicy enum at top of page)
        raise NotImplementedError()

    def _instantiate_particles(self):
        #depends on set_search_dimensions
        self.particles = [Particle(self._init_position(), self._init_velocity()) for i in range(self.swarm_size)]


    def _init_position(self):
        # Check the list in search_dimensions
        # randomly initialise the position vector pointwise WITHIN the boundary of search_dimension list
        # look at Particle class: Particle.position = new value
        #! returns a new value (see _instantiate_particles)
        return [random.uniform(d[0], d[1]) for d in self.search_dimension]


    def _init_velocity(self):
        #! Not the same as _move_particle (no need to consider the boundary here)
        # randomly initialise the velocity vector (depending on velocity init policy) pointwise for the size of search_dimension list
        # look at Particle class: Particle.velocity = new value
        #! returns a new value (see _instantiate_particles)
        # quick naive velocity solution, needs testing
        return [random.uniform(d[0], d[1]) for d in self.search_dimension]


class Particle:
    def __init__(self, position, velocity = None):
        self.position = position
        self.velocity = velocity
        self.fitness = None

        self.personal_fittest_loc = None
        self.informat_fittest_loc = None
        self.informants = None

    def assess_fitness(self, fitness_fn):
        """Assess the fitness of this particle

        :param fitness_fn: the function to call to produce a fitness value from the model, this function should take a vector describing all the model parameters as an arg
        :type fitness_fn: np.array -> float
        """
        # position describes the neural networks parameters
        self.fitness = FitnessLoc(self.position, fitness_fn(self.position))

        if self.fitness > self.personal_fittest_loc:
            self.personal_fittest_loc = FitnessLoc(self.position, self.fitness)



@total_ordering
class FitnessLoc:
    """A comparable class to store a fitness value and vector (position)
    """
    def __init__(self, location, fitness):
        self.location = location
        self.fitness = fitness   

    def _is_valid_operand(self, other):
        return(hasattr(other, "fitness") and hasattr(other, "location"))

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness == other.fitness

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.fitness < other.fitness

class IterationController:
    def __init__(self, termination_policy, max_iter=None, min_fitness_delta=None, time_delta=None):

        if type(termination_policy) is not list:
            self.termination_policy = [termination_policy]
        else:
            self.termination_policy = termination_policy
        
        if TerminationPolicy.ITERATIONS in self.termination_policy:
            if max_iter is None:
                raise ValueError("Max iteration undefined")

        if TerminationPolicy.DURATION in self.termination_policy:
            if time_delta is None:
                raise ValueError("Time delta undefined")

        if TerminationPolicy.CONVERGENCE in self.termination_policy:
            if min_fitness_delta is None:
                raise ValueError("Fitness delta undefined")
        
        # Loop until terminate is true
        self.terminate = False

        # Number of iterations so far
        self.current_iter = 0
        self.max_iter = max_iter

        # The fitness from the last iteration
        self.min_fitness_delta = min_fitness_delta
        self.current_fitness_delta = None
        self.got_fitness_delta = False

        self.start_time = datetime.now()
        self.time_delta = time_delta


    def next_iteration(self, fitness_delta=None):

        if TerminationPolicy.ITERATIONS in self.termination_policy:
            if self.max_iter < self.current_iter:
                self.terminate = True

        if TerminationPolicy.DURATION in self.termination_policy:
            elapsed = datetime.now() - self.start_time
            if elapsed > self.time_delta:
                self.terminate = True

        if TerminationPolicy.CONVERGENCE in self.termination_policy:
            if fitness_delta is not None and self.min_fitness_delta >= fitness_delta:
                self.terminate = True
            elif self.got_fitness_delta:
                self.got_fitness_delta = False
                if self.min_fitness_delta >= self.current_fitness_delta:
                    self.terminate = True
            else: 
                raise ValueError('Fitness delta unspecified')

        self.current_iter += 1

        def __iter__(self):
            return self

        def __next__(self):
            self.next_iteration()
            if self.terminate is True:
               raise StopIteration 
