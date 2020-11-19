from datetime import timedelta
import numpy as np
import random
import copy
from tqdm.autonotebook import tqdm
from .psobehaviour import FitnessLoc, TerminationPolicyManager, TerminationPolicy, BoundaryPolicy

all_term_policy = [TerminationPolicy.ITERATIONS, TerminationPolicy.CONVERGENCE, TerminationPolicy.DURATION]
class PSO:
    """Particle Swarm Optimiser

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

    def __init__(self, swarm_size=10, num_informants=6, bound=(1, -1), alpha=0.1, beta=1.3, gamma=1.4, delta=1.3, epsilon=0.1,  boundary_policy=BoundaryPolicy.RANDOMREINIT, termination_policy=all_term_policy, termination_args={'max_iter': int(1e6), 'time_delta': timedelta(minutes=4), 'min_fitness_delta': 0}):
        self.swarm_size = swarm_size
        self.boundary = bound
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        self.boundary_policy = boundary_policy
        self.termination_policy = termination_policy
        self.termination_args = termination_args

        self.search_dimension = None
        self.search_dimension_set = False
        self.best = None
        self.previous_best = FitnessLoc([], -999999.0)
        self.particles = None
        self.num_informants = num_informants

        self.fitness_fn = None # The arg to this is the shape of the ANN (wieghts + activation)

        self.verbose = True


    def set_fitness_fn(self, fitness_function):
        """Specify the function to use to calculate the fitness score

        :param fitness_function: a function object that can assess the fitness based on a vector
        :type fitness_function: numpy.array -> bool
        """
        self.fitness_fn = fitness_function
        #print(fitness_function)



    def set_search_dimensions(self, dimensions):
        """Specify the dimensionality of the search

        :param dimensions: define dimensionality. Either with an int (using default boundaries), or a list/np.array of tuples descriping the boundaries for each dimension
        :type dimensions: int / list
        :raises ValueError: When dimension parameter does not meet specified requirements
        """
        # set by a list of tuples || an integer -- eg. search_dimension = 3 => [(-1, 1), (-1, 1), (-1, 1)]
        if type(dimensions) is int:
            self.search_dimension = [self.boundary for _ in range(dimensions)]

        elif type(dimensions) is list:
            #TODO check list is valid
            self.search_dimension = dimensions

        else:
            self.search_dimension_set = False
            raise ValueError("Invalid dimensions parameter")

        self.search_dimension_set = True
        

    def run(self):
        """Begin Particle Swarm Optimisation - Search dimensions must have been specified
        """
        if not self.search_dimension_set:
            raise ValueError('Search dimentions have not yet been specified')
        self._instantiate_particles()
        self.best = FitnessLoc([], -9999.0)

        controller = TerminationPolicyManager(TerminationPolicy.ITERATIONS, **self.termination_args)

        if self.verbose:
            pbar = tqdm(total=100, position=0, leave=True, desc='Fitness: {}'.format(self.best.fitness))

        while not controller.terminate:
            # Update best and personal fitness values based on the current positions
            # only iterate through particles that will move, thus continue if all velocities of a given particle are 0
            self.pso_assess_fitness()

            # Update the informant fitness and velocity of all particles
            self._update_particle()

            # Move the particles based on their velocity
            self._move_particles()

            fitness_delta = (self.best.fitness - self.previous_best.fitness)

            if self.verbose:
                pbar.update(controller.estimate_progress()*100)
                pbar.set_description(
                    desc='Fitness: {}'.format(self.best.fitness), refresh=True)

            
            controller.next_iteration(fitness_delta=fitness_delta)

        if self.verbose:
            pbar.close()

        #print('Iteration: ', controller.current_iter)
        #print('Fitness: ', self.best.fitness)
        return self.best

    def pso_assess_fitness(self):
        # evaluate and update fitness for each particle at current location 
        # Particle class: self.fitness should be updated here
        # update best

        for particle in self.particles:
            
            #if not all(v != 0 for v in iter(particle.velocity)):
            #    continue
            #if not any(particle.velocity != 0):
            #    continue

            particle.assess_fitness()

            if self.best is None or particle.fitness_loc > self.best:
                self.previous_best = copy.deepcopy(self.best)
                self.best = copy.deepcopy(particle.fitness_loc)


    def _update_particle(self):
        #TODO set the informant fitness and new velocity of all particles
        #! Doesnt move yet (this is important because the position of each particle affect how they all get a new velocity)
        # Its ok if the particles velocity would take it out of bounds, handle that in _move_particles()
        for particle in self.particles:
            velocity = copy.deepcopy(particle.velocity)
            #if not any(particle.velocity != 0):
            #    continue
            particle.velocity_list.append(velocity)
            fittest_informant_loc = FitnessLoc([], -999999.0)
            for informant in particle.informants:
                if fittest_informant_loc < informant.fitness_loc:
                    fittest_informant_loc = informant.fitness_loc

            prev_fittest_loc = particle.personal_fittest_loc
            prev_fittest_loc_informants = fittest_informant_loc
            for i in range(len(self.search_dimension)):
                b = random.uniform(0.0, self.beta)
                c = random.uniform(0.0, self.gamma)
                d = random.uniform(0.0, self.delta)
                #TODO once informants implemented c*(...) part might need to be corrected
                velocity[i] = self.alpha * velocity[i] + b*(prev_fittest_loc.location[i] - particle.position[i]) + c*(prev_fittest_loc_informants.location[i] - particle.position[i]) + d*(self.best.location[i] - particle.position[i])
            particle.update_velocity(velocity)

    def _move_particles(self):
        #TODO change each particle position based on its velocity value
        #TODO handle out of bounds based on policies (see BoundaryPolicy enum at top of page)
        for particle in self.particles:
            #if not any(particle.velocity != 0):
            #    continue

            temp_position = particle.position + (self.epsilon*particle.velocity)

            # if position not within boundaries use appropriate boundary policy
            # else update particle position at dimension d
            # TODO: refuse only the dimension it is out of bounds with or refuse all dimensions?
            for index, d in enumerate(self.search_dimension):
                if not (d[0] <= temp_position[index] <= d[1]):

                    # TODO Bounce might be totally wrong, requires code review
                    if self.boundary_policy == BoundaryPolicy.BOUNCE:
                        #! Bug below, self.boundary[index wont work]
                        distance_left = temp_position[index] - self.boundary[index]
                        temp_position[index] = self.boundary[index] - \
                            distance_left

                    elif self.boundary_policy == BoundaryPolicy.RANDOMREINIT:
                        temp_position[index] = random.uniform(d[0], d[1])

                    # else - REFUSE, do nothing
                    elif self.boundary_policy == BoundaryPolicy.REFUSE:
                        temp_position[index] = particle.position[index]
            particle.update_position(temp_position)


    def _instantiate_particles(self):
        #depends on set_search_dimensions
        if self.fitness_fn is None:
            raise ValueError('No fitness function defined')

        self.particles = [Particle(self._init_position(), self._init_velocity(), self.fitness_fn) for _ in range(self.swarm_size)]
        self._init_informants()



    def _init_position(self):
        # Check the list in search_dimensions
        # randomly initialise the position vector pointwise WITHIN the boundary of search_dimension list
        # look at Particle class: Particle.position = new value
        #! returns a new value (see _instantiate_particles)
        return np.array([random.uniform(d[0], d[1]) for d in self.search_dimension])


    def _init_velocity(self):
        #! Not the same as _move_particle (no need to consider the boundary here)
        # randomly initialise the velocity vector (depending on velocity init policy) pointwise for the size of search_dimension list
        # look at Particle class: Particle.velocity = new value
        #! returns a new value (see _instantiate_particles)
        # quick naive velocity solution, needs testing
        return np.array([random.uniform(d[0], d[1]) for d in self.search_dimension])


    def _init_informants(self):
        # choose how many n informants each particle will have (variable self.num_informants)
        # assign randomly n informants to each particle
        for particle in self.particles:
            no_self = np.delete(np.array(self.particles),
                                np.where(np.array(self.particles) == particle ))
            particle.set_informants(np.random.choice(no_self, self.num_informants, replace=False))


class Particle:
    """A particle within a PSO optimiser

        :param position: The start position of the Particle
        :type position: numpy.array
        :param velocity: The initial velocity of the particle, defaults to None
        :type velocity: numpy.array, optional
    """
    def __init__(self, position, velocity = None, fitness_fn=None):
        self.position = position
        self.velocity = velocity
        self.fitness_loc = None
        self.fitness_fn = fitness_fn

        self.personal_fittest_loc = None
        self.informat_fittest_loc = None
        self.informants = None
        self.velocity_list = []

    def update_position(self, new_position):
        self.position = new_position

    def update_velocity(self, new_velocity):
        self.velocity = new_velocity

    def set_informants(self, informants):
        self.informants = informants

    def assess_fitness(self):
        """Assess the fitness of this particle

        :param fitness_fn: the function to call to produce a fitness value from the model, this function should take a vector describing all the model parameters as an arg
        :type fitness_fn: np.array -> float
        """
        # position describes the neural networks parameters
        fitness = self.fitness_fn(self.position)
        self.fitness_loc = FitnessLoc(copy.deepcopy(self.position), fitness)

        if self.personal_fittest_loc is None:
            self.personal_fittest_loc = self.fitness_loc

        if self.fitness_loc.fitness > self.personal_fittest_loc.fitness:
            self.personal_fittest_loc = self.fitness_loc


