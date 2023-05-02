import numpy as np
from potatorch.optimization import BayesianOptimizer

class HyperOptimizer():
  def __init__(self, sweep_config):
    self.sweep_config = sweep_config
    self.minimize = sweep_config['metric']['goal'] == 'minimize'
    self.metric = sweep_config['metric']['name']
    self.method = sweep_config['method']
    self.config = sweep_config.get('fixed', {})
    self.best_config = self.config.copy()
    self.best_metric = np.inf if self.minimize else -np.inf

  def _assign_state(self, state):
    """ Set the current hyperparameters configuration """
    for k in state.keys():
      self.config[k] = state[k]

  def _random_state(self, rng, parameters, ranges):
    """ Generate a random configuration of hypeparameters following the
        distributions indicated into the sweep configuration.
    """
    state = {}
    for i in range(len(parameters)):
      p = parameters[i]
      r = ranges[i]
      dist = r.get('distribution') # Determine random distribution to use

      # Check if the distribution is finite (and provided as a list of values)
      if 'values' in r and self.method == 'grid':
        # Uniform sampling within r['values']
        v = rng.choice(r['values'], replace=True)
      elif 'min' in r and 'max' in r:
        # Uniform sampling between min and max
        low = r['min']
        high = r['max']
        # Discrete distribution
        if isinstance(low, int) or isinstance(high, int):
          v = rng.integers(low, high, endpoint=True, size=None)
        # Continuos distribution
        else:
          v = rng.random() # Sample from U(0, 1)
          v = (high - low) * v + low # Convert into U(low, high), low <= v < high
      else:
        raise Exception('Invalid distribution for {}'.format(p))

      # Log distribution:
      # must convert the sampled value from logarithmic form into normal form
      # TODO: set logarithm base as a configuration parameter
      if dist and dist.startswith('log_'):
        v = 10.0**v
      
      assert v is not None
      state[p] = v
    return state

  def _grid_search(self, score_function, parameters, ranges):
    """ Perform am exhaustive grid search over the hyperparameters' space """
    # Base case: call the score function with the current parameters
    if len(parameters) == 0:
      print('Evaluating configuration: {}'.format(self.config))
      target = score_function(self.config)[self.metric]
      if (self.minimize and target < self.best_metric) or (not self.minimize and target > self.best_metric):
        self.best_metric = target
        self.best_config = self.config.copy()
      return
  
    # Recursively test every possible configuration of parameters
    for v in ranges[0]['values']:
      self.config[parameters[0]] = v
      self._grid_search(score_function, 
                        parameters[1:],
                        ranges[1:])

  def _random_search(self, score_function, parameters, ranges, iterations=30):
    """ Perform a random search over the hyperparameters' space """
    if iterations is None or iterations <= 0: 
      raise Exception('Must provide a positive number of iterations')

    # TODO: provide probability distributions other than uniform
    # Random number generator
    # Note: this method always uses a different initial random state;
    #       if reproducibility is required, pass a seed to default_rnd(seed)
    rng = np.random.default_rng()

    for _ in range(iterations):
      # Generate a random configuration of hyperparameters
      state = self._random_state(rng, parameters, ranges)
      self._assign_state(state)
      # Evaluate random hyperparameters configuration
      print('Evaluating configuration: {}'.format(self.config))
      target = score_function(self.config)[self.metric]
      if (self.minimize and target < self.best_metric) or (not self.minimize and target > self.best_metric):
        self.best_metric = target
        self.best_config = self.config.copy()

  def _bayesian(self, score_function, parameters, ranges, iterations=30, sample_size=500):
    """ Perform a bayesian search over the hyperparameters' space """
    # Random number generator
    rng = np.random.default_rng()

    state = self._random_state(rng, parameters, ranges)
    self._assign_state(state)
    
    # TODO encanpsulate this print into a lambda with score_function
    print('Evaluating configuration: {}'.format(self.config))
    score = score_function(self.config)[self.metric]
    opt = BayesianOptimizer(state, score, self.minimize) 
    self.best_metric = score
    self.best_config = self.config.copy()
    self.optimizer = opt

    history = [score]
    best_iteration = 0
    for i in range(1, iterations):
      # Sample `sample_size` configurations of hyperparameters
      sample = np.array([self._random_state(rng, parameters, ranges) for _ in range(sample_size)])
      new_state = opt.optimize(sample) # Get the next hyperparameters to test
      self._assign_state(new_state) # Update bayesian model

      print('Evaluating configuration: {}'.format(self.config))
      score = score_function(self.config)[self.metric]
      opt.update(new_state, score)

      if (self.minimize and score < self.best_metric) or (not self.minimize and score > self.best_metric):
        best_iteration = i
        self.best_metric = score
        self.best_config = self.config.copy()

      history.append(score)

  def optimize(self, score_function, return_error=False, window_iterations=1):
    """ Perform hyperparameters optimization.
        The search method is indicated under the 'method' key of the
        sweep configuration dictionary,

        returns: the best configuration of hyperparameters found by the method
    """
    parameters = list(self.sweep_config['parameters'].keys())
    values = list(self.sweep_config['parameters'].values())
    iterations = self.sweep_config.get('iterations')
    assert iterations or (self.method == 'grid'), 'When performing bayesian or random optimization you must specify the number of iterations'

    default_values = values

    for i in range(window_iterations):
        if self.method == 'grid':
          self._grid_search(score_function, parameters, values)
        elif self.method == 'bayes':
          self._bayesian(score_function, parameters, values, iterations, sample_size=100)
        elif self.method == 'random':
          self._random_search(score_function, parameters, values, iterations)
        else:
          raise Exception('No search method provided')

        if window_iterations > 1:
          w = 0.9**(i+1)
          centers = self.best_config
          values = self._narrow_ranges(values, default_values, parameters, centers, window_size=w)
          print(f'Best configuration so far: {centers}')
          print(f'New ranges: {values}')

    if return_error:
        return self.best_config, self.best_metric
    return self.best_config

  def _narrow_ranges(self, ranges, default, parameters, centers, window_size=0.2):
    new_ranges = ranges.copy()
    for i in range(len(parameters)):
      p = parameters[i]
      c = centers[p]

      low = new_ranges[i]['min']
      high = new_ranges[i]['max']
      new_min = max(c * (1 - window_size), default[i]['min'])
      new_max = min(c * (1 + window_size), default[i]['max'])
      if isinstance(low, int) or isinstance(high, int):
          new_min = int(new_min)
          new_max = int(new_max)

      new_ranges[i]['min'] = new_min
      new_ranges[i]['max'] = new_max

    return new_ranges
