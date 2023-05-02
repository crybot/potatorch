import numpy as np
from scipy.stats import norm

class ValueEncoder():
  """ Encode categorical python objects into numbers """
  def __init__(self):
    self.map = {}
    self.code = 0
  
  def encode(self, value):
    """ Encode boolean and string values into unique integers """
    if isinstance(value, bool):
      return int(value)

    if not isinstance(value, str):
      return value
    
    if value not in self.map: 
      self.map[value] = self.code
      self.code += 1
    
    return self.map[value]

  def encode_dict(self, d):
    """ Encode every entry of a dictionary """
    new_d = {}
    for k in d.keys():
      new_d[k] = self.encode(d[k])
    return new_d

class GaussianProcessRegressor():
  def __init__(self, sigma):
    self.sigma = sigma
    self.sigma2 = sigma**2
  
  def k(self, x1, x2):
    """ Gaussian kernel """
    return np.exp(-10/2 * np.linalg.norm(x1 - x2)**2)

  def gram(self, xs1, xs2):
    """ Gram (covariance) matrix """
    return np.array([[self.k(x1, x2) for x2 in xs2] for x1 in xs1])
    
  def fit(self, x, y):
    """ Fit the model with the given dataset D = {(x,y)}"""
    self.x_train = x
    self.y_train = y
    self.C = self.gram(x, x) + self.sigma2*np.eye(len(x))
    self.C_inv = np.linalg.inv(self.C)
  
  def predict(self, x):
    """ Perform inference on a new set of data points """
    K = self.gram(self.x_train, x)
    mu = K.T.dot(self.C_inv).dot(self.y_train)
    mu = mu.reshape((len(x),))
    cov = self.gram(x, x) - K.T.dot(self.C_inv).dot(K)
    return mu, cov

  def sample(self, x, n=1):
    """ Sample a function from the posterior distribution modeled by the GP """
    mu, cov = self.predict(x)
    return np.random.multivariate_normal(mu, cov, size=n)

class BayesianOptimizer():
  """ Bayesian function optimizer:
    Perform optimization in a bayesian setting by modelling the function's
    likelihood distribution with a Gaussian Process, which is then used as a
    surrogate function to optimize the original function.
    The acquisition function determines the next point to evaluate.
    New evaluated points can be provided to the optimizer with the `update`
    method, which will in turn improve the Bayesian model's beliefs.
    -----------------------------------
    Parameters:
  - initial_x:   intial configuration of hyperparameters (dictionary object)
  - initial_y:   value of score_function(initial_x) (scalar value)
  - minimize:    whether to minimize or maximize the function
  """
  def __init__(self, initial_x, initial_y, minimize=True, sigma=1e-1):
    self.gp = GaussianProcessRegressor(sigma=sigma)
    self.encoder = ValueEncoder()
    # Convert initial_x into a vector of numbers (see ValueEncoder)
    self.x = np.array([list(self.encoder.encode_dict(initial_x).values())])
    self.y = np.array([initial_y])
    self.minimize = minimize
    self.gp.fit(self.x, self.y)

  def acquisition(self, x_sampled):
    """ Acquisition function: excepted improvement """
    mu, _ = self.gp.predict(self.x)
    best = np.min(mu) if self.minimize else np.max(mu)
    mu, cov = self.gp.predict(x_sampled)
    mu = mu.reshape((len(x_sampled),))
    std = np.sqrt(np.diag(cov))
    sign = 1 if self.minimize else -1
    Z = lambda mu, std: sign*(best - mu)/(std**2) if std > 0 else 0.0

    improvements = [
             sign*(best - mu_x)*norm.cdf(Z(mu_x, std_x)) +
             (std_x**2)*norm.pdf(Z(mu_x, std_x))
              for mu_x, std_x in zip(mu, std)
               ]
    return improvements

  def optimize(self, sample):
    """ Perform an iteration of bayesian optimization """
    # Convert each sample into a vector of numbers
    encoded = np.array([list(self.encoder.encode_dict(x).values()) for x in sample])
    scores = self.acquisition(encoded)
    best_ix = np.argmax(scores) # always take the max (maximizing expectation of improvement)
    query = sample[best_ix]
    return query
  
  def update(self, x, y):
    """ Update the bayesian model's belief """
    x = np.array(list(self.encoder.encode_dict(x).values()))
    self.x = np.vstack((self.x, [x]))
    self.y = np.vstack((self.y, [y]))
    self.gp.fit(self.x, self.y)
