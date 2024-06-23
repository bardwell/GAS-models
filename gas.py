# Useful refs
# ScoreDrivenModels.jl: a Julia Package for Generalized Autoregressive Score Models
# https://stats.stackexchange.com/questions/215084/what-is-the-difference-between-gas-generalized-autoregressive-score-model-and
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

class GAS_Normal:
    
    def scaled_score(self, y, f):
        """
        Score multiplied by inverse Fisher information for
        y_t = sigma_t * epsilon_t
        where epsilon_t ~ N(0,1).	 
        """
        return (y**2 - f)/f
    
    def simulate(self, theta = (0.2,0.5,0.5), n = 100, seed = None):
        """
	Simulate from a GAS(1,1) model with parameter vector 
	theta = (omega, A, B) and length n. 
	"""
        omega = theta[0]
        A = theta[1]
        B = theta[2]
        rng = np.random.default_rng(seed)
        f = np.zeros(n)
        y = np.zeros(n)
        for i in range(n):
            if i == 0:
                f[i] = omega
                y[i] = np.sqrt(np.exp(f[i])) * rng.normal()
            else:
                # update log f 
                f[i] = omega + A*self.scaled_score(y[(i-1)], np.exp(f[(i-1)])) + B*f[(i-1)]
                y[i] = np.sqrt(np.exp(f[i])) * rng.normal()
        return y, f
    
    def log_likelihood(self, y, params):
        """
	Evaluate log-likelihood
	"""
        omega = params[0]
        A = params[1]
        B = params[2]
        f0 = params[3]
        s0 = params[4]
        ll = 0
        for i in range(len(y)):
            f = omega + A*s0 + B*f0
            sigma2 = np.exp(f)
            ll += -0.5*np.log(sigma2) - (y[i]**2)/(2*sigma2)
            f0 = f
            s0 = self.scaled_score(y[i], np.exp(f0))
        return ll

    def fit(self, y, x0):
        res = minimize(lambda x: -1*self.log_likelihood(y, x),
                       x0, method='nelder-mead',
                       options={'disp': True})
        print(f'omega = {res.x[0]}, A = {res.x[1]}, B = {res.x[2]}')

        
def plot_sims(y, f):
    # plot sims
    fig, ax = plt.subplots(2, sharex = True)
    ax[0].plot(y, label = 'y')
    ax[1].plot(f, label = 'f', color = 'red')
    ax[1].set_xlabel('Time')
    fig.legend(loc='upper left')
    fig.tight_layout()
    plt.show()

