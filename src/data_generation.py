import numpy as np

class constant_dgp:
    """A data-generating process (DGP) where the treatment effect is constant across all units.

    Attributes:
        p (int): Number of covariates.
        xi (float): Probability of treatment assignment.
        c0 (float): Constant term for control potential outcome.
        c1 (float): Constant term for treated potential outcome.
        sigma0 (float): Standard deviation of noise for control potential outcome.
        sigma1 (float): Standard deviation of noise for treated potential outcome.
    """
    def __init__(self, p=1, xi=0.5, c0=10, c1=30, sigma0=100, sigma1=100):
        self.p = p
        self.xi = xi
        self.c0 = c0
        self.c1 = c1
        self.sigma0 = sigma0
        self.sigma1 = sigma1

    def sample(self, n_obs: int, seed: int,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples data from the constant treatment effect DGP.

        Args:
            n_obs (int): Number of observations to generate.
            seed (int | None): Random seed for reproducibility.

        Returns:
            Arrays of covariates X, outcomes y, and treatment assignments d,
            each of length n_obs.
        """
        rng = np.random.default_rng(seed)

        X = rng.uniform(0,100,size=(n_obs, self.p))
        
        T = rng.binomial(n=1, p=self.xi, size=n_obs)
        te_scalar = self.c1 - self.c0
        treatment_effect = np.full(n_obs, te_scalar)
        
        mu_x = 0

        eps0 = rng.normal(0.0, self.sigma0, size=n_obs)  
        eps1 = rng.normal(0.0, self.sigma1, size=n_obs)
        y0 = self.c0 + mu_x + eps0  
        y1 = self.c1 + mu_x + eps1
        y = T * y1 + (1 - T) * y0

        return X, y, T, treatment_effect

    
    


