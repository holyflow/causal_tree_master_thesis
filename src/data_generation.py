import numpy as np

class constant_dgp:
    """A data-generating process (DGP) where the treatment effect is constant across all units.

    Attributes:s
        p (int): Number of covariates.
        xi (float): Probability of treatment assignment.
        c0 (float): Constant term for control potential outcome.
        c1 (float): Constant term for treated potential outcome.
        sigma0 (float): Standard deviation of noise for control potential outcome.
        sigma1 (float): Standard deviation of noise for treated potential outcome.
    """
    def __init__(self, p=1, xi=0.5, c0=1.0, c1=3.0, sigma0=1.0, sigma1=1.0):
        self.p = p
        self.xi = xi
        self.c0 = c0
        self.c1 = c1
        self.sigma0 = sigma0
        self.sigma1 = sigma1

    def sample(self, n_obs: int, seed: int, distribution: str, outcome_mechanism: str, **kwargs,
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

        if distribution == "normal":
            X = rng.normal(0.0, 1.0, size=(n_obs, self.p))

        elif distribution == "uniform":
            X = rng.uniform(0,100,size=(n_obs, self.p))

        else: 
            raise ValueError("Unkown distribution specified")
        
        T = rng.binomial(n=1, p=self.xi, size=n_obs)
        te_scalar = self.c1 - self.c0
        treatment_effect = np.full(n_obs, te_scalar)
        
        if outcome_mechanism == "linear":
            beta_values = [2.0, -1.5, 0.5]

            beta = np.zeros(self.p)
            beta[:len(beta_values)] = beta_values[:self.p]

            mu_x = X @ beta

        elif outcome_mechanism == "nonlinear":
            if self.p < 2:
                mu_x = 2 * (X[:, 0] ** 2)
            else:
                mu_x = 2 * (X[:, 0] ** 2) + np.sin(X[:, 1])

        else:
            raise ValueError("Unknown outcome mechanism specified.")

        eps0 = rng.normal(0.0, self.sigma0, size=n_obs)  
        eps1 = rng.normal(0.0, self.sigma1, size=n_obs)
        y0 = self.c0 + mu_x + eps0
        y1 = self.c1 + mu_x + eps1
        y = T * y1 + (1 - T) * y0

        return X, y, T, treatment_effect
    


class heterogeneous_dgp(constant_dgp):
    """A data-generating process (DGP) where the treatment effect varies across units.

    Inherits from constant_dgp but modifies the potential outcomes to include
    heterogeneity based on covariates.
    """
    def sample(self, n_obs: int, seed: int, distribution: str, outcome_mechanism: str, treatment_mechanism: str | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        if distribution == "normal":
            X = rng.normal(0.0, 1.0, size=(n_obs, self.p))

        elif distribution == "uniform":
            X = rng.uniform(0,100,size=(n_obs, self.p))

        else: 
            raise ValueError("Unkown distribution specified")        
               
        T = rng.binomial(n=1, p=self.xi, size=n_obs)

        if outcome_mechanism == "linear":
            beta_values = [2.0, -1.5, 0.5]

            beta = np.zeros(self.p)
            beta[:len(beta_values)] = beta_values[:self.p]

            mu_x = X @ beta

        elif outcome_mechanism == "nonlinear":
            if self.p < 2:
                mu_x = 2 * (X[:, 0] ** 2)
            else:
                mu_x = 2 * (X[:, 0] ** 2) + np.sin(X[:, 1])

        else:
            raise ValueError("Unknown outcome mechanism specified.")

        if treatment_mechanism == "first_feature":
            treatment_effect = 1.0 + 2 * X[:, 0]  # Effect varies with x1

        elif treatment_mechanism == "positive_features":
            if self.p == 1:
                treatment_effect = 1.0 + (X[:, 0] > 0)  # Effect is higher for positive x1

            elif self.p == 2:
                treatment_effect = 1.0 + (X[:, 0] > 0) + (X[:, 1] > 0)  # Effect increases with positive x1 and x2

            else:
                treatment_effect = 1.0 + (X[:, 0] > 0) + (X[:, 1] > 0) + (X[:, 2] > 0)  # Effect increases with positive x1, x2, x3

        else:
            raise ValueError("Unknown treatment mechanism specified.")
        
        eps0 = rng.normal(0.0, self.sigma0, size=n_obs)
        eps1 = rng.normal(0.0, self.sigma1, size=n_obs)

        y0 = mu_x + eps0
        y1 = mu_x + treatment_effect + eps1
        y = T * y1 + (1 - T) * y0

        return X, y, T, treatment_effect
    
    


