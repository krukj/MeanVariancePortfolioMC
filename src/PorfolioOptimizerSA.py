import numpy as np


class PortfolioOptimizerSA:
    def __init__(
        self,
        T_0: float,
        T_f: float,
        max_iter: int,
        step_size: float,
        annealing_rate: float,
        states: np.ndarray,
        probabilities: np.ndarray,
        alpha: float,
        S_0: float,
        V_0: float,
        return_rate: float,
        user_portfolio: np.ndarray = None,
    ) -> None:
        """
        Initialize the PortfolioOptimizerSA object.

        Args:
            T_0 (float): Initial temperature for simulated annealing.
            T_f (float): Final temperature for simulated annealing.
            max_iter (int): Maximum number of iterations for simulated annealing.
            step_size (float): Step size for simulated annealing.
            annealing_rate (float): Annealing rate for simulated annealing.
            states (np.ndarray): Array of possible states for the portfolio.
            probabilities (np.ndarray): Array of probabilities corresponding to the states.
            alpha (float): Risk aversion parameter.
            S_0 (float): Initial state.
            V_0 (float): Initial portfolio value.
            return_rate (float): Expected return rate.
            user_portfolio (np.ndarray, optional): User-defined initial portfolio. Defaults to None.
        
        Returns:
            None
        """
        
        # simulated annealing params
        self.T_0 = T_0
        self.T_f = T_f
        self.max_iter = max_iter
        self.step_size = step_size
        self.annealing_rate = annealing_rate

        # probabilistic params
        self.S_0 = S_0
        self.states = states
        self.probabilities = probabilities
        self.alpha = alpha
        self.return_rate = return_rate
        self.V_0 = V_0
        
        self._prepare_constraints()

        if user_portfolio is not None:
            self.x = user_portfolio
        else:
            self.x = self._initialize_portfolio()

    def _annealing_schedule(self, T: float, iteration: int) -> float:
        """
        Calculate the temperature for simulated annealing based on the current iteration.

        Args:
            T (float): The initial temperature.
            iteration (int): The current iteration number.

        Returns:
            float: The updated temperature for the given iteration.
        """
        return T * (self.annealing_rate - 0.2 * (iteration / self.max_iter))

    def _initialize_portfolio(self) -> np.ndarray:
        """
        Initializes the portfolio by generating a random feasible portfolio.

        Returns:
            np.ndarray: The initialized portfolio.

        Raises:
            ValueError: If a feasible portfolio cannot be found after a maximum number of attempts.
        """
        n = len(self.S_0)
        max_tries = 2000

        for _ in range(max_tries):
            x = np.zeros(shape=n)
            x[2:] = np.random.uniform(-0.5, 0.5, size=n - 2)

            x[1] = (self.c_hat[1] - np.sum(self.B_hat[1, 2:] * x[2:])) / self.B_hat[1, 1]
            x[0] = (self.c_hat[0] - np.sum(self.B_hat[0, 1:] * x[1:])) / self.B_hat[0, 0]

            budget_constraint = np.abs(np.dot(x, self.S_0) - self.V_0)

            expected_return = np.dot(
                x,
                np.sum(
                    self.probabilities.reshape(-1, 1) * (self.states - self.S_0), axis=0
                ),
            )
            target_return = self.return_rate * self.V_0
            return_constraint = np.abs(expected_return - target_return)

            if budget_constraint <= 1e-5 and return_constraint <= 1e-5:
                return x

        raise ValueError(
            f"Failed to find a feasible portfolio after {max_tries} of attempts."
        )

    def _prepare_constraints(self) -> None:
        """
        Prepares the constraints for the portfolio optimization problem.

        Raises:
            ValueError: If matrix reduction cannot be performed due to a zero value
                in the first element of the first row of B.
        """
        n = len(self.S_0)
        B = np.zeros(shape=(2, n))
        c = np.zeros(shape=2)

        B[0, :] = self.S_0
        B[1, :] = (
            np.sum(self.probabilities.reshape(-1, 1) * (self.states - self.S_0), axis=0)
            - self.return_rate * self.S_0
        )

        c[0] = self.V_0
        c[1] = 0

        B_hat = B.copy()

        if np.abs(B[0, 0]) < 1e-10:
            raise ValueError("Cannot perform matrix reduction")

        B_hat[1, :] = B_hat[1, :] - B_hat[0, :] * B_hat[1, 0] / B_hat[0, 0]

        self.B_hat = B_hat
        self.c_hat = c

    def optimize(self) -> None:
        """
        Performs optimization using the Simulated Annealing algorithm.

        Returns:
            None
        """
        T = self.T_0
        iteration = 1
        best_x = self.x.copy()
        best_CVaR = self._calculate_CVaR(best_x)

        while T > self.T_f and iteration <= self.max_iter:
            k = np.random.randint(2, len(self.x))
            delta = np.random.uniform(-self.step_size, self.step_size)
            x_new = self.x.copy()
            x_new[k] += delta
            x_new[1] = (
                self.c_hat[1] - np.sum(self.B_hat[1, 2:] * x_new[2:])
            ) / self.B_hat[1, 1]
            x_new[0] = (
                self.c_hat[0] - np.sum(self.B_hat[0, 1:] * x_new[1:])
            ) / self.B_hat[0, 0]

            current_CVaR = self._calculate_CVaR(self.x)
            new_CVaR = self._calculate_CVaR(x_new)

            delta_CVaR = new_CVaR - current_CVaR
            p_accept = min(1, np.exp(-delta_CVaR / T))
            if np.random.rand() < p_accept:
                self.x = x_new
                current_CVaR = new_CVaR

            if new_CVaR < best_CVaR:
                best_CVaR = new_CVaR
                best_x = x_new.copy()

            T = self._annealing_schedule(T, iteration)
            iteration += 1

    def _calculate_CVaR(self, portfolio: np.ndarray) -> float:
        """
        Calculates the Conditional Value at Risk (CVaR) for a given portfolio.

        Args:
            portfolio (np.ndarray): The portfolio weights.

        Returns:
            float: The CVaR value.
        """
        L_vals = -(self.states - self.S_0) @ portfolio
        beta = np.percentile(L_vals, self.alpha * 100)
        excess_losses = L_vals[L_vals > beta] - beta
        CVaR = beta + np.mean(excess_losses) / (1 - self.alpha)
        return CVaR
