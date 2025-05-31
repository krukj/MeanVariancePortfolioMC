import numpy as np


class PortfolioOptimizerSA:
    def __init__(self,
                 T_0: float,
                 T_f: float,
                 max_iter: int,
                 step_size: float,
                 annealing_rate: float,
                 probabilities: np.ndarray,
                 alpha: float,
                 S_0: np.ndarray,
                 S_T: np.ndarray,
                 V_0: float,
                 return_rate: float,
                 initial_portfolio: np.ndarray = None) -> None:
        """
        Initialize the PortfolioOptimizerSA object.

        Args:
            T_0 (float): Initial temperature for simulated annealing.
            T_f (float): Final temperature for simulated annealing.
            max_iter (int): Maximum number of iterations for simulated annealing.
            step_size (float): Step size for simulated annealing.
            annealing_rate (float): Annealing rate for simulated annealing.
            probabilities (np.ndarray): Array of probabilities corresponding to S_T.
            alpha (float): Risk aversion parameter.
            S_0 (np.ndarray): Initial prices.
            S_T (np.ndarray): Possible prices at time T.
            V_0 (float): Initial portfolio value.
            return_rate (float): Expected return rate.
            initial_portfolio (np.ndarray, optional): User-defined initial portfolio. Defaults to None.
        """
        
        # simulated annealing params
        self.T_0 = T_0
        self.T_f = T_f
        self.max_iter = max_iter
        self.step_size = step_size
        self.annealing_rate = annealing_rate

        # probabilistic params
        self.S_0 = S_0
        self.S_T = S_T
        self.probabilities = probabilities
        self.alpha = alpha
        self.return_rate = return_rate
        self.V_0 = V_0
        
        self._prepare_constraints()
        
        if initial_portfolio is not None:
            if len(initial_portfolio) != len(S_0):
                raise ValueError("Initial portfolio length must match the number of assets.")
            x = initial_portfolio.copy()
            self._correct_x(x)
            self.x = x
        else:
            self.x = self._initialize_portfolio()

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
            np.sum(self.probabilities.reshape(-1, 1) * (self.S_T - self.S_0), axis=0)
            - self.return_rate * self.S_0
        )

        c[0] = self.V_0
        c[1] = 0

        B_hat = B.copy()

        # TODO maybe try changing columns
        if np.abs(B[0, 0]) < 1e-10:
            raise ValueError("Cannot perform matrix reduction")

        row_scaling_factor = B_hat[1, 0] / B_hat[0, 0]
        B_hat[1, :] = B_hat[1, :] - B_hat[0, :] * row_scaling_factor
        c[1] -= c[0] * row_scaling_factor

        self.B_hat = B_hat
        self.c_hat = c

    def _correct_x(self, x: np.ndarray) -> None:
        x[1] = (self.c_hat[1] - np.dot(self.B_hat[1, 2:], x[2:])) / self.B_hat[1, 1]
        x[0] = (self.c_hat[0] - np.dot(self.B_hat[0, 1:], x[1:])) / self.B_hat[0, 0]

    def _initialize_portfolio(self) -> np.ndarray:
        """
        Initializes the portfolio.

        Returns:
            np.ndarray: The initialized portfolio.
        """
        x = np.zeros_like(self.S_0, dtype=np.float32)
        self._correct_x(x)

        return x

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

    def optimize(self, regularization_lambda=0) -> None:
        """
        Performs optimization using the Simulated Annealing algorithm.

        Returns:
            None
        """
        T = self.T_0
        iteration = 1
        best_x = self.x.copy()
        best_CVaR = self.calculate_CVaR(best_x,regularization_lambda=regularization_lambda)

        self.objective_value_history = []
        self.portfolio_history = []
        self.temperature_history = []

        while T > self.T_f and iteration <= self.max_iter:
            k = np.random.randint(2, len(self.x))
            delta = np.random.uniform(-self.step_size, self.step_size)
            x_new = self.x.copy()
            x_new[k] += delta
            self._correct_x(x_new)

            current_CVaR = self.calculate_CVaR(self.x, regularization_lambda=regularization_lambda)
            new_CVaR = self.calculate_CVaR(x_new,regularization_lambda=regularization_lambda)

            delta_CVaR = new_CVaR - current_CVaR
            p_accept = min(1, np.exp(-delta_CVaR / T))
            if np.random.rand() < p_accept:
                self.x = x_new
                current_CVaR = new_CVaR

            if new_CVaR < best_CVaR:
                best_CVaR = new_CVaR
                best_x = x_new.copy()

            self.objective_value_history.append(current_CVaR)
            self.portfolio_history.append(self.x.copy())
            self.temperature_history.append(T)

            T = self._annealing_schedule(T, iteration)
            iteration += 1

    def _calculate_Var(self, L_vals: np.ndarray) -> float:
        """
        Calculate the Value at Risk (VaR) for a given array of L_vals.

        Args:
            L_vals (np.ndarray): Array of L_vals.

        Returns:
            float: The Value at Risk (VaR) for the given L_vals.
        """
        L_val_proba_map = {L_vals[i]: self.probabilities[i] for i in range(len(L_vals))}
        L_vals_sorted = np.sort(L_vals)
        cum_probs = np.cumsum([L_val_proba_map[L_val] for L_val in L_vals_sorted])

        z_index = last_feasible_z_index = len(L_vals_sorted) - 1
        while z_index >= 0 and cum_probs[z_index] >= self.alpha:
            last_feasible_z_index = z_index
            z_index -= 1

        return L_vals_sorted[last_feasible_z_index]


    def calculate_CVaR(self, portfolio: np.ndarray, regularization_lambda: float = 0) -> float:
        """
        Calculates the Conditional Value at Risk (CVaR) for a given portfolio.

        Args:
            portfolio (np.ndarray): The portfolio weights.

        Returns:
            float: The CVaR value.
        """
        L_vals = -(self.S_T - self.S_0) @ portfolio
        beta = self._calculate_Var(L_vals)
        
        loss_filter = L_vals > beta
        excess_losses = L_vals[loss_filter] - beta
        corresponding_probs = self.probabilities[loss_filter]
        
        expected_excess_loss = np.dot(excess_losses, corresponding_probs)
        CVaR = beta + expected_excess_loss / (1 - self.alpha)
        penalty = regularization_lambda * np.sum(portfolio**2)
        CVaR += penalty
        
        return CVaR