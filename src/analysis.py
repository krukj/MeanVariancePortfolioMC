import numpy as np
import matplotlib.pyplot as plt
import imageio
from tabulate import tabulate

from src.PorfolioOptimizerSA import PortfolioOptimizerSA

def return_rate_influence(T_0: float,
                          T_f: float,
                          max_iter: int,
                          step_size: float,
                          annealing_rate: float,
                          probabilities: np.ndarray,
                          alpha: float,
                          S_0: np.ndarray,
                          S_T: np.ndarray,
                          V_0: float,
                          return_rates: list
                          ):
    results = {}
    
    for return_rate in return_rates:
        optimizer = PortfolioOptimizerSA(
                    T_0=T_0,
                    T_f=T_f,
                    max_iter=max_iter,
                    step_size=step_size,
                    annealing_rate=annealing_rate,
                    probabilities=probabilities,
                    alpha=alpha,
                    S_0=S_0,
                    S_T=S_T,
                    V_0=V_0,
                    return_rate=return_rate
                )
        optimizer.optimize()
        optimal_portfolio = optimizer.x
        optimal_CVaR = optimizer.calculate_CVaR(optimal_portfolio)
        
        results[return_rate] = {
            'optimal_portfolio': optimal_portfolio,
            'optimal_CVaR': optimal_CVaR
        }
        
    return results 

def plot_return_rate_influence(results: dict):
    
    return_rates = list(results.keys())
    optimal_CVaRs = [results[rr]['optimal_CVaR'] for rr in return_rates]
    
    plt.figure(figsize=(10, 6))
    plt.plot(return_rates, optimal_CVaRs, marker='o', color='goldenrod')
    plt.title('Influence of Return Rate on Optimal CVaR')
    plt.xlabel('Return Rate')
    plt.ylabel('Optimal CVaR')
    plt.grid()
    plt.show()
    

def plot_temperature_history(optimizer: PortfolioOptimizerSA):
    y = optimizer.temperature_history
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(12,8))
    plt.plot(x, y, color='navy')
    plt.title('Temperature vs. Iteraration')
    plt.xlabel('Iterations')
    plt.ylabel('Temperature')
    plt.grid(True)
    
def plot_cvar_history(T_0: float,
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
                          n_paths = 1):
    plt.figure(figsize=(12,8))
    for i in range(n_paths):
        optimizer = PortfolioOptimizerSA(
            T_0=T_0,
            T_f=T_f,
            max_iter=max_iter,
            step_size=step_size,
            annealing_rate=annealing_rate,
            probabilities=probabilities,
            alpha=alpha,
            S_0=S_0,
            S_T=S_T,
            V_0=V_0,
            return_rate=return_rate
        )
        optimizer.optimize()
        y = optimizer.objective_value_history
        x = np.arange(1, len(y) + 1)
        print(f"Path {i+1}: Final CVaR = {y[-1]:.4f}")
        
        plt.plot(x, y, color=plt.cm.Blues(np.random.uniform(0.5,1)))
    plt.title(f'CVaR vs. Iteraration ({n_paths} separate simulations)')
    plt.xlabel('Iterations')
    plt.ylabel('CVaR')
    plt.grid(True)
            
    

def plot_weights(optimizer: PortfolioOptimizerSA, filename: str, plot_every_k_iterations: int = 1, fps: int = 10) -> None:
    weights = optimizer.portfolio_history
    iterations = len(weights)
    
    cvars = optimizer.objective_value_history
    
    y_min = float("inf")
    y_max = -float("inf")
    
    for i in range(iterations):
        for j in range(len(weights[i])):
            y_min = min(y_min, np.min(weights[i][j]))
            y_max = max(y_max, np.max(weights[i][j]))

    y_min *= 1.05
    y_max *= 1.05
    images = []
            
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(0, iterations, plot_every_k_iterations):
        ax.clear()
        ax.bar(np.arange(len(weights[i])), weights[i], color='blue')
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Asset')
        ax.set_ylabel('Weight')
        current_cvar = cvars[i] if i < len(cvars) else None
        ax.set_title(f'Portfolio Weights at Iteration {i+1}\nCVaR: {current_cvar:.4f}' if current_cvar is not None else f'Portfolio Weights at Iteration {i+1}')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        
    fig_final, ax_final = plt.subplots(figsize=(8, 5))
    initial_weights = weights[0]
    final_weights = weights[-1]
    indices = np.arange(len(initial_weights))
    ax_final.bar(indices - 0.2, initial_weights, width=0.4, color='blue', label='Initial Weights (Iteration 1)')
    ax_final.bar(indices + 0.2, final_weights, width=0.4, color='orange', label='Final Weights')
    ax_final.set_xlabel('Asset')
    ax_final.set_ylabel('Weight')
    ax_final.set_title('Initial vs Final Portfolio Weights')
    ax_final.legend()
    fig_final.tight_layout()
    fig_final.canvas.draw()
    final_image = np.frombuffer(fig_final.canvas.tostring_rgb(), dtype='uint8')
    final_image = final_image.reshape(fig_final.canvas.get_width_height()[::-1] + (3,))
    images.append(final_image)
    plt.close(fig_final)
    images += [images[-1]] * fps
    imageio.mimsave(filename, images, fps=fps)
    plt.close(fig)
    

def plot_regulariztation_weights(T_0: float,
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
                          regularization_lambda: float,
                          initial_portfolio: np.ndarray = None):
    optimizer = PortfolioOptimizerSA(
        T_0=T_0,
        T_f=T_f,
        max_iter=max_iter,
        step_size=step_size,
        annealing_rate=annealing_rate,
        probabilities=probabilities,
        alpha=alpha,
        S_0=S_0,
        S_T=S_T,
        V_0=V_0,
        return_rate=return_rate,
        initial_portfolio=initial_portfolio
    )
    weights_initial = optimizer.x
    optimizer.optimize(regularization_lambda=regularization_lambda)
    weights_final_reg = optimizer.x
    cvar_reg = optimizer.calculate_CVaR(weights_final_reg, regularization_lambda=regularization_lambda)
    
    optimizer = PortfolioOptimizerSA(
        T_0=T_0,
        T_f=T_f,
        max_iter=max_iter,
        step_size=step_size,
        annealing_rate=annealing_rate,
        probabilities=probabilities,
        alpha=alpha,
        S_0=S_0,
        S_T=S_T,
        V_0=V_0,
        return_rate=return_rate,
        initial_portfolio=initial_portfolio
    )
    optimizer.optimize(regularization_lambda=0)
    weights_final_no_reg = optimizer.x
    cvar_no_reg = optimizer.calculate_CVaR(weights_final_no_reg, regularization_lambda=0)
    
    plt.figure(figsize=(12,8))
    indices = np.arange(len(weights_final_reg))
    plt.bar(indices - 0.3, weights_initial, width=0.2, color='green', label='Initial Portfolio (adjusted)')
    plt.bar(indices - 0.1, weights_final_no_reg, width=0.2, color='blue', label='No Regularization')
    plt.bar(indices + 0.1, weights_final_reg, width=0.2, color='orange', label='With Regularization')
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    plt.title(f'Portfolio Weights\nCVaR no reg: {cvar_no_reg:.4f}, CVaR reg: {cvar_reg:.4f}')
    plt.legend()
    plt.show()
    
    print(f"CVaR with regularization (lambda={regularization_lambda}): {cvar_reg:.2f}")
    print(f"CVaR without regularization: {cvar_no_reg:.2f}")
  
    

def analyze_different_portfolios(T_0: float,
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
                          initial_portfolios: list):
    results = {}
    
    for i, initial_portfolio in enumerate(initial_portfolios):
        optimizer = PortfolioOptimizerSA(
            T_0=T_0,
            T_f=T_f,
            max_iter=max_iter,
            step_size=step_size,
            annealing_rate=annealing_rate,
            probabilities=probabilities,
            alpha=alpha,
            S_0=S_0,
            S_T=S_T,
            V_0=V_0,
            return_rate=return_rate,
            initial_portfolio=initial_portfolio
        )
        optimizer.optimize()
        optimal_portfolio = optimizer.x
        optimal_CVaR = optimizer.calculate_CVaR(optimal_portfolio,)
        
        results[f'Portfolio {i+1}'] = {
            'optimal_portfolio': optimal_portfolio,
            'optimal_CVaR': optimal_CVaR
        }
        
    print(f"{'No':<3} {'initial portfolio':<50} {'optimal portfolio':<50} {'cvar':<10}")
    for idx, (key, value) in enumerate(results.items(), 1):
        print(f"{idx:<3} {str(initial_portfolios[idx-1]):<50} {str(value['optimal_portfolio']):<50} {value['optimal_CVaR']:.4f}")
    
    
def plot_regulariztation_weights(T_0: float,
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
                          regularization_lambda: float,
                          initial_portfolio: np.ndarray = None):
    optimizer = PortfolioOptimizerSA(
        T_0=T_0,
        T_f=T_f,
        max_iter=max_iter,
        step_size=step_size,
        annealing_rate=annealing_rate,
        probabilities=probabilities,
        alpha=alpha,
        S_0=S_0,
        S_T=S_T,
        V_0=V_0,
        return_rate=return_rate,
        initial_portfolio=initial_portfolio
    )
    weights_initial = optimizer.x
    optimizer.optimize(regularization_lambda=regularization_lambda)
    weights_final_reg = optimizer.x
    cvar_reg = optimizer.calculate_CVaR(weights_final_reg, regularization_lambda=regularization_lambda)
    
    optimizer = PortfolioOptimizerSA(
        T_0=T_0,
        T_f=T_f,
        max_iter=max_iter,
        step_size=step_size,
        annealing_rate=annealing_rate,
        probabilities=probabilities,
        alpha=alpha,
        S_0=S_0,
        S_T=S_T,
        V_0=V_0,
        return_rate=return_rate,
        initial_portfolio=initial_portfolio
    )
    optimizer.optimize(regularization_lambda=0)
    weights_final_no_reg = optimizer.x
    cvar_no_reg = optimizer.calculate_CVaR(weights_final_no_reg, regularization_lambda=0)
    
    plt.figure(figsize=(12,8))
    indices = np.arange(len(weights_final_reg))
    plt.bar(indices - 0.3, weights_initial, width=0.2, color='green', label='Initial Portfolio (adjusted)')
    plt.bar(indices - 0.1, weights_final_no_reg, width=0.2, color='blue', label='No Regularization')
    plt.bar(indices + 0.1, weights_final_reg, width=0.2, color='orange', label='With Regularization')
    plt.xlabel('Asset')
    plt.ylabel('Weight')
    plt.title(f'Portfolio Weights\nCVaR no reg: {cvar_no_reg:.4f}, CVaR reg: {cvar_reg:.4f}')
    plt.legend()
    plt.show()
    
    print(f"CVaR with regularization (lambda={regularization_lambda}): {cvar_reg:.2f}")
    print(f"CVaR without regularization: {cvar_no_reg:.2f}")
  
    

def probabilities_ifluence(T_0: float,
                          T_f: float,
                          max_iter: int,
                          step_size: float,
                          annealing_rate: float,
                          probabilities: list,
                          alpha: float,
                          S_0: np.ndarray,
                          S_T: np.ndarray,
                          V_0: float,
                          return_rate: float,
                          n_paths = 1):

    results = []
    for k, p in enumerate(probabilities):
        cvar_list = []
        for i in range(n_paths):
            optimizer = PortfolioOptimizerSA(
                T_0=T_0,
                T_f=T_f,
                max_iter=max_iter,
                step_size=step_size,
                annealing_rate=annealing_rate,
                probabilities=p,
                alpha=alpha,
                S_0=S_0,
                S_T=S_T,
                V_0=V_0,
                return_rate=return_rate)
            optimizer.optimize()
            cvar_list.append(optimizer.objective_value_history[-1])
        results.append({
            'probabilities': f'non-uniform' if k == 0 else 'uniform',
            'mean_CVaR': np.mean(cvar_list),
            'std_CVaR': np.std(cvar_list)
        })
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='github', floatfmt=('.2f', '.2f', '.2f')))  
    
def generate_scenarios(S_0, number_of_scenarios=1000):
    """
    Generuje losowe scenariusze na podstawie stanu początkowego.
    
    """
    
    p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    scenarios = np.zeros((number_of_scenarios, len(S_0)))
    for i in range(number_of_scenarios):
        coefs = [np.random.normal(0.3, 0.1), 
                 np.random.normal(0.9, 0.1),
                 np.random.normal(1, 0.1),
                 np.random.normal(1.2, 0.2),
                 np.random.normal(1.5, 1)]
        chosen = np.random.choice(coefs, size=len(S_0), p=p)
        
        # ensure at least one growth
        if not np.any(np.array(chosen) > 1):
            idx = np.random.randint(len(S_0))
            chosen[idx] = np.random.uniform(1.01, 1.1)
        
        S_T = S_0 * chosen
        scenarios[i] = S_T
    
    return scenarios
    
    
def examine_num_of_scenarios_influence(T_0: float,
                          T_f: float,
                          max_iter: int,
                          step_size: float,
                          annealing_rate: float,
                          alpha: float,
                          S_0: np.ndarray,
                          S_T: np.ndarray,
                          V_0: float,
                          return_rate: float,
                          nums_of_scenarios: list[int] = [100, 1000, 10000, 100000],
                          n_paths = 5,
                          probabilities: str = 'uniform'):
    
    res = {}
    
    plt.subplots(len(nums_of_scenarios), 1, figsize=(10, 5 * len(nums_of_scenarios)))
    
    for i, num in enumerate(nums_of_scenarios):
        S_T_subset = generate_scenarios(S_0, number_of_scenarios=num)
        for j in range(n_paths):
            if probabilities == 'uniform':
                probabilities_subset = np.ones(S_T_subset.shape[0]) / S_T_subset.shape[0]
            else:
                probabilities_subset = 1 / np.arange(1, S_T_subset.shape[0] + 1)
                probabilities_subset /= probabilities_subset.sum()
            
            optimizer = PortfolioOptimizerSA(
                T_0=T_0,
                T_f=T_f,
                max_iter=max_iter,
                step_size=step_size,
                annealing_rate=annealing_rate,
                probabilities=probabilities_subset,
                alpha=alpha,
                S_0=S_0,
                S_T=S_T_subset,
                V_0=V_0,
                return_rate=return_rate
            )
            optimizer.optimize()
            
            plt.subplot(len(nums_of_scenarios), 1, i + 1)
            plt.plot(optimizer.objective_value_history, color=plt.cm.Blues(np.random.uniform(0.5, 1)))
            plt.grid(True)  # Add grid

            if num not in res:
                res[num] = []
            res[num].append(optimizer.objective_value_history[-1])

        plt.xlabel('Iteracja')
        plt.ylabel('Wartość funkcji celu (CVaR)')
        plt.title(f'Liczba scenariuszy: {num}')
        plt.grid(True)  
        
        print(f'Zakończono obliczenia dla {num} scenariuszy.')

    plt.tight_layout()
    plt.show()
    for key, values in res.items():
        res[key] = values / np.max(values)
        
    plt.figure(figsize=(10, 5))
    box = plt.boxplot(list(res.values()), labels=[str(k) for k in res.keys()], patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(box['boxes'])))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.xlabel('Liczba scenariuszy')
    plt.ylabel('Znormalizowana wartość CVaR')
    # plt.yscale('log')
    plt.title('Wpływ liczby scenariuszy na wartość CVaR')
    plt.show()
    
    table = []
    for num, values in res.items():
        mean = np.mean(values)
        std = np.std(values)
        max_min = np.max(values) - np.min(values)
        table.append([num, mean, std, max_min])
        
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    nums = [row[0] for row in table]
    stds = [row[2] for row in table]
    max_mins = [row[3] for row in table]

    axs[0].plot(nums, stds, marker='o')
    axs[0].set_xlabel('Liczba scenariuszy')
    axs[0].set_ylabel('Odchylenie std')
    axs[0].set_title('Odchylenie std vs liczba scenariuszy')
    axs[0].set_xscale('log')
    axs[0].grid(True)
    
    axs[1].plot(nums, max_mins, marker='o', color='orange')
    axs[1].set_xlabel('Liczba scenariuszy')
    axs[1].set_ylabel('Max - Min')
    axs[1].set_title('Max - Min vs liczba scenariuszy')
    axs[1].set_xscale('log')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(tabulate(table, headers=["Liczba scenariuszy", "Średnia", "Odchylenie std", "Max - Min"], floatfmt=".6f"))
    return res