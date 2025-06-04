import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from tabulate import tabulate

from src.PorfolioOptimizerSA import PortfolioOptimizerSA


def return_rate_influence(
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
    return_rates: list,
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
            return_rate=return_rate,
        )
        optimizer.optimize()
        optimal_portfolio = optimizer.x
        optimal_CVaR = optimizer.calculate_CVaR(optimal_portfolio)

        results[return_rate] = {
            "optimal_portfolio": optimal_portfolio,
            "optimal_CVaR": optimal_CVaR,
        }

    return results


def plot_return_rate_influence(
    results: dict, relative: bool = False, V_0: float = None
):
    return_rates = list(results.keys())
    optimal_CVaRs = [results[rr]["optimal_CVaR"] for rr in return_rates]

    if relative and V_0 is not None:
        optimal_CVaRs = [cvar / V_0 for cvar in optimal_CVaRs]
        ylabel = "Optymalne CVaR/V0"
    else:
        ylabel = "Optymalne CVaR"

    plt.figure(figsize=(10, 6))
    plt.plot(return_rates, optimal_CVaRs, marker="o", color="goldenrod")
    plt.title("Wpływ wymaganej stopy zwrotu na ryzyko portfela")
    plt.xlabel("Wymagana stopa zwrotu")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_temperature_history(optimizer: PortfolioOptimizerSA):
    y = optimizer.temperature_history
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, color="navy")
    plt.title("Zmiana temperatury w procesie symulowanego wyżarzania")
    plt.xlabel("Numer iteracji")
    plt.ylabel("Temperatura")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_cvar_history(
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
    n_paths: int = 1,
    relative: bool = False,
):
    plt.figure(figsize=(12, 8))
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
            return_rate=return_rate,
        )
        optimizer.optimize()
        y = optimizer.objective_value_history
        if relative:
            y = [val / V_0 for val in y]
        x = np.arange(1, len(y) + 1)
        if relative:
            print(f"Ścieżka {i+1}: Końcowe CVaR/V0 = {y[-1]:.4f}")
        else:
            print(f"Ścieżka {i+1}: Końcowe CVaR = {y[-1]:.4f}")

        plt.plot(x, y, color=plt.cm.Blues(np.random.uniform(0.5, 1)))

    ylabel = "CVaR/V0" if relative else "CVaR"
    plt.title(
        f"Zmiana {ylabel} w procesie optymalizacji ({n_paths} niezależnych symulacji)"
    )
    plt.xlabel("Numer iteracji")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_weights(
    optimizer: PortfolioOptimizerSA,
    filename: str,
    plot_every_k_iterations: int = 1,
    fps: int = 10,
    relative: bool = False,
    V_0: float = None,
) -> None:
    weights = optimizer.portfolio_history
    iterations = len(weights)
    cvars = optimizer.objective_value_history

    if relative and V_0 is not None:
        cvars = [cvar / V_0 for cvar in cvars]
        cvar_label = "CVaR/V0"
    else:
        cvar_label = "CVaR"

    y_min = min(np.min(w) for w in weights) * 1.05
    y_max = max(np.max(w) for w in weights) * 1.05

    fig, ax = plt.subplots(figsize=(10, 6))
    images = []

    for i in range(0, iterations, plot_every_k_iterations):
        ax.clear()
        indices = np.arange(len(weights[i])) + 1
        ax.bar(indices, weights[i])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Numer aktywa")
        ax.set_ylabel("Waga w portfelu")

        current_cvar = cvars[i] if i < len(cvars) else None
        title = (
            f"Rozkład wag w iteracji {i+1}\n{cvar_label}: {current_cvar:.4f}"
            if current_cvar is not None
            else f"Rozkład wag w iteracji {i+1}"
        )
        ax.set_title(title)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    fig_final, ax_final = plt.subplots(figsize=(10, 6))
    initial_weights = weights[0]
    final_weights = weights[-1]
    indices = np.arange(len(initial_weights)) + 1

    ax_final.bar(
        indices - 0.2, initial_weights, width=0.4, color="blue", label="Wagi początkowe"
    )
    ax_final.bar(
        indices + 0.2, final_weights, width=0.4, color="orange", label="Wagi końcowe"
    )
    ax_final.set_xlabel("Numer aktywa")
    ax_final.set_ylabel("Waga w portfelu")
    ax_final.set_title("Porównanie wag początkowych i końcowych")
    ax_final.legend()
    ax_final.grid(True, axis="y", linestyle="--", alpha=0.7)
    fig_final.tight_layout()

    fig_final.canvas.draw()
    final_image = np.frombuffer(fig_final.canvas.tostring_rgb(), dtype="uint8")
    final_image = final_image.reshape(fig_final.canvas.get_width_height()[::-1] + (3,))
    images.append(final_image)
    plt.close(fig_final)

    images += [images[-1]] * fps

    imageio.mimsave(filename, images, fps=fps)
    plt.close(fig)


def plot_regularization_weights(
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
    regularization_lambda: float,
    initial_portfolio: np.ndarray = None,
    relative: bool = False,
):
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
        initial_portfolio=initial_portfolio,
    )
    weights_initial = optimizer.x
    optimizer.optimize(regularization_lambda=regularization_lambda)
    weights_final_reg = optimizer.x
    cvar_reg = optimizer.calculate_CVaR(
        weights_final_reg, regularization_lambda=regularization_lambda
    )

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
        initial_portfolio=initial_portfolio,
    )
    optimizer.optimize(regularization_lambda=0)
    weights_final_no_reg = optimizer.x
    cvar_no_reg = optimizer.calculate_CVaR(
        weights_final_no_reg, regularization_lambda=0
    )

    if relative:
        cvar_reg /= V_0
        cvar_no_reg /= V_0
        cvar_label = "CVaR/V0"
    else:
        cvar_label = "CVaR"

    plt.figure(figsize=(14, 7))
    indices = np.arange(len(weights_final_reg)) + 1
    bar_width = 0.25

    plt.bar(
        indices - bar_width,
        weights_initial,
        width=bar_width,
        color="green",
        label="Portfel początkowy",
    )
    plt.bar(
        indices,
        weights_final_no_reg,
        width=bar_width,
        color="blue",
        label="Bez regularyzacji",
    )
    plt.bar(
        indices + bar_width,
        weights_final_reg,
        width=bar_width,
        color="orange",
        label=f"Z regularyzacją ($λ$={regularization_lambda})",
    )

    plt.xlabel("Numer aktywa")
    plt.ylabel("Waga w portfelu")
    plt.title(
        f"Porównanie wag portfela\n{cvar_label} bez reg: {cvar_no_reg:.4f}, {cvar_label} z reg: {cvar_reg:.4f}"
    )
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.xticks(indices)
    plt.show()

    print(f"{cvar_label} z regularyzacją ($λ$={regularization_lambda}): {cvar_reg:.4f}")
    print(f"{cvar_label} bez regularyzacji: {cvar_no_reg:.4f}")


def analyze_different_portfolios(
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
    initial_portfolios: list,
    relative: bool = False,
):
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
            initial_portfolio=initial_portfolio,
        )
        optimizer.optimize()
        optimal_portfolio = optimizer.x
        optimal_CVaR = optimizer.calculate_CVaR(optimal_portfolio)

        results[f"Portfel {i+1}"] = {
            "optimal_portfolio": optimal_portfolio,
            "optimal_CVaR": optimal_CVaR,
        }

    header = (
        f"{'Nr':<4} {'Portfel początkowy':<40} {'Portfel optymalny':<40} {'CVaR':<15}"
    )
    if relative:
        header = f"{'Nr':<4} {'Portfel początkowy':<40} {'Portfel optymalny':<40} {'CVaR / V0':<15}"

    print(header)
    print("-" * len(header))

    for idx, (key, value) in enumerate(results.items(), 1):
        cvar_display = (
            value["optimal_CVaR"] / V_0 if relative else value["optimal_CVaR"]
        )
        print(
            f"{idx:<4} {str(initial_portfolios[idx-1]):<40} "
            f"{str(value['optimal_portfolio']):<40} {cvar_display:<15.6f}"
        )

def plot_regularization_weights(
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
    regularization_lambda: float,
    initial_portfolio: np.ndarray = None,
    relative: bool = False,
):
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
        initial_portfolio=initial_portfolio,
    )
    weights_initial = optimizer.x
    optimizer.optimize(regularization_lambda=regularization_lambda)
    weights_final_reg = optimizer.x
    cvar_reg = optimizer.calculate_CVaR(weights_final_reg, regularization_lambda=regularization_lambda)
    cvar_reg_out = cvar_reg / V_0 if relative and V_0 > 0 else cvar_reg

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
        initial_portfolio=initial_portfolio,
    )
    optimizer.optimize(regularization_lambda=0)
    weights_final_no_reg = optimizer.x
    cvar_no_reg = optimizer.calculate_CVaR(weights_final_no_reg, regularization_lambda=0)
    cvar_no_reg_out = cvar_no_reg / V_0 if relative and V_0 > 0 else cvar_no_reg

    plt.figure(figsize=(12, 8))
    indices = np.arange(len(weights_final_reg))
    plt.bar(indices - 0.3, weights_initial, width=0.2, color="green", label="Portfel początkowy")
    plt.bar(indices - 0.1, weights_final_no_reg, width=0.2, color="blue", label="Bez regularizacji")
    plt.bar(indices + 0.1, weights_final_reg, width=0.2, color="orange", label="Z regularizacją")
    plt.xlabel("Aktywo")
    plt.ylabel("Udział w portfelu")

    title = (
        f"Wagi portfela\n"
        f"CVaR bez reg.: {cvar_no_reg_out:.4f}, "
        f"CVaR z reg. ($\lambda$={regularization_lambda}): {cvar_reg_out:.4f}"
    )
    if relative:
        title += " (wartości relatywne)"
    else:
        title += " (wartości bezwzględne)"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    if relative:
        print(f"CVaR bez regularizacji: {cvar_no_reg_out:.4f} (relatywne)")
        print(f"CVaR z regularizacją ($\lambda$={regularization_lambda}): {cvar_reg_out:.4f} (relatywne)")
    else:
        print(f"CVaR bez regularizacji: {cvar_no_reg_out:.2f}")
        print(f"CVaR z regularizacją ($\lambda$={regularization_lambda}): {cvar_reg_out:.2f}")

def probabilities_ifluence(
    T_0: float,
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
    n_paths=1,
    relative: bool = True
):

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
                return_rate=return_rate,
            )
            optimizer.optimize()
            cvar_list.append(optimizer.objective_value_history[-1])
        if relative and V_0 > 0:
            cvar_list = [cvar / V_0 for cvar in cvar_list]
        results.append(
            {
                "probabilities": f"non-uniform" if k == 0 else "uniform",
                "mean_CVaR": np.mean(cvar_list),
                "std_CVaR": np.std(cvar_list),
            }
        )
    df = pd.DataFrame(results)
    print(
        tabulate(df, headers="keys",  floatfmt=(".2f", ".2f", ".2f"))
    )


def generate_scenarios_uncor(S_0, number_of_scenarios=1000):
    """
    Generuje losowe scenariusze na podstawie stanu początkowego.
    
    """
    
    p = np.array([0.1, 0.8, 0.1])
    scenarios = np.zeros((number_of_scenarios, len(S_0)))
    for i in range(number_of_scenarios):
        coefs = [np.random.normal(0.6, 0.1), 
                 np.random.normal(1, 0.1),
                 np.random.normal(1.4, 0.1)]
        chosen = np.random.choice(coefs, size=len(S_0), p=p)
        
        S_T = S_0 * chosen
        scenarios[i] = S_T
    
    return scenarios

def generate_scenarios_corr(S_0, number_of_scenarios=1000, correlated_indices=[0,1], rho=0.5, return_rate=0.05):
    assert len(correlated_indices) == 2, "Two indices must be provided for correlation."
    
    n = len(S_0)
    scenarios = np.zeros((number_of_scenarios, n))
    
    sigmas = np.array([0.05, 0.1, 0.05, 0.1, 0.001, 0.02, 0.1, 0.1][:n])  
    
    cov_matrix = np.diag(sigmas**2)
    i, j = correlated_indices
    cov_matrix[i,j] = cov_matrix[j,i] = rho * sigmas[i] * sigmas[j]
    
    means = np.full(n, return_rate) - 0.5 * sigmas**2
    
    for i in range(number_of_scenarios):
        coefs = np.random.multivariate_normal(means, cov_matrix)
        S_T = S_0 * np.exp(coefs)
        scenarios[i] = S_T
    return scenarios
        


def examine_num_of_scenarios_influence(
    T_0: float,
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
    n_paths: int = 5,
    probabilities: str = "uniform",
    relative: bool = False,
):
    res = {}

    fig, axs = plt.subplots(
        len(nums_of_scenarios), 1, figsize=(10, 5 * len(nums_of_scenarios))
    )
    if len(nums_of_scenarios) == 1:
        axs = [axs]

    for i, num in enumerate(nums_of_scenarios):
        S_T_subset = generate_scenarios_uncor(S_0, number_of_scenarios=num)
        res[num] = []

        for j in range(n_paths):
            if probabilities == "uniform":
                probabilities_subset = (
                    np.ones(S_T_subset.shape[0]) / S_T_subset.shape[0]
                )
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
                return_rate=return_rate,
            )
            optimizer.optimize()

            axs[i].plot(
                optimizer.objective_value_history if not relative else [v / V_0 if V_0 > 0 else np.nan for v in optimizer.objective_value_history],
                color=plt.cm.Blues(np.random.uniform(0.5, 1)),
                alpha=0.8,
            )
            axs[i].set_title(f"Liczba scenariuszy: {num}")
            axs[i].set_xlabel("Iteracja")
            axs[i].set_ylabel("CVaR" if not relative else "CVaR / V0")
            axs[i].grid(True)

            final_CVaR = optimizer.objective_value_history[-1]
            relative_CVaR = final_CVaR / V_0 if relative and V_0 > 0 else final_CVaR
            res[num].append(relative_CVaR)

        print(f"Zakończono obliczenia dla {num} scenariuszy.")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    box = plt.boxplot(
        list(res.values()),
        labels=[str(k) for k in res.keys()],
        patch_artist=True,
    )
    plt.yscale("log")
    colors = plt.cm.viridis(np.linspace(0, 1, len(box["boxes"])))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    plt.xlabel("Liczba scenariuszy")
    plt.ylabel("Relatywna wartość CVaR (CVaR / V0)" if relative else "CVaR")
    plt.title("Wpływ liczby scenariuszy na relatywne CVaR" if relative else "Wpływ liczby scenariuszy na CVaR")
    plt.grid(True)
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

    axs[0].plot(nums, stds, marker="o")
    axs[0].set_xlabel("Liczba scenariuszy")
    axs[0].set_ylabel("Odchylenie std (CVaR rel.)" if relative else "Odchylenie std (CVaR)")
    axs[0].set_title("Odchylenie std vs liczba scenariuszy")
    axs[0].set_xscale("log")
    axs[0].grid(True)

    axs[1].plot(nums, max_mins, marker="o", color="orange")
    axs[1].set_xlabel("Liczba scenariuszy")
    axs[1].set_ylabel("Max - Min (CVaR rel.)" if relative else "Max - Min (CVaR)")
    axs[1].set_title("Max - Min vs liczba scenariuszy")
    axs[1].set_xscale("log")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(
        tabulate(
            table,
            headers=[
                "Liczba scenariuszy",
                "Średnia (CVaR/V0)" if relative else "Średnia (CVaR)",
                "Odchylenie std",
                "Max - Min",
            ],
            floatfmt=".6f",
        )
    )

    return res
