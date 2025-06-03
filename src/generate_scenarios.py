import numpy as np 
from typing import List

def generate_scenarios_single_stock(path_to_stock_data: str, n_scenarios: int, period_in_days: int, initial_price: float | int) -> np.ndarray:
    closing_prices = np.loadtxt(path_to_stock_data, skiprows=1, usecols=(4), delimiter=",")

    if n_scenarios + period_in_days > len(closing_prices):
        raise ValueError(f"Provided data only allows for {len(closing_prices) - period_in_days} scenarios. Got {n_scenarios}.")
    
    return initial_price * closing_prices[-n_scenarios:] / closing_prices[-(n_scenarios + period_in_days):-period_in_days]

def generate_scenarios_historical_data(paths_to_stock_data: List[str], n_scenarios: int, period_in_days: int, initial_price: float | int) -> np.ndarray:
    n_stocks = len(paths_to_stock_data)
    scenarios = np.empty((n_scenarios, n_stocks))

    for i, path_to_stock_data in enumerate(paths_to_stock_data):
        stock_prices = generate_scenarios_single_stock(path_to_stock_data, n_scenarios, period_in_days, initial_price)
        scenarios[:, i] = stock_prices

    return scenarios