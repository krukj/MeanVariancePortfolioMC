# MeanVariancePortfolioMC

### Struktura projektu
```
MeanVariancePortfolioMC/
  ├── src/ 
  │   ├── PorfolioOptimizerSA.py      # klasa optymalizująca CVaR przy użyciu symulowanego wyżarzania
  │   ├── analysis.py                 # funkcje do wizualizacji
  │   └── generate_scenarios.py       # funckje do generowania scenariuszy z danych historycznych
  ├── stock_data/                     # pliki .csv z cenami akcji        
  ├── visualizations/                 # wykresy
  ├── README.md                       # ten plik
  ├── example.ipynb                   # przykładowe użycie klasy PorfolioOptimizerSA
  ├── historical_data_scenarios.ipynb # wywołanie metod klasy na danych historycznych
  ├── paths_generation.ipynb          # eksperymenty
  └── raport.md                       # raport
```
