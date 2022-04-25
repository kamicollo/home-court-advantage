# Estimating the home-court advantage in Lithuanian Basketball League using Bayesian methods. 

In this project, I applied Bayesian methods to estimate home-court advantage in the Lithuanian Basketball League. In line with prior research, I find that home-court advantage exists, and its overall magnitude is somewhat smaller than what has been identified in the research focused on US basketball (c. 2.3 - 2.6 points vs. above 4 points identified in the US). Intuitively, this makes sense given smaller arena sizes in Lithuania and thus, arguably, smaller spectator impact, as well smaller distances that the opponent teams need to travel compared to the US.

I also explored whether team-specific models perform better, with inconclusive results. It appears that the models that allow for team-specific home-court advantages perform similarly to the ones that treat home-court advantage as a uniform phenomenon. At the same time, such models do not necessarily perform worse, and provide further insights into team-level dynamics, where it appears that only a fraction of the teams benefit from a consistent home-court advantage.


# Directory structure
```
├── LICENSE
├── README.md
├── data
│   └── data.sqlite.db -> game data scraped from LKL website
├── data-collection
│   └── lkl-scraper.ipynb -> data scraper
├── docs -> course project report
├── modelling
│   ├── EDA.ipynb -> Basic EDA facts about home-court advantage
│   ├── data_utils.py -> Shaping the data into the right form for model fitting
│   ├── model_fitting.ipynb -> Running MCMC on all models
│   └── model_evaluation.ipynb -> Evaluation of models, inspection of key coefficients
├── models
│   ├── OneAdvantage.py -> Home-court advantage as a uniform phenomenon
│   ├── TeamSpecificAdvantage.py -> Home-court advantage as a team-specific phenomenon
├── requirements.txt -> python packages (this project uses pymc4 bv5)
├── storage -> model data storage
└── vis -> plots for the report
    ├── 2020-trace.png
    ├── 5-seasons-trace.png
    ├── home_court_eda.png
    └── model_cmp.png
```