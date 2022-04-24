import pymc as pm
from pymc import Model

class TeamSpecificAdvantage(Model):
    def __init__(self, name='', model=None, data = None):
        super().__init__(name, model)

        #data setup
        teams = pm.ConstantData("team_identifiers", data['pair_vals'])
        scores = pm.ConstantData("scores", data['score_diffs'])
        home_position = pm.ConstantData("home_pos", data['is_game_home'])
        strength_priors = pm.ConstantData("priors", data['pair_priors'])
        is_cup = pm.ConstantData("cup", data['is_cup'])
        home_teams = pm.ConstantData("home_team", data['home_teams'])

        #hyper params for variance
        pair_strength_var = pm.Gamma("pair_var", alpha=0.001, beta=0.001)
        home_var = pm.Gamma("home_var", alpha=0.001, beta=0.001)        
        home_var_team = pm.Gamma("home_var_team", alpha=0.001, beta=0.001)        
        cup_var = pm.Gamma("cup_var", alpha=0.001, beta=0.001)
        error_var = pm.Gamma("error_var", alpha=0.001, beta=0.001)
        
        #parameters of interest
        teampairstrength = pm.Normal("pair_strength", mu=strength_priors, tau = pair_strength_var, shape=(data['no_pairs'], 1))
        home_advantage = pm.Normal("regular_advantage", mu=4, tau=home_var, shape=(1,1))        
        home_advantage_team = pm.Normal("regular_advantage_team", mu=home_advantage, tau=home_var_team, shape=(data['no_home_teams'],1))
        cup_impact = pm.Normal("cup_impact", mu=0, tau=cup_var, shape=(1,1))

        #calculation of score
        strength_score = pm.math.dot(teams, teampairstrength)
        regular_home_advantage = pm.math.dot(home_teams, home_advantage_team)
        home_score = regular_home_advantage * (home_position + is_cup * cup_impact)
        full_score = strength_score + home_score

        #likelihood        
        pm.Normal("observed", mu=full_score, tau=error_var, observed=scores)