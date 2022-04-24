import pymc as pm
from pymc import Model

class DuelSpecificAdvantage(Model):
    def __init__(self, name='', model=None, data = None):
        super().__init__(name, model)

        #data setup
        pairs = pm.ConstantData("pairs", data['pair_ids'])
        home_position = pm.ConstantData("home_pos", data['is_game_home'])
        scores = pm.ConstantData("scores", data['score_diffs'])        
        is_cup = pm.ConstantData("cup", data['is_cup'])
        home_teams = pm.ConstantData("home_team", data['home_teams'])
        away_teams = pm.ConstantData("away_team", data['home_teams'])
        strength_priors = pm.ConstantData("strength_priors", data['team_priors'])

        #hyper params for variance
        team_strength_var = pm.Gamma("pair_var", alpha=0.001, beta=0.001)
        home_var = pm.Gamma("home_var", alpha=0.001, beta=0.001)        
        home_var_team = pm.Gamma("home_var_team", alpha=0.001, beta=0.001)        
        cup_var = pm.Gamma("cup_var", alpha=0.001, beta=0.001)
        error_var = pm.Gamma("error_var", alpha=0.001, beta=0.001)
        
        #parameters of interest
        team_strength = pm.Normal("team_strength", mu=strength_priors, tau = team_strength_var, shape=(data['no_teams'], 1))


        teampairstrength = pm.Normal("pair_strength", mu=strength_priors, sigma = pair_strength_var, shape=(data['no_pairs'], 1))
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