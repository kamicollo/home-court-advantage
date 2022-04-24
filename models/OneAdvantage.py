import pymc as pm
from pymc import Model

class OneAdvantage(Model):
    def __init__(self, name='', model=None, data = None):
        super().__init__(name, model)

        #data setup
        teams = pm.ConstantData("team_identifiers", data['pair_vals'])
        scores = pm.ConstantData("scores", data['score_diffs'])
        home_position = pm.ConstantData("home_pos", data['is_game_home'])
        strength_priors = pm.ConstantData("priors", data['pair_priors'])
        is_cup = pm.ConstantData("cup", data['is_cup'])           

        #hyper params for variance
        pair_strength_var = pm.HalfCauchy("pair_var", beta=5)
        home_var = pm.HalfCauchy("home_var", beta=5)
        cup_var = pm.HalfCauchy("cup_var", beta=5)
        error_var = pm.HalfCauchy("error_var", beta=5)
        
        #parameters of interest
        teampairstrength = pm.Normal("pair_strength", mu=strength_priors, sigma = pair_strength_var, shape=(data['no_pairs'], 1))
        home_advantage = pm.Normal("regular_advantage", mu=4, sigma=home_var, shape=(1,1))
        cup_impact = pm.Normal("cup_impact", mu=0, sigma=cup_var, shape=(1,1))

        #calculation of score
        strength_score = pm.math.dot(teams, teampairstrength)
        home_score = home_advantage * (home_position + is_cup * cup_impact)
        full_score = strength_score + home_score

        #likelihood        
        pm.Normal("observed", mu=full_score, sigma=error_var, observed=scores)