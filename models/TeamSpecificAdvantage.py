import pymc as pm
from pymc import Model

class Mod(Model):
    def __init__(self, name='', model=None, data = None, advantage_prior = (4, None)):
        super().__init__(name, model)

        #data setup
        teams = pm.ConstantData("team_identifiers", data['pair_vals'])
        scores = pm.ConstantData("scores", data['score_diffs'])
        home_position = pm.ConstantData("home_pos", data['is_game_home'])
        strength_priors = pm.ConstantData("priors", data['pair_priors'])
        is_cup = pm.ConstantData("cup", data['is_cup'])
        home_teams = pm.ConstantData("home_team", data['home_teams'])
        
        #pair strenghts
        pair_strength_var = pm.HalfCauchy("pair_var", beta=5)
        teampairstrength = pm.Normal("pair_strength", mu=strength_priors, sigma = pair_strength_var, shape=(data['no_pairs'], 1))

        
        #homecount advantage
        if advantage_prior[1] is None:
            home_var = pm.HalfCauchy("home_var", beta=2)
        else:
            home_var = advantage_prior[1]
            
        home_advantage = pm.Normal("regular_advantage", mu=advantage_prior[0], sigma=home_var, shape=(1,1))
        
        #team-level advantage
        home_var_team = pm.HalfCauchy("home_var_team", beta=2)
        team_advantage = pm.Normal("team_advantage", mu=home_advantage, sigma=home_var_team, shape=(data['no_home_teams'],1))
                        
        #cup-impact
        cup_var = pm.HalfCauchy("cup_var", beta=2)
        cup_impact = pm.Normal("cup_impact", mu=0, sigma=cup_var, shape=(1,1))        

        #calculation of score
        strength_score = pm.math.dot(teams, teampairstrength)
        regular_home_advantage = pm.math.dot(home_teams, team_advantage)
        home_score = (regular_home_advantage * home_position) + (home_position * is_cup * cup_impact)
        full_score = strength_score + home_score
        
        #tracking of variables        
        pm.Deterministic("total_cup_advantage", pm.math.switch(home_advantage + cup_impact > 0, 1, 0))

        #likelihood     
        error_var = pm.HalfCauchy("error_var", beta=5)
        pm.Normal("observed", mu=full_score, sigma=error_var, observed=scores)