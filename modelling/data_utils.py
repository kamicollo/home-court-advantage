import sqlite3 as sq
import pandas as pd
import itertools as itt
import numpy as np

def get_data(path_to_sql):
    with sq.connect(path_to_sql) as conn:
        sql = """
        SELECT `season_id`, `home-points`, `away-points`, `home-team-id`, `away-team-id`, `start-year`, `regular` FROM games
        JOIN seasons ON games.season_id = seasons.id
        ORDER BY `game-id`
        ;
        """
        all_games = pd.read_sql_query(sql, conn)

        #convert types
        all_games['home-team-id']= all_games['home-team-id'].astype("int")
        all_games['away-team-id']= all_games['away-team-id'].astype("int")
        all_games['start-year']= all_games['start-year'].astype("int")

        #apply identifiers to the dataframe
        def helper_(x):
            return (max(x['home-team-id'], x['away-team-id']), min(x['home-team-id'], x['away-team-id']))
        all_games['pair_id'] = all_games.apply(helper_, axis=1)

        #define is_home variable and set score with a correct sign
        all_games['is_home'] = (all_games['home-team-id'] > all_games['away-team-id']) * 1
        all_games['is_home'] = all_games['is_home'].apply(lambda x: x if x == 1 else -1)
        all_games['score_diff'] = (all_games['home-points'] - all_games['away-points']) * all_games['is_home']
    return all_games

def compute_data(games, start_year = 2020, seasons = 2, informative_priors = None, by_season = False):
    
    df = games.copy()
    
    #define grouping / identifier variables
    if by_season:
        df['group'] = pd.Series(zip(df['pair_id'].values, df['start-year']))
        df['home-team'] = pd.Series(zip(df['home-team-id'].values, df['start-year']))
    else:
        df['group'] = df['pair_id']
        df['home-team'] = df['home-team-id']
    
    #get relevant time periods
    end_year = start_year + seasons
    mask = (df['start-year'] >= start_year) & (df['start-year'] < end_year)
    last_seasons = df[mask]
    
    #get team-pair identifiers
    pairs = pd.get_dummies(last_seasons['group'])
    pair_vals = pairs.values
    
    #get home team identifiers
    home_teams = pd.get_dummies(last_seasons['home-team'])
    home_team_dummy = home_teams.values    
    
    #variables for convenience
    is_home = last_seasons['is_home'].values.reshape(-1 , 1)
    score_diffs = last_seasons['score_diff'].values.reshape(-1, 1)
    is_cup = 1 - last_seasons['Regular'].values.reshape(-1, 1)
    no_obs, no_pairs = pair_vals.shape
        
    
    #non-informative strenght priors
    if not informative_priors:
        team_pair_advantages = np.zeros((no_pairs, 1))
        
    else:
        assert by_season == False, "Informative priors when grouped by season don't make sense!"
        
        #get the period for the priors
        begin_cut_off = start_year - informative_priors
        mask = (df['start-year'] < start_year) & (df['start-year'] >= begin_cut_off)
        prior_seasons = df[mask]
        
        #calculate average score difference
        average_diffs = prior_seasons[['group', 'score_diff']].groupby('group').mean().reset_index()

        sorted_diffs = []
        for p in pairs.columns.values:
            v = average_diffs['score_diff'][average_diffs['group'] == p].values        
            if len(v) == 0:
                v = [0]
            sorted_diffs.append(v)

        team_pair_advantages = np.fromiter(itt.chain(*sorted_diffs), dtype=float).reshape(-1, 1)
        
                
    return {
        'pair_vals' : pair_vals,
        'pair_ids': pairs.columns.values,
        'is_game_home': is_home,
        'score_diffs': score_diffs,
        'is_cup': is_cup,
        'no_obs': no_obs,
        'no_pairs': no_pairs,
        'home_teams': home_team_dummy,
        'home_team_ids': home_teams.columns.values,
        'no_home_teams': home_team_dummy.shape[1],        
        'pair_priors' : team_pair_advantages,        
    }
