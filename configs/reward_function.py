def reward_function(params):
    if params['all_wheels_on_track']:
        return 1.0
    return 1e-3
