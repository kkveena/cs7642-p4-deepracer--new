def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    steering = abs(params['steering_angle'])
    
    if not all_wheels_on_track:
        return 1e-3

    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    else:
        reward = 1e-3 

    # Penalize sharp steering
    if steering > 20.0:
        reward *= 0.8

    # Speed bonus
    if reward > 0.5:
        reward += (speed * 0.5)
        
    return float(reward)
