def reward_function(params):
    '''
    Time Trial Reward Function
    Encourages staying near the centerline and maintaining high speed.
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    SPEED_THRESHOLD = 1.0 

    # 1. Penalize if off track (Critical for termination)
    if not all_wheels_on_track:
        return 1e-3

    # 2. Calculate distance reward (The closer to center, the better)
    # We use markers for 10%, 25%, and 50% of the track width
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # Likely crashed or very close to off-track

    # 3. Speed Incentive (Time Trial specific)
    # If the car is stable (reward > 0.1) and fast, give a bonus
    if reward > 0.1:
        if speed > SPEED_THRESHOLD:
            reward *= 1.5  # 50% bonus for high speed
    
    return float(reward)