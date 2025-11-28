def reward_function(params):
    '''
    Part 1: Time Trial Reward
    Incentivizes speed and staying close to the center.
    '''
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    steering = abs(params['steering_angle'])
    
    # 1. FAIL CONDITION: Off Track
    if not all_wheels_on_track:
        return 1e-3

    # 2. POSITION REWARD: Stay within the safety borders
    # We penalize heavily if the car is on the edge (danger zone)
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
        reward = 1e-3  # Too close to edge

    # 3. SPEED PENALTY (Prevent zig-zagging)
    # If steering is high (sharp turn), we shouldn't be at max speed
    # If steering is low (straight), we should be fast.
    
    SPEED_THRESHOLD = 2.0
    
    if reward > 0.1: # Only bonus if valid position
        if speed > SPEED_THRESHOLD:
            reward *= 2.0  # Big bonus for speed
        
        # Penalize steering while going fast (instability)
        if speed > 3.0 and steering > 15:
            reward *= 0.8

    return float(reward)