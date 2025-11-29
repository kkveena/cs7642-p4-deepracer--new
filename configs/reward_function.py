def reward_function(params):
    '''
    Head-to-Bot Reward
    - Penalize collisions
    - Maintain speed if clear
    - Avoid objects (bots)
    '''
    
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    objects_distance = params['objects_distance']
    closest_objects = params['closest_objects']
    is_crashed = params['is_crashed']
    speed = params['speed']

    # 1. Fail Conditions
    if is_crashed or not all_wheels_on_track:
        return 1e-3

    # 2. Base Reward (Stay on track)
    reward = 1.0
    if distance_from_center > 0.4 * track_width:
        reward = 0.5

    # 3. Bot Avoidance Logic
    # In H2B, bots appear in the object list
    closest_obj_index = closest_objects[0]
    dist_to_obj = objects_distance[closest_obj_index]

    # Bots move, so we need a larger safety margin than static boxes
    if dist_to_obj < 2.0:
        reward *= 0.5
        if dist_to_obj < 1.0:
            reward = 1e-3  # Danger zone

    # 4. Speed Incentive
    if dist_to_obj >= 2.0:
        reward += (speed * 1.0)

    return float(reward)
