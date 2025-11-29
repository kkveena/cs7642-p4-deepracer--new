def reward_function(params):
    '''
    Object Avoidance Reward
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

    # 3. Obstacle Avoidance Logic
    # Check distance to closest object
    closest_obj_index = closest_objects[0]
    dist_to_obj = objects_distance[closest_obj_index]

    # If close (< 1.5m), reduce reward
    if dist_to_obj < 1.5:
        reward *= 0.5
        # If DANGEROUSLY close (< 0.8m), almost 0 reward
        if dist_to_obj < 0.8:
            reward = 1e-3

    # 4. Speed (Only if safe)
    if dist_to_obj >= 1.5:
        reward += (speed * 0.5)

    return float(reward)
