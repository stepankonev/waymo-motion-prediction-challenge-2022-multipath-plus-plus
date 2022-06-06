import tensorflow as tf

_roadgraph_features = {
    "roadgraph_samples/dir": tf.io.FixedLenFeature(
        [20000, 3], tf.float32, default_value=None
    ),
    "roadgraph_samples/id": tf.io.FixedLenFeature(
        [20000, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/type": tf.io.FixedLenFeature(
        [20000, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/valid": tf.io.FixedLenFeature(
        [20000, 1], tf.int64, default_value=None
    ),
    "roadgraph_samples/xyz": tf.io.FixedLenFeature(
        [20000, 3], tf.float32, default_value=None
    ),
}

_general_state_features = {
    "state/id": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    "state/type": tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    "state/is_sdc": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    "state/tracks_to_predict": tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    "scenario/id": tf.io.FixedLenFeature([1], tf.string, default_value=None)}


_values_number_for_timezone = {
    "current": 1,
    "future": 80,
    "past": 10
}

def _generate_agent_features_by_timezone(timezone):
    assert timezone in ["current", "future", "past"]
    n_values = _values_number_for_timezone[timezone]
    return {
        f"state/{timezone}/x": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),
        f"state/{timezone}/y": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),
        f"state/{timezone}/z": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),

        f"state/{timezone}/velocity_x": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),
        f"state/{timezone}/velocity_y": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),
        f"state/{timezone}/speed": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),

        f"state/{timezone}/length": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),
        f"state/{timezone}/width": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),
        f"state/{timezone}/height": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),

        f"state/{timezone}/bbox_yaw": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None),
        f"state/{timezone}/timestamp_micros": tf.io.FixedLenFeature(
            [128, n_values], tf.int64, default_value=None),
        f"state/{timezone}/valid": tf.io.FixedLenFeature(
            [128, n_values], tf.int64, default_value=None),
        f"state/{timezone}/vel_yaw": tf.io.FixedLenFeature(
            [128, n_values], tf.float32, default_value=None)}

_traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

def generate_features_description():
    features_description = {}
    features_description.update(_roadgraph_features)
    features_description.update(_general_state_features)
    features_description.update(_traffic_light_features)
    for timezone in ["past", "current", "future"]:
        features_description.update(_generate_agent_features_by_timezone(timezone))
    return features_description