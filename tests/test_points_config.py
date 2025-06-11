from cfspopcon.shaping_and_selection.point_config_class import PointsConfig, PointConfig

def test_points_config(example_inputs):

    points_spec = example_inputs["points"]

    PointsConfig(points_spec)