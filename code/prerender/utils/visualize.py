import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from utils.features_description import generate_features_description

def parse_one_scene(filename):
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    data = next(dataset.as_numpy_iterator())
    parsed = tf.io.parse_single_example(data, generate_features_description())
    return parsed

def plot_arrowbox(center, yaw, length, width, color, alpha=1):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array(((c, -s), (s, c))).reshape(2, 2)
    box = np.array([
        [-length / 2, -width / 2],
        [-length / 2,  width / 2],
        [ length / 2,  width / 2],
        [ length * 1.3 / 2,  0],
        [ length / 2, -width / 2],
        [-length / 2, -width / 2]])
    box = box @ R.T + center
    plt.plot(box[:, 0], box[:, 1], color=color, alpha=alpha)
    
def plot_roadlines(segments):
    plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="black", s=0.1)

def plot_scene(scene_data):
    for timezone, color in [('history', 'blue'), ('future', 'yellow')]:
        for i in range(len(scene_data[f"other/{timezone}/xy"])):
            for other_position, other_yaw, other_valid in zip(
                    scene_data[
                        f"other/{timezone}/xy"][i],
                        scene_data[f"other/{timezone}/yaw"][i],
                        scene_data[f"other/{timezone}/valid"][i]):
                if other_valid.item() == 0:
                    continue
                plot_arrowbox(
                    other_position, other_yaw, scene_data["other/length"][i],
                    scene_data["target/width"], color, alpha=0.5)

    for timezone, color in [('history', 'red'), ('future', 'green')]:
        for target_position, target_yaw, target_valid in zip(
                scene_data[f"target/{timezone}/xy"][0],
                scene_data[f"target/{timezone}/yaw"][0, :, 0],
                scene_data[f"target/{timezone}/valid"][0, :, 0]):
            if target_valid == 0:
                continue
            plot_arrowbox(target_position, target_yaw, scene_data["target/length"],
            scene_data["target/width"], color)
            
    plot_roadlines(scene_data["road_network_segments"])