import itertools
import numpy as np
import random
from pathlib import Path
from scipy.spatial import KDTree

def make_list(input1, input2):
    return list(itertools.product(input1, input2))

def find_max_positions(lst):
    max_val = max(lst)
    max_positions = []
    for i, val in enumerate(lst):
        if val == max_val:
            max_positions.append(i)
    return max_positions[-1]

def find_min_positions(lst):
    min_val = min(lst)
    min_positions = []
    for i, val in enumerate(lst):
        if val == min_val:
            min_positions.append(i)
    return min_positions[0]

# def create_list(point_list):
#     x = random.sample(point_list, 3)
#     l = len(point_list)
#     pos_choice = []
#     for i in range(l):
#         pos_choice.append(point_list[i][0] + point_list[i][1])
#     x_max_pos = find_max_positions(pos_choice)
#     x_min_pos = find_min_positions(pos_choice)
#     x.append(point_list[x_max_pos])
#     x.append(point_list[x_min_pos])
#     return x
# 
# def create_list(point_list):
#     x = random.sample(point_list, 5)
#     print(x)
#     return x

def find_nearlist_point(supernet, current_point, distance):
    tree = KDTree(supernet)
    _, indices = tree.query(current_point, distance)
    choose_point = []
    for item in indices:
        choose_point.append(supernet[item])
    return choose_point

# data = [[1, 2, 10], [3, 4, 10], [5, 6, 10], [6, 8, 10], [9, 10, 10]]
# print(find_nearlist_point(data, [4,5], 5))

class evolution_supernet(object):
    def __init__(self, result_array, rank_ratio, patch_size):
        self.result_array = result_array
        self.rank_ratio = rank_ratio
        self.patch_size = patch_size

    def fit_SRAM_plane(self):
        X = np.array([[1, x1, x2] for x1, x2, y1, y2 in self.result_array])
        y = np.array([[1-y2] for x1, x2, y1, y2 in self.result_array]).reshape(-1, 1)
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        return w.reshape(-1)[-2:]

    def get_SRAM_evolution_step(self):
        weight_array = self.fit_SRAM_plane()
        threshold_array = np.array([0.2, 2])
        print("sram",weight_array)
        step_array = np.array([0.05, 4])
        return (weight_array//threshold_array)*step_array


    def fit_error_plane(self):
        X = np.array([[1, x1, x2] for x1, x2, y1, y2 in self.result_array])
        y = np.array([[100-y1] for x1, x2, y1, y2 in self.result_array]).reshape(-1, 1)

        w = np.linalg.inv(X.T @ X) @ X.T @ y
        return w.reshape(-1)[-2:]

    def get_error_evolution_step(self):
        weight_array = self.fit_error_plane()
        threshold_array = np.array([10, 20])
        step_array = np.array([0.05, 4])
        print("error",weight_array)
        return (weight_array//threshold_array)*step_array
    
    def evolution_step(self):
        error_step = self.get_error_evolution_step()
        sram_step = self.get_SRAM_evolution_step()
        step = error_step + sram_step
        if self.patch_size + step[1] < 16:
            step[1] = 16 - self.patch_size
        if self.patch_size + step[1] > 32:
            step[1] = 32 - self.patch_size
        if self.rank_ratio + step[0] < 0.4:
            step[0] = 0.4 - self.rank_ratio
        if self.rank_ratio + step[0] > 0.95:
            step[0] = 0.95 - self.rank_ratio
        return step


# input_vectors = [[0.9, 16, 66.298, 0.384], [0.6, 16, 60.6, 0.666], [0.65, 28, 45.119, 1], [0.95, 28, 57.52, 1], [0.85, 20, 61.2, 1]]
# # input_vectors = [[0.1, 1.5, 1.2, 2.6], [0.2, 0.4, 1.4,2.7], [0.3, 0.3, 1.6,2.8], [0.4, 0.2, 1.8,2.9], [0.5, 0.1, 2.0,3.0]]
# # evolution_step = evolution_supernet(input_vectors, 20)
# print(evolution_supernet(input_vectors, 0.9, 16).evolution_step())