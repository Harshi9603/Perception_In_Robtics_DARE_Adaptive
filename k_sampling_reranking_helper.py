import os
import time
import random
import collections

from copy import deepcopy

import torch
import numpy as np
from matplotlib import pyplot as plt

from test_parameter import *
from classes.utils import *
from classes.env.env import Env
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager

class KSamplingReRanking:

    def __init__(self) -> None:
        # coefficients
        self.a = 0.0  # utility sum
        self.b = 0.0  # unique node count
        self.c = 0.0  # revisit penalty
        self.d = 0.0  # invalid edge count
        self.e = 0.0  # terminal node utility
        self.f = 0.0  # terminal node connectivity
        self.g = 0.0  # guidepost overlap
        self.k = 1

        self.candidate_action_preds = None
        self.robot_list = None
        self.current_robot_location = None

    def set_parameters(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        g: float,
        k: int
    ) -> None:
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.k = k

    def get_k(self) -> int:
        return self.k

    def set_current_actions_to_be_ranked(self, candidate_action_preds: list) -> None:
        self.candidate_action_preds = candidate_action_preds

    def set_robot_list(self, robot_list: list) -> None:
        self.robot_list = robot_list

    def set_current_robot_location(self, current_robot_location: np.ndarray) -> None:
        self.current_robot_location = np.array(current_robot_location, dtype=float)

    def min_max_normalize(
        self,
        value: float,
        min_value: float,
        max_value: float,
        higher_is_better: bool = True
    ) -> float:
        """
        Min-max normalization to keep all base metrics in the same range.
        """
        if max_value == min_value:
            return 0.0

        if higher_is_better:
            normalized = (value - min_value) / (max_value - min_value)
        else:
            normalized = (max_value - value) / (max_value - min_value) 

        return float(np.clip(normalized, 0.0, 1.0))

    def rank_actions_and_get_best_index(self) -> int:
        """
        Combines all the normalized scores with their coefficients to find the max score index (idx).
        """
        ranking_scores = []

        for action_pred in self.candidate_action_preds:
            path_indices = self.map_action_pred_to_path_indices(action_pred)
            metrics = self.compute_all_metrics(path_indices)

            score = 0.0
            score += self.a * self.min_max_normalize(metrics["utility_sum"], 0.0, 400.0, True)
            score += self.b * self.min_max_normalize(metrics["unique_node_count"], 1.0, 8.0, True)
            score += self.c * self.min_max_normalize(metrics["revisit_penalty"], 0.0, 7.0, False)
            score += self.d * self.min_max_normalize(metrics["invalid_edge_count"], 0.0, 7.0, False)
            score += self.e * self.min_max_normalize(metrics["terminal_node_utility"], 0.0, 50.0, True)
            score += self.f * self.min_max_normalize(metrics["terminal_node_connectivity"], 1.0, 25.0, True)
            score += self.g * self.min_max_normalize(metrics["guidepost_overlap"], 0.0, 8.0, True)

            ranking_scores.append(score)

        best_idx = int(np.argmax(ranking_scores))
        return best_idx

    def map_action_pred_to_path_indices(self, action_pred: np.ndarray) -> list:
        """
        Convert one predicted action sequence into an implied node-index path
        on the current informative graph.
        """
        agent = self.robot_list[0]
        node_manager = agent.node_manager

        node_coords = agent.node_coords
        coord_to_idx = {
            (float(coord[0]), float(coord[1])): i
            for i, coord in enumerate(node_coords)
        }

        current_coord = self.current_robot_location.copy()
        path_indices = []

        for step_action in action_pred:
            proposed_coord = current_coord + step_action

            current_node = node_manager.nodes_dict.find(current_coord.tolist()).data

            chosen_coord = None

            # valid neighbor exact match
            for nbr in current_node.neighbor_list[1:]:
                nbr_arr = np.array(nbr, dtype=float)
                if np.allclose(proposed_coord, nbr_arr):
                    chosen_coord = nbr_arr
                    break

            # otherwise snap to nearest valid neighbor
            if chosen_coord is None:
                best_neighbor = None
                best_dist = float("inf")

                for nbr in current_node.neighbor_list[1:]:
                    nbr_arr = np.array(nbr, dtype=float)
                    dist = np.linalg.norm(proposed_coord - nbr_arr)
                    if dist < best_dist:
                        best_dist = dist
                        best_neighbor = nbr_arr

                # fallback: stay put if needed
                if best_neighbor is None:
                    chosen_coord = current_coord.copy()
                else:
                    chosen_coord = best_neighbor

            key = (float(chosen_coord[0]), float(chosen_coord[1]))
            if key in coord_to_idx:
                path_indices.append(coord_to_idx[key])

            current_coord = chosen_coord.copy()

        return path_indices

    def compute_all_metrics(self, path_indices: list) -> dict:
        """
        Compute all the metrics and return a dict of the metrics. 
        """
        agent = self.robot_list[0]

        utility_sum = self.utility_sum(path_indices, agent)
        unique_node_count = self.unique_node_count(path_indices)
        revisit_penalty = self.revisit_penalty(path_indices)
        invalid_edge_count = self.invalid_edge_count(path_indices, agent)
        terminal_node_utility = self.terminal_node_utility(path_indices, agent)
        terminal_node_connectivity = self.terminal_node_connectivity(path_indices, agent)
        guidepost_overlap = self.guidepost_overlap(path_indices, agent)

        return {
            "utility_sum": utility_sum,
            "unique_node_count": unique_node_count,
            "revisit_penalty": revisit_penalty,
            "invalid_edge_count": invalid_edge_count,
            "terminal_node_utility": terminal_node_utility,
            "terminal_node_connectivity": terminal_node_connectivity,
            "guidepost_overlap": guidepost_overlap,
        }

    def utility_sum(self, path_indices: list, agent) -> float:
        """
        Sums the utilities of all nodes in the path, including duplicates. This captures the total reward potential of the path.
        """
        return float(sum(agent.utility[idx] for idx in path_indices))

    def unique_node_count(self, path_indices: list) -> int:
        """
        Counts the number of unique nodes in the path. This encourages paths that cover more distinct areas.
        """
        return len(set(path_indices))

    def revisit_penalty(self, path_indices: list) -> int:
        """
        Calculates the penalty for revisiting nodes in the path.
        """
        return len(path_indices) - len(set(path_indices))

    def invalid_edge_count(self, path_indices: list, agent) -> int:
        """
        Counts the number of invalid edges in the path.
        """

        count = 0
        for i in range(len(path_indices) - 1):
            a = path_indices[i]
            b = path_indices[i + 1]
            if agent.adjacent_matrix[a, b] != 0:
                count += 1
        return count

    def terminal_node_utility(self, path_indices: list, agent) -> float:
        """
        Gets the utility of the terminal node in the path. This captures the reward potential of where the path ends.
        """
        if len(path_indices) == 0:
            return 0.0
        terminal_idx = path_indices[-1]
        return float(agent.utility[terminal_idx])

    def terminal_node_connectivity(self, path_indices: list, agent) -> int:
        """
        Gets the connectivity of the terminal node in the path. This captures how many neighbors the terminal node has.
        """
        if len(path_indices) == 0:
            return 0

        terminal_idx = path_indices[-1]
        terminal_coord = agent.node_coords[terminal_idx]

        terminal_node = agent.node_manager.nodes_dict.find(
            terminal_coord.tolist()
        ).data

        return len(terminal_node.neighbor_list) - 1

    def guidepost_overlap(self, path_indices: list, agent) -> float:
        """
        Calculates the overlap with guideposts along the path.
        """
        return float(sum(agent.guidepost[idx] for idx in path_indices))
    

# Global instance to be used in the main loop
k_sampling_reranking_instance = KSamplingReRanking()
