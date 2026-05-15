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



if not os.path.exists(gifs_path):

    os.makedirs(gifs_path)



from k_sampling_reranking_helper import k_sampling_reranking_instance



# ── NEW: Adaptive Step Scheduler ──────────────────────────────────────────────

from diffusion_exploration.utils.adaptive_scheduler import (

    AdaptiveStepScheduler,

    AdaptiveSchedulerConfig,

    reset_scheduler,

    get_scheduler_summary,

)



# One scheduler instance per worker; configure bounds to match your DDPM config

_scheduler_cfg = AdaptiveSchedulerConfig(

    t_min=20,

    t_max=100,

    w_density=0.6,

    w_distance=0.4,

    ema_alpha=0.3,

)

adaptive_scheduler = AdaptiveStepScheduler(cfg=_scheduler_cfg)

# ── END NEW ───────────────────────────────────────────────────────────────────





def set_random_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False





class TestWorker:

    def __init__(self, meta_agent_id, policy, global_step, device='cpu', save_image=False):

        self.meta_agent_id = meta_agent_id

        self.policy = policy

        self.global_step = global_step

        self.device = device

        self.save_image = save_image



        self.env = Env(global_step, TEST_N_AGENTS, plot=save_image, test=USE_TEST_DATASET)

        self.node_manager = NodeManager(plot=save_image)

        self.robot_list = [Agent(i, self.node_manager, self.device, save_image) for i in range(TEST_N_AGENTS)]



        self.perf_metrics = dict()

        self.obs_horizon = policy.n_obs_steps

        self.action_horizon = policy.n_action_steps if ACTION_HORIZON is None else ACTION_HORIZON



        self.planned_path_x = []

        self.planned_path_y = []



        # ── NEW: reset scheduler state at the start of each worker ────────────

        adaptive_scheduler.reset()

        # ── END NEW ───────────────────────────────────────────────────────────



    def run_episode(self):

        unique_seed = int(time.time())

        set_random_seed(unique_seed)



        done = False



        for robot in self.robot_list:

            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))

        for robot in self.robot_list:

            robot.update_planning_state(self.env.robot_locations)



        # Get the first observation

        if DATA_TYPE == 'node':

            observation = self.robot_list[0].get_observation()

            node_inputs         = observation[0].squeeze(0)

            node_padding_mask   = observation[1].squeeze(0)

            edge_mask           = observation[2].squeeze(0)

            current_index       = observation[3].squeeze(0)

            current_edge        = observation[4].squeeze(0)

            edge_padding_mask   = observation[5].squeeze(0)

            obs = {

                'node_inputs':       node_inputs,

                'node_padding_mask': node_padding_mask,

                'edge_mask':         edge_mask,

                'current_index':     current_index,

                'current_edge':      current_edge,

                'edge_padding_mask': edge_padding_mask,

            }

        elif DATA_TYPE == 'map':

            image     = deepcopy(self.env.robot_belief)

            state     = deepcopy(self.env.robot_locations[0])

            agent_pos = state.astype(np.float32)

            image     = image.astype(np.float32) / 255

            image     = np.expand_dims(image, axis=0)

            obs = {'image': image, 'agent_pos': state}

        else:

            raise ValueError('Invalid data type, check test_parameter.py')



        obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)



        step = 0

        for step in range(MAX_EPISODE_STEP):



            # ── Stack observations ────────────────────────────────────────────

            if DATA_TYPE == 'node':

                node_inputs       = torch.stack([x['node_inputs']       for x in obs_deque])

                node_padding_mask = torch.stack([x['node_padding_mask'] for x in obs_deque])

                edge_mask         = torch.stack([x['edge_mask']         for x in obs_deque])

                current_index     = torch.stack([x['current_index']     for x in obs_deque])

                current_edge      = torch.stack([x['current_edge']      for x in obs_deque])

                edge_padding_mask = torch.stack([x['edge_padding_mask'] for x in obs_deque])



                node_inputs       = node_inputs.to(self.device,       dtype=torch.float32)

                node_padding_mask = node_padding_mask.to(self.device, dtype=torch.int16)

                edge_mask         = edge_mask.to(self.device,         dtype=torch.int64)

                current_index     = current_index.to(self.device,     dtype=torch.int64)

                current_edge      = current_edge.to(self.device,      dtype=torch.int64)

                edge_padding_mask = edge_padding_mask.to(self.device, dtype=torch.int16)



                obs_dict = {

                    'node_inputs':       node_inputs.unsqueeze(0),

                    'node_padding_mask': node_padding_mask.unsqueeze(0),

                    'edge_mask':         edge_mask.unsqueeze(0),

                    'current_index':     current_index.unsqueeze(0),

                    'current_edge':      current_edge.unsqueeze(0),

                    'edge_padding_mask': edge_padding_mask.unsqueeze(0),

                }

            elif DATA_TYPE == 'map':

                image     = torch.stack([torch.tensor(x['image'])     for x in obs_deque])

                agent_pos = torch.stack([torch.tensor(x['agent_pos']) for x in obs_deque])

                image     = image.to(self.device,     dtype=torch.float32)

                agent_pos = agent_pos.to(self.device, dtype=torch.float32)

                obs_dict  = {

                    'image':     image.unsqueeze(0),

                    'agent_pos': agent_pos.unsqueeze(0),

                }

            else:

                raise ValueError('Invalid data type, check test_parameter.py')



            # ── NEW: compute adaptive step count ─────────────────────────────

            frontiers = (

                np.array(list(self.env.global_frontiers))

                if hasattr(self.env, 'global_frontiers') and len(self.env.global_frontiers) > 0

                else None

            )

            num_steps = adaptive_scheduler.get_num_steps(

                belief_map     = self.env.robot_belief,

                robot_location = self.env.robot_locations[0],

                frontiers      = frontiers,

                cell_size      = self.env.cell_size,

                map_origin     = np.array([

                    self.env.belief_info.map_origin_x,

                    self.env.belief_info.map_origin_y,

                ]),

            )

            # ── END NEW ───────────────────────────────────────────────────────



            # Sample k action predictions

            with torch.no_grad():

                candidate_action_preds = []

                for _ in range(k_sampling_reranking_instance.get_k()):

                    # ── NEW: inject adaptive timesteps into the policy ────────

                    # The diffusion policy exposes its noise scheduler via

                    # self.policy.noise_scheduler.  We temporarily override the

                    # timestep sequence for this forward pass only.

                    original_timesteps = self.policy.noise_scheduler.timesteps

                    self.policy.noise_scheduler.timesteps = original_timesteps[:num_steps]



                    action_dict = self.policy.predict_action(obs_dict)



                    # Restore original timesteps (non-destructive)

                    self.policy.noise_scheduler.timesteps = original_timesteps

                    # ── END NEW ───────────────────────────────────────────────



                    candidate_action_preds.append(

                        action_dict['action_pred'].squeeze(0).cpu().numpy()

                    )



            k_sampling_reranking_instance.set_current_actions_to_be_ranked(candidate_action_preds)

            k_sampling_reranking_instance.set_robot_list(self.robot_list)

            k_sampling_reranking_instance.set_current_robot_location(self.env.robot_locations[0])

            best_idx    = k_sampling_reranking_instance.rank_actions_and_get_best_index()

            action_pred = candidate_action_preds[best_idx]



            # Round to nearest node resolution

            action_pred = np.round(action_pred / NODE_RESOLUTION) * NODE_RESOLUTION



            start  = self.obs_horizon - 1

            end    = start + self.action_horizon

            action = action_pred[start:end, :]



            for action_step in range(self.action_horizon):

                if action_step == 0:

                    planned_location = deepcopy(self.env.robot_locations[0])

                    self.planned_path_x.append([planned_location[0]])

                    self.planned_path_y.append([planned_location[1]])



                    if USE_DELTA_POSITION:

                        for i in range(start, len(action_pred)):

                            planned_location = planned_location + action_pred[i]

                            self.planned_path_x[step].append(planned_location[0])

                            self.planned_path_y[step].append(planned_location[1])

                    else:

                        for i in range(start, len(action_pred)):

                            planned_location = action_pred[i]

                            self.planned_path_x[step].append(planned_location[0])

                            self.planned_path_y[step].append(planned_location[1])

                else:

                    self.planned_path_x.append(self.planned_path_x[step - action_step])

                    self.planned_path_y.append(self.planned_path_y[step - action_step])



                if USE_DELTA_POSITION:

                    selected_coord = self.env.robot_locations[0] + action[action_step]

                else:

                    selected_coord = action[action_step]



                current_node = self.robot_list[0].node_manager.nodes_dict.find(

                    self.env.robot_locations[0].tolist()

                ).data



                # Collision avoidance (unchanged from original)

                if not any(np.all(selected_coord == neighbor) for neighbor in current_node.neighbor_list):

                    direction_vectors = np.cumsum(action_pred[start: start + 3], axis=0)

                    best_neighbor     = None

                    best_average_angle = float('inf')



                    for neighbor_coords in current_node.neighbor_list:

                        if np.all(neighbor_coords == self.env.robot_locations[0]):

                            continue

                        neighbor_direction = neighbor_coords - self.env.robot_locations[0]

                        angles  = []

                        for dv in direction_vectors:

                            dm = np.linalg.norm(dv)

                            nm = np.linalg.norm(neighbor_direction)

                            if dm == 0 or nm == 0:

                                continue

                            angle = np.arctan2(

                                np.linalg.det([dv, neighbor_direction]),

                                np.dot(dv, neighbor_direction)

                            )

                            angles.append(angle)

                        weights = np.arange(len(angles), 0, -1)

                        waa = np.average(np.abs(angles), weights=weights)

                        if waa < best_average_angle:

                            best_average_angle = waa

                            best_neighbor      = neighbor_coords



                    selected_coord = best_neighbor



                # Step environment

                self.env.step(selected_coord, 0)

                self.robot_list[0].update_graph(

                    self.env.belief_info, deepcopy(self.env.robot_locations[0])

                )

                self.robot_list[0].update_planning_state(self.env.robot_locations)



                # Next observation

                if DATA_TYPE == 'node':

                    observation       = self.robot_list[0].get_observation()

                    node_inputs       = observation[0].squeeze(0)

                    node_padding_mask = observation[1].squeeze(0)

                    edge_mask         = observation[2].squeeze(0)

                    current_index     = observation[3].squeeze(0)

                    current_edge      = observation[4].squeeze(0)

                    edge_padding_mask = observation[5].squeeze(0)

                    obs = {

                        'node_inputs':       node_inputs,

                        'node_padding_mask': node_padding_mask,

                        'edge_mask':         edge_mask,

                        'current_index':     current_index,

                        'current_edge':      current_edge,

                        'edge_padding_mask': edge_padding_mask,

                    }

                elif DATA_TYPE == 'map':

                    image     = deepcopy(self.env.robot_belief)

                    state     = deepcopy(self.env.robot_locations[0])

                    agent_pos = state.astype(np.float32)

                    image     = image.astype(np.float32) / 255

                    if len(image.shape) == 2:

                        image = np.expand_dims(image, axis=0)

                    obs = {'image': image, 'agent_pos': state}

                else:

                    raise ValueError('Invalid data type, check test_parameter.py')



                obs_deque.append(obs)



                if USE_EXPLORATION_RATE_FOR_DONE:

                    self.env.check_done()

                    done = self.env.done

                else:

                    done = self.robot_list[0].utility.sum() == 0



                if self.save_image:

                    self.plot_env(step)



                if done:

                    break



            if done:

                break



        self.perf_metrics['travel_dist']  = self.robot_list[0].travel_dist

        self.perf_metrics['success_rate'] = done



        # ── NEW: attach adaptive scheduler summary to metrics ─────────────────

        sched_summary = adaptive_scheduler.get_summary()

        self.perf_metrics.update({

            'adaptive_steps_mean':   sched_summary.get('steps_mean',       None),

            'adaptive_steps_min':    sched_summary.get('steps_min',        None),

            'adaptive_steps_max':    sched_summary.get('steps_max',        None),

            'complexity_mean':       sched_summary.get('complexity_mean',  None),

            'steps_saved_pct':       sched_summary.get('steps_saved_pct',  None),

        })

        # ── END NEW ───────────────────────────────────────────────────────────



        if self.save_image:

            make_gif(gifs_path, self.global_step, self.env.frame_files,

                     self.env.explored_rate, delete_images=True)



    def plot_env(self, step, planned_paths=None):

        self.env.global_frontiers = get_frontier_in_map(self.env.belief_info)



        plt.switch_backend('agg')

        color_list = ['r', 'b', 'g', 'y']

        plt.figure(figsize=(10, 5))



        plt.subplot(1, 2, 2)

        plt.imshow(self.env.robot_belief, cmap='gray')

        plt.axis('off')

        for robot in self.robot_list:

            c = color_list[robot.id]

            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)

            plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=16, zorder=5)

            plt.plot(

                (np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,

                (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size,

                c, linewidth=2, zorder=1,

            )

        plt.plot(

            (np.array(self.planned_path_x[step]) - self.env.belief_info.map_origin_x) / self.env.cell_size,

            (np.array(self.planned_path_y[step]) - self.env.belief_info.map_origin_y) / self.env.cell_size,

            'g', linewidth=1, zorder=2,

        )



        plt.subplot(1, 2, 1)

        plt.imshow(self.env.robot_belief, cmap='gray')

        for robot in self.robot_list:

            c = color_list[robot.id]

            if robot.id == 0:

                nodes = get_cell_position_from_coords(robot.node_coords, robot.map_info)

                plt.imshow(robot.map_info.map, cmap='gray')

                plt.axis('off')

                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2)

                for node, utility in zip(nodes, robot.utility):

                    plt.text(node[0], node[1], str(utility), zorder=3)

                robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)

                plt.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=16, zorder=5)



        if len(self.env.global_frontiers) > 0:

            frontiers = get_cell_position_from_coords(

                np.array(list(self.env.global_frontiers)), self.env.belief_info

            ).reshape(-1, 2)

            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)



        plt.axis('off')

        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(

            self.env.explored_rate,

            max([robot.travel_dist for robot in self.robot_list]),

        ))

        plt.tight_layout()

        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)

        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)

        self.env.frame_files.append(frame)

        plt.close()
