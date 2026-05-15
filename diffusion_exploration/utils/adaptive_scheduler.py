"""

adaptive_scheduler.py

---------------------

Adaptive Step Scheduler for DARE diffusion inference.



Dynamically adjusts the number of DDIM/DDPM denoising steps per episode

based on a per-observation complexity score derived from:

  1. Local obstacle density  (occupancy map)

  2. Normalized distance to the nearest frontier (proxy for goal distance)








Course: Perception in Robotics – Class Project

"""



from __future__ import annotations



import math

import time

import numpy as np

from dataclasses import dataclass, field

from typing import List, Optional





# ---------------------------------------------------------------------------

# Configuration

# ---------------------------------------------------------------------------



@dataclass

class AdaptiveSchedulerConfig:

    """All tuneable hyper-parameters in one place."""



    # Denoising step range

    t_min: int = 20          # steps for the easiest environments

    t_max: int = 100         # steps for the hardest environments  (DDPM default)



    # Complexity score weights  (must sum ≤ 1; remainder is ignored)

    w_density: float = 0.6   # weight for obstacle density

    w_distance: float = 0.4  # weight for frontier distance



    # Local window for density estimation (cells around robot)

    density_radius: int = 15



    # Maximum expected frontier distance for normalisation (cells)

    max_frontier_dist: float = 200.0



    # Smoothing: exponential moving average coefficient (0 = no smoothing)

    ema_alpha: float = 0.3



    # Logging

    log_every_n_steps: int = 50





# ---------------------------------------------------------------------------

# Core scheduler

# ---------------------------------------------------------------------------



class AdaptiveStepScheduler:

    """

    Computes a scalar complexity score C ∈ [0, 1] per inference call and

    maps it to an integer number of denoising steps T ∈ [T_min, T_max].



        C = w_density * density + w_distance * dist_norm

        T = round(T_min + (T_max - T_min) * C)



    Usage

    -----

    Instantiate once, then call ``get_num_steps(belief_map, robot_location,

    frontiers)`` before every diffusion inference call.  The returned integer

    is passed directly to the policy's scheduler:



        timesteps = policy.noise_scheduler.timesteps[:num_steps]

    """



    def __init__(self, cfg: Optional[AdaptiveSchedulerConfig] = None):

        self.cfg = cfg or AdaptiveSchedulerConfig()



        # EMA state

        self._ema_complexity: Optional[float] = None



        # Telemetry

        self._history: List[dict] = []

        self._call_count: int = 0

        self._total_time_saved: float = 0.0  # relative to always using T_max



    # ------------------------------------------------------------------

    # Public API

    # ------------------------------------------------------------------



    def get_num_steps(

        self,

        belief_map: np.ndarray,

        robot_location: np.ndarray,

        frontiers: Optional[np.ndarray] = None,

        cell_size: float = 1.0,

        map_origin: Optional[np.ndarray] = None,

    ) -> int:

        """

        Compute adaptive number of denoising steps for the current observation.



        Parameters

        ----------

        belief_map      : 2-D uint8 array (H × W). 255 = free, 0 = occupied,

                          127 (or similar) = unknown.  As used by DARE's Env.

        robot_location  : (2,) array with robot world-coords [x, y].

        frontiers       : (N, 2) array of frontier world-coords, or None.

        cell_size       : metres per cell (for distance normalisation).

        map_origin      : (2,) array [origin_x, origin_y] in world coords.



        Returns

        -------

        int  –  number of denoising steps to use for this inference call.

        """

        t0 = time.perf_counter()



        density  = self._obstacle_density(belief_map, robot_location,

                                          cell_size, map_origin)

        dist_norm = self._frontier_distance_norm(belief_map, robot_location,

                                                 frontiers, cell_size, map_origin)



        raw_complexity = (self.cfg.w_density   * density

                        + self.cfg.w_distance  * dist_norm)

        raw_complexity = float(np.clip(raw_complexity, 0.0, 1.0))



        # Exponential moving average smoothing

        if self._ema_complexity is None:

            self._ema_complexity = raw_complexity

        else:

            α = self.cfg.ema_alpha

            self._ema_complexity = α * raw_complexity + (1 - α) * self._ema_complexity



        complexity = self._ema_complexity

        num_steps  = int(round(

            self.cfg.t_min + (self.cfg.t_max - self.cfg.t_min) * complexity

        ))

        num_steps  = int(np.clip(num_steps, self.cfg.t_min, self.cfg.t_max))



        # Bookkeeping

        elapsed = time.perf_counter() - t0

        self._call_count += 1

        self._total_time_saved += (self.cfg.t_max - num_steps)   # relative



        record = {

            "call":         self._call_count,

            "density":      round(density,       4),

            "dist_norm":    round(dist_norm,      4),

            "complexity":   round(complexity,     4),

            "num_steps":    num_steps,

            "overhead_ms":  round(elapsed * 1000, 3),

        }

        self._history.append(record)



        if self._call_count % self.cfg.log_every_n_steps == 0:

            self._log(record)



        return num_steps



    # ------------------------------------------------------------------

    # Complexity sub-components

    # ------------------------------------------------------------------



    def _obstacle_density(

        self,

        belief_map: np.ndarray,

        robot_location: np.ndarray,

        cell_size: float,

        map_origin: Optional[np.ndarray],

    ) -> float:

        """

        Fraction of occupied cells in a square window of radius

        ``density_radius`` centred on the robot.



        DARE uses belief_map values: 255 = free, 0 = occupied, ~127 = unknown.

        We count cells whose value < 50 as obstacles.

        """

        r   = self.cfg.density_radius

        row, col = self._world_to_cell(robot_location, belief_map, cell_size, map_origin)



        r0 = max(0,               row - r)

        r1 = min(belief_map.shape[0], row + r + 1)

        c0 = max(0,               col - r)

        c1 = min(belief_map.shape[1], col + r + 1)



        patch = belief_map[r0:r1, c0:c1]

        if patch.size == 0:

            return 0.0



        n_occupied = np.sum(patch < 50)

        n_known    = np.sum(patch != 127)   # exclude unknown cells

        if n_known == 0:

            return 0.0



        return float(n_occupied) / float(n_known)



    def _frontier_distance_norm(

        self,

        belief_map: np.ndarray,

        robot_location: np.ndarray,

        frontiers: Optional[np.ndarray],

        cell_size: float,

        map_origin: Optional[np.ndarray],

    ) -> float:

        """

        Normalised distance to nearest frontier.  High distance → high

        complexity score (robot is far from the exploration boundary).



        Falls back to map-diagonal normalisation when no frontiers are given.

        """

        if frontiers is None or len(frontiers) == 0:

            # Fallback: use fraction of unexplored map as proxy

            unknown = np.sum(belief_map == 127)

            total   = belief_map.size

            return float(unknown) / float(total)



        # Euclidean distance in world coordinates

        diffs  = frontiers - robot_location        # (N, 2)

        dists  = np.linalg.norm(diffs, axis=1)    # (N,)

        min_d  = float(np.min(dists))



        # Normalise by max expected distance (in world units)

        max_d  = self.cfg.max_frontier_dist * cell_size

        return float(np.clip(min_d / max_d, 0.0, 1.0))



    # ------------------------------------------------------------------

    # Helpers

    # ------------------------------------------------------------------



    @staticmethod

    def _world_to_cell(

        world_coords: np.ndarray,

        belief_map: np.ndarray,

        cell_size: float,

        map_origin: Optional[np.ndarray],

    ):

        """Convert world (x, y) → (row, col) in belief_map."""

        if map_origin is None:

            map_origin = np.array([0.0, 0.0])



        col = int((world_coords[0] - map_origin[0]) / cell_size)

        row = int((world_coords[1] - map_origin[1]) / cell_size)



        # Clamp to valid indices

        row = int(np.clip(row, 0, belief_map.shape[0] - 1))

        col = int(np.clip(col, 0, belief_map.shape[1] - 1))

        return row, col



    # ------------------------------------------------------------------

    # Reporting

    # ------------------------------------------------------------------



    def get_summary(self) -> dict:

        """Return episode-level statistics for logging / CSV export."""

        if not self._history:

            return {}



        steps_arr  = np.array([h["num_steps"]  for h in self._history])

        comp_arr   = np.array([h["complexity"]  for h in self._history])



        return {

            "n_calls":          self._call_count,

            "steps_mean":       float(np.mean(steps_arr)),

            "steps_min":        int(np.min(steps_arr)),

            "steps_max":        int(np.max(steps_arr)),

            "complexity_mean":  float(np.mean(comp_arr)),

            "steps_saved_pct":  round(

                100.0 * self._total_time_saved

                / (self._call_count * self.cfg.t_max), 2

            ) if self._call_count > 0 else 0.0,

        }



    def reset(self):

        """Call between episodes to clear EMA state and history."""

        self._ema_complexity = None

        self._history.clear()

        self._call_count     = 0

        self._total_time_saved = 0.0



    def _log(self, record: dict):

        print(

            f"[AdaptiveScheduler] call={record['call']:>5d} | "

            f"density={record['density']:.3f}  dist={record['dist_norm']:.3f}  "

            f"complexity={record['complexity']:.3f}  "

            f"steps={record['num_steps']:>3d}  "

            f"overhead={record['overhead_ms']:.2f}ms"

        )





# ---------------------------------------------------------------------------

# Module-level singleton (drop-in compatible with k_sampling_reranking style)

# ---------------------------------------------------------------------------



_default_scheduler = AdaptiveStepScheduler()





def get_adaptive_num_steps(

    belief_map: np.ndarray,

    robot_location: np.ndarray,

    frontiers: Optional[np.ndarray] = None,

    cell_size: float = 1.0,

    map_origin: Optional[np.ndarray] = None,

) -> int:

    """Convenience wrapper around the module-level singleton."""

    return _default_scheduler.get_num_steps(

        belief_map, robot_location, frontiers, cell_size, map_origin

    )

 

 

def reset_scheduler():

    """Reset the singleton between episodes."""

    _default_scheduler.reset()

 

 

def get_scheduler_summary() -> dict:

    """Return summary stats from the singleton."""

    return _default_scheduler.get_summary()
