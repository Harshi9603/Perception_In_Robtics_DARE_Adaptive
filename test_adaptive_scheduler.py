"""

test_adaptive_scheduler.py

--------------------------

Unit tests for the AdaptiveStepScheduler.



Run with:

    python test_adaptive_scheduler.py

or:

    python -m pytest test_adaptive_scheduler.py -v

"""



import sys

import math

import numpy as np

import unittest



# Allow running from repo root

sys.path.insert(0, ".")



from diffusion_exploration.utils.adaptive_scheduler import (

    AdaptiveStepScheduler,

    AdaptiveSchedulerConfig,

    get_adaptive_num_steps,

    reset_scheduler,

    get_scheduler_summary,

)





# ---------------------------------------------------------------------------

# Helpers

# ---------------------------------------------------------------------------



def make_open_map(h=100, w=100):

    """All-free 2-D belief map (255 = free)."""

    return np.full((h, w), 255, dtype=np.uint8)





def make_cluttered_map(h=100, w=100, obstacle_frac=0.4):

    """Map with obstacle_frac of cells set to 0 (occupied)."""

    m = np.full((h, w), 255, dtype=np.uint8)

    n_obs = int(h * w * obstacle_frac)

    idx = np.random.choice(h * w, n_obs, replace=False)

    m.flat[idx] = 0

    return m





def make_frontiers(robot_loc, distances):

    """Synthetic frontier array at given distances from robot."""

    fronts = []

    for d in distances:

        fronts.append(robot_loc + np.array([d, 0.0]))

    return np.array(fronts)





# ---------------------------------------------------------------------------

# Tests

# ---------------------------------------------------------------------------



class TestAdaptiveSchedulerConfig(unittest.TestCase):



    def test_defaults(self):

        cfg = AdaptiveSchedulerConfig()

        self.assertGreater(cfg.t_max, cfg.t_min)

        self.assertAlmostEqual(cfg.w_density + cfg.w_distance, 1.0, places=5)



    def test_custom(self):

        cfg = AdaptiveSchedulerConfig(t_min=10, t_max=50)

        self.assertEqual(cfg.t_min, 10)

        self.assertEqual(cfg.t_max, 50)





class TestAdaptiveStepScheduler(unittest.TestCase):



    def setUp(self):

        self.cfg = AdaptiveSchedulerConfig(

            t_min=20, t_max=100,

            w_density=0.6, w_distance=0.4,

            ema_alpha=0.0,   # disable EMA for deterministic tests

            log_every_n_steps=9999,

        )

        self.sched = AdaptiveStepScheduler(cfg=self.cfg)

        self.robot = np.array([50.0, 50.0])



    # -- Output range --



    def test_output_in_range_open(self):

        bmap = make_open_map()

        fronts = make_frontiers(self.robot, [5.0])

        steps = self.sched.get_num_steps(bmap, self.robot, fronts)

        self.assertGreaterEqual(steps, self.cfg.t_min)

        self.assertLessEqual(steps,    self.cfg.t_max)



    def test_output_in_range_cluttered(self):

        bmap = make_cluttered_map(obstacle_frac=0.5)

        fronts = make_frontiers(self.robot, [180.0])

        steps = self.sched.get_num_steps(bmap, self.robot, fronts)

        self.assertGreaterEqual(steps, self.cfg.t_min)

        self.assertLessEqual(steps,    self.cfg.t_max)



    # -- Monotonicity --



    def test_more_obstacles_more_steps(self):

        """Cluttered map should yield at least as many steps as open map."""

        fronts = make_frontiers(self.robot, [50.0])



        sched_open = AdaptiveStepScheduler(cfg=self.cfg)

        steps_open = sched_open.get_num_steps(make_open_map(), self.robot, fronts)



        sched_clut = AdaptiveStepScheduler(cfg=self.cfg)

        steps_clut = sched_clut.get_num_steps(

            make_cluttered_map(obstacle_frac=0.5), self.robot, fronts

        )



        self.assertGreaterEqual(steps_clut, steps_open,

            f"Expected steps_clut ({steps_clut}) >= steps_open ({steps_open})")



    def test_far_frontier_more_steps_than_near(self):

        """Farther frontier → higher complexity → more steps."""

        bmap = make_open_map()



        sched_near = AdaptiveStepScheduler(cfg=self.cfg)

        steps_near = sched_near.get_num_steps(

            bmap, self.robot, make_frontiers(self.robot, [10.0])

        )



        sched_far = AdaptiveStepScheduler(cfg=self.cfg)

        steps_far = sched_far.get_num_steps(

            bmap, self.robot, make_frontiers(self.robot, [180.0])

        )



        self.assertGreaterEqual(steps_far, steps_near,

            f"Expected steps_far ({steps_far}) >= steps_near ({steps_near})")



    # -- Boundary / edge cases --



    def test_no_frontiers_fallback(self):

        """Should not crash when frontiers=None; falls back to unknown fraction."""

        bmap = make_open_map()

        # Mark half as unknown

        bmap[:50, :] = 127

        steps = self.sched.get_num_steps(bmap, self.robot, frontiers=None)

        self.assertGreaterEqual(steps, self.cfg.t_min)

        self.assertLessEqual(steps,    self.cfg.t_max)



    def test_empty_frontier_array(self):

        bmap  = make_open_map()

        steps = self.sched.get_num_steps(bmap, self.robot, frontiers=np.empty((0, 2)))

        self.assertGreaterEqual(steps, self.cfg.t_min)

        self.assertLessEqual(steps,    self.cfg.t_max)



    def test_robot_at_map_edge(self):

        """Robot at boundary should not cause index-out-of-bounds."""

        bmap = make_open_map(50, 50)

        for loc in [np.array([0.0, 0.0]), np.array([49.0, 49.0])]:

            steps = self.sched.get_num_steps(bmap, loc, frontiers=None)

            self.assertGreaterEqual(steps, self.cfg.t_min)



    # -- EMA smoothing --



    def test_ema_smooths_transitions(self):

        """EMA should prevent extreme jumps in step count between calls."""

        cfg = AdaptiveSchedulerConfig(t_min=20, t_max=100, ema_alpha=0.3,

                                       log_every_n_steps=9999)

        sched  = AdaptiveStepScheduler(cfg=cfg)

        fronts = make_frontiers(self.robot, [10.0])



        prev_steps = sched.get_num_steps(make_open_map(), self.robot, fronts)

        # Next call: very different environment

        next_steps = sched.get_num_steps(

            make_cluttered_map(obstacle_frac=0.5), self.robot, make_frontiers(self.robot, [180.0])

        )

        # Absolute jump must be less than t_max - t_min (full range)

        self.assertLess(abs(next_steps - prev_steps), cfg.t_max - cfg.t_min,

            "EMA should dampen the transition")



    # -- Reset --



    def test_reset_clears_state(self):

        bmap   = make_open_map()

        fronts = make_frontiers(self.robot, [50.0])

        self.sched.get_num_steps(bmap, self.robot, fronts)

        self.sched.get_num_steps(bmap, self.robot, fronts)

        self.sched.reset()

        summary = self.sched.get_summary()

        self.assertEqual(summary, {})



    # -- Summary --



    def test_summary_fields(self):

        bmap   = make_open_map()

        fronts = make_frontiers(self.robot, [30.0])

        for _ in range(5):

            self.sched.get_num_steps(bmap, self.robot, fronts)

        summary = self.sched.get_summary()

        for key in ['n_calls', 'steps_mean', 'steps_min', 'steps_max',

                    'complexity_mean', 'steps_saved_pct']:

            self.assertIn(key, summary, f"Missing key: {key}")



    def test_summary_n_calls(self):

        bmap   = make_open_map()

        fronts = make_frontiers(self.robot, [30.0])

        N = 7

        for _ in range(N):

            self.sched.get_num_steps(bmap, self.robot, fronts)

        self.assertEqual(self.sched.get_summary()['n_calls'], N)





class TestModuleLevelAPI(unittest.TestCase):

    """Test the convenience singleton wrappers."""



    def setUp(self):

        reset_scheduler()



    def test_get_adaptive_num_steps_returns_int(self):

        bmap  = make_open_map()

        robot = np.array([50.0, 50.0])

        steps = get_adaptive_num_steps(bmap, robot)

        self.assertIsInstance(steps, int)



    def test_summary_after_calls(self):

        bmap  = make_open_map()

        robot = np.array([50.0, 50.0])

        for _ in range(3):

            get_adaptive_num_steps(bmap, robot)

        s = get_scheduler_summary()

        self.assertEqual(s['n_calls'], 3)



    def test_reset_between_episodes(self):

        bmap  = make_open_map()

        robot = np.array([50.0, 50.0])

        get_adaptive_num_steps(bmap, robot)

        reset_scheduler()

        s = get_scheduler_summary()

        self.assertEqual(s, {})





# ---------------------------------------------------------------------------

# Main

# ---------------------------------------------------------------------------



if __name__ == "__main__":

    print("=" * 60)

    print("AdaptiveStepScheduler – Unit Tests")

    print("=" * 60)

    unittest.main(verbosity=2)
