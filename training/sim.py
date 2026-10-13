"""The training simulator: f1tenth_gym's GPU (JAX) simulator, set up to
match the judged race as closely as it can, plus the driver's own
controller running inside it.

What matches the race (external/f1tenth_gym_ros/f1tenth_gym_ros/gym_bridge.py
and race_ws/src/roboracer_referee/launch/simulator.launch.py):
    car model ST (single-track, with tyre slip), RK4 integrator, 0.01 s steps,
    f1tenth vehicle parameters, target speed + steering angle control,
    the icra26 map and centreline, LiDAR 0.275 m ahead of the rear axle with
    no noise, odometry position reported at the rear axle.

What differs, on purpose:
    - only the LiDAR beams in the driver's "ahead" cone are simulated, at
      exactly the race's beam angles. The driver uses no other beams, and
      simulating all 819 makes training about 4x slower.
    - the driver acts on every 0.01 s physics step. That is what the race
      does too: one scan per physics step was measured on 2026-10-03.

Reaction delay (see `delay_commands`): in ROS a command reaches the physics
a few steps after the scan it answers. That matters a lot on icra26: in the
race's own engine, pure pursuit crashes in the infield hairpin with a delay
of 0-1 steps but laps (17.2 s) with 2-3 steps (measured 2026-10-11). The
real delay varies with the computer, so every episode gets a random delay of
0..MAX_DELAY steps and the network has to cope with all of them.
"""

import math
import os
import sys
from pathlib import Path

# Do not grab 75% of the GPU's memory up front: under WSL that first grab
# fails and prints pages of out-of-memory errors before it retries.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
import numpy as np

# Make the driver's ROS-free modules importable (race_ws/src/team_driver).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'race_ws' / 'src' / 'team_driver'))

from team_driver.pure_pursuit import DEFAULTS, preprocess, pure_pursuit  # noqa: E402
from team_driver.raceline import (  # noqa: E402
    CAR_HALF_WIDTH, ITERS, WALL_GAP, build_raceline, load_clearance_map, speed_table)
from team_driver.residual import (  # noqa: E402
    features, lookahead_inputs, path_geometry, speed_table_inputs)

from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType  # noqa: E402
from f1tenth_gym.envs.contact import ContactConfig  # noqa: E402
from f1tenth_gym.envs.dynamic_models import DynamicModel, F1TENTH_VEHICLE_PARAMETERS  # noqa: E402
from f1tenth_gym.envs.env_config import (  # noqa: E402
    ControlConfig, EnvConfig, ResetConfig, RewardConfig, RewardMode,
    SimulationConfig, TerminationConfig)
from f1tenth_gym.envs.integrators import IntegratorType  # noqa: E402
from f1tenth_gym.envs.jax_simulator import JaxSimulator  # noqa: E402
from f1tenth_gym.envs.lidar import LiDARConfig  # noqa: E402
from f1tenth_gym.envs.reset import ResetStrategy  # noqa: E402
from f1tenth_gym.envs.track import Track  # noqa: E402

MAP_YAML = REPO / 'maps' / 'icra26.yaml'
CENTRELINE_CSV = REPO / 'maps' / 'icra26_centerline.csv'
START_POSE = (-1.68, -0.01, 0.0)    # grid slot from maps/tracks.yaml (rear axle)
TIMESTEP = 0.01                     # [s] one physics step
LR = float(F1TENTH_VEHICLE_PARAMETERS.lr)   # [m] centre of mass to rear axle
MAX_DELAY = 3                       # [steps] largest reaction delay simulated

# The race's LiDAR (simulator.launch.py + the bridge's sim.yaml).
SCAN_BEAMS = 819
SCAN_MIN_DEG, SCAN_MAX_DEG = -135.0, 135.0
SCAN_RANGE = (0.05, 25.0)
LIDAR_X = 0.275                     # [m] ahead of the rear axle


def racing_line():
    """The racing line, built by the same code and settings as the driver."""
    centre = np.loadtxt(CENTRELINE_CSV, delimiter=',', comments='#')
    clearance = load_clearance_map(str(MAP_YAML))
    return build_raceline(centre, clearance, CAR_HALF_WIDTH + WALL_GAP, ITERS)


def driver_speed_table(path, curvature):
    """The speed table the driver would use (as a jax array), or None when
    DEFAULTS['use_speed_table'] is off, exactly as driver.py decides."""
    if not DEFAULTS['use_speed_table']:
        return None
    return jnp.asarray(speed_table(path, curvature, DEFAULTS))


def scaled(speeds):
    """The table slowed down by DEFAULTS['table_scale'], as driver.py does,
    or None if there is no table."""
    return None if speeds is None else jnp.asarray(speeds) * DEFAULTS['table_scale']


def scan_slice():
    """The race's beams that the driver's "ahead" cone can see, plus two
    spare beams each side. Returns (first angle, last angle, beam count),
    angles in radians. The beam angles are exactly the race's."""
    increment = (SCAN_MAX_DEG - SCAN_MIN_DEG) / (SCAN_BEAMS - 1)
    half = DEFAULTS['ahead_angle']
    first = math.floor((-half - SCAN_MIN_DEG) / increment) - 2
    last = math.ceil((half - SCAN_MIN_DEG) / increment) + 2
    return (math.radians(SCAN_MIN_DEG + first * increment),
            math.radians(SCAN_MIN_DEG + last * increment),
            last - first + 1)


def make_simulator(device, training, episode_steps=3000, collision_penalty=10.0):
    """One simulator for a whole batch of cars on icra26.

    training=True: a crash ends the episode, episodes last at most
    `episode_steps`, and the built-in reward is the metres of progress along
    the centreline each step, minus `collision_penalty` on a crash.
    training=False (evaluation): crashes do not end anything, like the race.
    """
    track = Track.from_track_path(MAP_YAML)
    first, last, beams = scan_slice()
    config = EnvConfig(
        map_name=track,
        num_agents=1,
        params=F1TENTH_VEHICLE_PARAMETERS,
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.SPEED,
            steering_mode=SteerActionType.STEERING_ANGLE,
        ),
        simulation_config=SimulationConfig(
            timestep=TIMESTEP,
            integrator_timestep=TIMESTEP,
            integrator=IntegratorType.RK4,
            dynamics_model=DynamicModel.ST,
            max_laps=None,
            count_partial_first_lap=False,
        ),
        # Training episodes start at a random point of the track, standing
        # still, so the network practises every corner, not just the first.
        reset_config=ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC),
        lidar_config=LiDARConfig(
            enabled=True, num_beams=beams, angle_min=first, angle_max=last,
            range_min=SCAN_RANGE[0], range_max=SCAN_RANGE[1],
            noise_std=0.0, base_link_to_lidar_tf=(LIDAR_X, 0.0, 0.0),
        ),
        contact_config=ContactConfig(device=device.platform),
        termination_config=TerminationConfig(
            max_episode_steps=episode_steps if training else None,
            terminate_on_collision=training,
        ),
        reward_config=RewardConfig(
            mode=RewardMode.PROGRESS,
            progress_weight=1.0,
            collision_penalty=collision_penalty,
        ),
        render_enabled=False,
    )
    return JaxSimulator(config, track, device=device)


def make_controller(path, speeds=None):
    """Return controller(observation, last_steer) for a batch of cars.

    For every car it runs the driver's pure pursuit and builds the residual
    network's inputs, exactly as driver.py does, and returns
    (base_steer, base_speed, features), each with one row per car.
    `speeds`: a speed table to drive by; None = whatever the driver would use.
    """
    heading, curvature = path_geometry(path)
    speeds = scaled(driver_speed_table(path, curvature) if speeds is None else speeds)
    path, heading, curvature = jnp.asarray(path), jnp.asarray(heading), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def one_car(car, yaw, speed, yaw_rate, ranges, last_steer):
        steer, v, nearest, ahead = pure_pursuit(
            path, car, yaw, speed, ranges, angles, DEFAULTS, xp=jnp, speed_table=speeds)
        x = features(path, heading, curvature, car, yaw, speed, yaw_rate, last_steer,
                     steer, v, nearest, ahead, DEFAULTS, xp=jnp)
        return steer, v, x

    def controller(observation, last_steer):
        car, yaw, speed, yaw_rate = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(one_car)(car, yaw, speed, yaw_rate, ranges, last_steer)

    return controller


def make_lookahead_controller(path, speeds=None):
    """For the dynamic-lookahead policy. Returns two functions, each taking
    a batch of cars:
        inputs(observation)        -> (cars, 5) the policy's inputs s
        drive(observation, L)      -> (steering, speed) from pure pursuit
                                      aiming L metres ahead (one L per car)
    `speeds`: a speed table to drive by; None = whatever the driver would use.
    """
    _, curvature = path_geometry(path)
    speeds = scaled(driver_speed_table(path, curvature) if speeds is None else speeds)
    path, curvature = jnp.asarray(path), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def inputs_one_car(car, speed):
        nearest = jnp.argmin(jnp.linalg.norm(path - car, axis=1))
        return lookahead_inputs(curvature, nearest, speed, xp=jnp)

    def drive_one_car(car, yaw, speed, ranges, distance):
        steer, v, _, _ = pure_pursuit(path, car, yaw, speed, ranges, angles, DEFAULTS,
                                      xp=jnp, lookahead=distance, speed_table=speeds)
        return steer, v

    def inputs(observation):
        car, _, speed, _ = odometry(observation)
        return jax.vmap(inputs_one_car)(car, speed)

    def drive(observation, distance):
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(drive_one_car)(car, yaw, speed, ranges, distance)

    return inputs, drive


def make_speed_table_controller(path):
    """For training the speed table. Returns two functions, each taking a
    batch of cars:
        observe(observation) -> (critic inputs (cars, 7), nearest path point (cars,))
        steer(observation)   -> steering from pure pursuit with its own
                                lookahead rule (the speed is the policy's choice)
    """
    _, curvature = path_geometry(path)
    path, curvature = jnp.asarray(path), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def observe_one_car(car, speed):
        nearest = jnp.argmin(jnp.linalg.norm(path - car, axis=1))
        return speed_table_inputs(curvature, nearest, speed, xp=jnp), nearest

    def steer_one_car(car, yaw, speed, ranges):
        steer, _, _, _ = pure_pursuit(path, car, yaw, speed, ranges, angles, DEFAULTS, xp=jnp)
        return steer

    def observe(observation):
        car, _, speed, _ = odometry(observation)
        return jax.vmap(observe_one_car)(car, speed)

    def steer(observation):
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(steer_one_car)(car, yaw, speed, ranges)

    return observe, steer


def odometry(observation):
    """What /ego_racecar/odom would report for each car: rear-axle position,
    heading, speed and yaw rate. The simulator itself tracks the centre of
    mass, which is LR metres ahead of the rear axle."""
    state = observation.standard_state[:, 0]   # [x, y, steer, speed, yaw, yaw rate, slip]
    yaw = state[:, 4]
    car = state[:, :2] - LR * jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=1)
    return car, yaw, jnp.abs(state[:, 3]), state[:, 5]


def delay_commands(queue, command, delay):
    """Simulate reaction delay. `queue` holds each car's last MAX_DELAY + 1
    commands, newest first, shape (cars, MAX_DELAY + 1, 2). The new command
    goes in at the front and each car gets the one from `delay` steps ago.
    Returns (new queue, commands to apply)."""
    queue = jnp.concatenate([command[:, None, :], queue[:, :-1]], axis=1)
    return queue, queue[jnp.arange(queue.shape[0]), delay]


def start_poses(count):
    """The race's grid slot as the simulator wants it: centre of mass."""
    x, y, yaw = START_POSE
    pose = [x + LR * math.cos(yaw), y + LR * math.sin(yaw), yaw]
    return jnp.tile(jnp.asarray(pose, dtype=jnp.float32), (count, 1, 1))
