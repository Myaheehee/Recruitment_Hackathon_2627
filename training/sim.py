"""Training simulator: f1tenth_gym's JAX simulator set up like the race, plus batched controllers.

Imported by train.py, evaluate.py, speed_table.py and test_parity.py. It matches the race's car
model, map, LiDAR mounting and odometry. What differs from the race:
- only the LiDAR beams in the driver's "ahead" cone are simulated (the driver uses no others);
- the driver acts every 0.01 s step (measured in the race; the rules say "roughly 40 Hz");
- each episode gets one random reaction delay of 0..MAX_DELAY steps (see delay_commands);
  a delay that changes mid-episode, and skipped scans, are not modelled.
"""

import math
import os
import sys
from pathlib import Path
from typing import Callable

# Don't grab 75% of the GPU up front: under WSL that fails, with pages of errors, before retrying.
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
from team_driver.policies import (  # noqa: E402
    lookahead_inputs, path_geometry, speed_table_inputs)
from team_driver.units import Array, Metres, MetresPerSecond, Radians  # noqa: E402

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
START_POSE = (-1.68, -0.01, 0.0)    # grid slot (maps/tracks.yaml): rear axle x, y [m], yaw [rad]
TIMESTEP = 0.01                     # [s] one physics step
LR = float(F1TENTH_VEHICLE_PARAMETERS.lr)   # [m] centre of mass to rear axle
MAX_DELAY = 3                       # [steps] largest reaction delay; --max-delay overrides it

# The race's LiDAR (simulator.launch.py + the bridge's sim.yaml).
SCAN_BEAMS = 819                    # beams over the full field of view
SCAN_MIN_DEG, SCAN_MAX_DEG = -135.0, 135.0   # [deg]
SCAN_RANGE = (0.05, 25.0)           # [m] shortest, longest reading
LIDAR_X = 0.275                     # [m] ahead of the rear axle


def racing_line() -> Array:
    """The driver's racing line, built by the same code: (points, 2) map x, y [m], ~0.2 m apart."""
    centre = np.loadtxt(CENTRELINE_CSV, delimiter=',', comments='#')
    clearance = load_clearance_map(str(MAP_YAML))
    return build_raceline(centre, clearance, CAR_HALF_WIDTH + WALL_GAP, ITERS)


def driver_speed_table(path: Array, curvature: Array) -> jax.Array | None:
    """The untuned speed table [m/s] driver.py would build; None if use_speed_table is off."""
    if not DEFAULTS['use_speed_table']:
        return None
    return jnp.asarray(speed_table(path, curvature, DEFAULTS))


def scaled(speeds: Array | None) -> jax.Array | None:
    """Multiply a speed table by DEFAULTS['table_scale'], as driver.py does (None stays None)."""
    return None if speeds is None else jnp.asarray(speeds) * DEFAULTS['table_scale']


def scan_slice() -> tuple[Radians, Radians, int]:
    """The race's beams covering the "ahead" cone, +2 spare each side: first, last angle, count."""
    increment = (SCAN_MAX_DEG - SCAN_MIN_DEG) / (SCAN_BEAMS - 1)
    half = DEFAULTS['ahead_angle']
    first = math.floor((-half - SCAN_MIN_DEG) / increment) - 2
    last = math.ceil((half - SCAN_MIN_DEG) / increment) + 2
    return (math.radians(SCAN_MIN_DEG + first * increment),
            math.radians(SCAN_MIN_DEG + last * increment),
            last - first + 1)


def make_simulator(device: jax.Device, training: bool, episode_steps: int = 3000,
                   collision_penalty: float = 10.0) -> JaxSimulator:
    """One simulator for a whole batch of cars on icra26.

    training=True: a crash ends the episode, episodes last at most episode_steps, and the reward is
    metres of progress per step minus collision_penalty on a crash. False: race rules (no ending).
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
        # Training episodes start standing still at a random point, so every corner gets practised.
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


def make_controller(path: Array, speeds: Array | None = None) -> Callable:
    """Plain pure pursuit for a batch of cars, as driver.py runs it with no lookahead policy.

    Returns controller(observation) -> (steering, speed), each shape (cars,).
    speeds=None means the driver's own table; any table is scaled by table_scale.
    """
    _, curvature = path_geometry(path)
    speeds = scaled(driver_speed_table(path, curvature) if speeds is None else speeds)
    path = jnp.asarray(path)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def one_car(car: jax.Array, yaw: Radians, speed: MetresPerSecond,
                ranges: jax.Array) -> tuple[Radians, MetresPerSecond]:
        """Pure pursuit for one car."""
        steer, v, _, _ = pure_pursuit(
            path, car, yaw, speed, ranges, angles, DEFAULTS, xp=jnp, speed_table=speeds)
        return steer, v

    def controller(observation) -> tuple[jax.Array, jax.Array]:
        """Run one_car on every car at once (jax.vmap)."""
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(one_car)(car, yaw, speed, ranges)

    return controller


def make_lookahead_controller(path: Array,
                              speeds: Array | None = None) -> tuple[Callable, Callable]:
    """The two batched functions the dynamic-lookahead policy needs.

    inputs(observation) -> the policy's 5 inputs, shape (cars, 5);
    drive(observation, distance) -> pure pursuit's (steering, speed), aiming distance[i] ahead.
    """
    _, curvature = path_geometry(path)
    speeds = scaled(driver_speed_table(path, curvature) if speeds is None else speeds)
    path, curvature = jnp.asarray(path), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def inputs_one_car(car: jax.Array, speed: MetresPerSecond) -> jax.Array:
        """The policy's 5 inputs for one car."""
        nearest = jnp.argmin(jnp.linalg.norm(path - car, axis=1))
        return lookahead_inputs(curvature, nearest, speed, xp=jnp)

    def drive_one_car(car: jax.Array, yaw: Radians, speed: MetresPerSecond, ranges: jax.Array,
                      distance: Metres) -> tuple[Radians, MetresPerSecond]:
        """Pure pursuit for one car, aiming `distance` ahead."""
        steer, v, _, _ = pure_pursuit(path, car, yaw, speed, ranges, angles, DEFAULTS,
                                      xp=jnp, lookahead=distance, speed_table=speeds)
        return steer, v

    def inputs(observation) -> jax.Array:
        """Run inputs_one_car on every car at once (jax.vmap)."""
        car, _, speed, _ = odometry(observation)
        return jax.vmap(inputs_one_car)(car, speed)

    def drive(observation, distance: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Run drive_one_car on every car at once; `distance` has one entry per car."""
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(drive_one_car)(car, yaw, speed, ranges, distance)

    return inputs, drive


def make_speed_table_controller(path: Array) -> tuple[Callable, Callable]:
    """The two batched functions that training the speed table needs.

    observe(observation) -> (critic inputs (cars, 7), nearest path point (cars,));
    steer(observation) -> pure pursuit's steering (cars,); the speed is the policy's choice.
    """
    _, curvature = path_geometry(path)
    path, curvature = jnp.asarray(path), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def observe_one_car(car: jax.Array, speed: MetresPerSecond) -> tuple[jax.Array, jax.Array]:
        """Critic inputs (7,) and nearest racing-line point for one car."""
        nearest = jnp.argmin(jnp.linalg.norm(path - car, axis=1))
        return speed_table_inputs(curvature, nearest, speed, xp=jnp), nearest

    def steer_one_car(car: jax.Array, yaw: Radians, speed: MetresPerSecond,
                      ranges: jax.Array) -> Radians:
        """Pure pursuit's steering for one car, with its own lookahead rule."""
        steer, _, _, _ = pure_pursuit(path, car, yaw, speed, ranges, angles, DEFAULTS, xp=jnp)
        return steer

    def observe(observation) -> tuple[jax.Array, jax.Array]:
        """Run observe_one_car on every car at once."""
        car, _, speed, _ = odometry(observation)
        return jax.vmap(observe_one_car)(car, speed)

    def steer(observation) -> jax.Array:
        """Run steer_one_car on every car at once."""
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(steer_one_car)(car, yaw, speed, ranges)

    return observe, steer


def odometry(observation) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """What /ego_racecar/odom reports for each car: position [m], yaw [rad], speed, yaw rate.

    Position is the rear axle, shape (cars, 2); the simulator tracks the centre of mass, LR ahead
    of it. Speed is unsigned (its size only).
    """
    state = observation.standard_state[:, 0]   # [x, y, steer, speed, yaw, yaw rate, slip]
    yaw = state[:, 4]
    car = state[:, :2] - LR * jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=1)
    return car, yaw, jnp.abs(state[:, 3]), state[:, 5]


def delay_commands(queue: jax.Array, command: jax.Array,
                   delay: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Reaction delay: push each car's new command; return (queue, the one from `delay` steps ago).

    queue is (cars, MAX_DELAY + 1, 2), newest first; command is (cars, 2) of (steering, speed);
    delay is (cars,) ints. The queue starts as zeros, so a car gets (0, 0) for its first steps.
    """
    queue = jnp.concatenate([command[:, None, :], queue[:, :-1]], axis=1)
    return queue, queue[jnp.arange(queue.shape[0]), delay]


def start_poses(count: int) -> jax.Array:
    """The race's grid slot for `count` cars, as the simulator wants it (centre of mass).

    Shape (count, 1, 3) of (x, y, yaw); the 1 is the one agent in each simulated world.
    """
    x, y, yaw = START_POSE
    pose = [x + LR * math.cos(yaw), y + LR * math.sin(yaw), yaw]
    return jnp.tile(jnp.asarray(pose, dtype=jnp.float32), (count, 1, 1))
