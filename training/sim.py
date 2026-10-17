"""Training simulator shared by train.py, evaluate.py, speed_table.py and test_parity.py; it is imported by them, not run directly.

The training simulator: f1tenth_gym's GPU (JAX) simulator, set up to
match the judged race as closely as it can, plus the driver's own
controller running inside it.

JAX words used below: jax.vmap turns a function written for one car into
one that runs on a whole batch of cars at once; jax.jit compiles a
function once so later calls run fast on the GPU.

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
      (docs/06-rules.md rule 35 says scans come at "roughly 40 Hz"; that
      measurement disagrees. When the real driver runs, the `gap` in its
      reaction-delay log line, from Driver.measure_delay, shows the real
      figure: 1 = it answered a scan every step.)

Reaction delay (see `delay_commands`): in ROS a command reaches the physics
a few steps after the scan it answers. That matters a lot on icra26: in the
race's own engine, pure pursuit crashes in the infield hairpin with a delay
of 0-1 steps but laps (17.2 s) with 2-3 steps (measured 2026-10-11). The
real delay varies with the computer, so every episode gets a random delay of
0..MAX_DELAY steps and the network has to cope with all of them.
MAX_DELAY (3) is only the default: train.py and evaluate.py set it from
--max-delay, and the runs behind the shipped models used 8 and 12 (speed
table) and 10 (lookahead).

Not modelled: a delay that changes during an episode, and skipped scans
(the real driver missing a scan, so one command is held for several steps).
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
from team_driver.policies import (  # noqa: E402
    lookahead_inputs, path_geometry, speed_table_inputs)

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
    """Build the racing line with the same code and settings as the driver.

    Returns
    -------
    numpy.ndarray, shape (points, 2)
        The racing line's (x, y) points in map coordinates [m], about
        0.2 m apart, going once round the lap (380 points on icra26).
    """
    centre = np.loadtxt(CENTRELINE_CSV, delimiter=',', comments='#')
    clearance = load_clearance_map(str(MAP_YAML))
    return build_raceline(centre, clearance, CAR_HALF_WIDTH + WALL_GAP, ITERS)


def driver_speed_table(path, curvature):
    """Return the untuned speed table the driver would build from DEFAULTS.

    This is the physics-based table (raceline.speed_table, from mu and
    a_brake) that driver.py races when no tuned speed_table.npz is
    accepted, as a jax array; or None when DEFAULTS['use_speed_table'] is
    off, exactly as driver.py decides.

    Parameters
    ----------
    path : numpy.ndarray, shape (points, 2)
        Racing-line points [m].
    curvature : numpy.ndarray, shape (points,)
        Curvature at each point [1/m].

    Returns
    -------
    jax.Array of shape (points,), or None
        Target speed at each point [m/s].
    """
    if not DEFAULTS['use_speed_table']:
        return None
    return jnp.asarray(speed_table(path, curvature, DEFAULTS))


def scaled(speeds):
    """Multiply a speed table by DEFAULTS['table_scale'].

    The table slowed down by DEFAULTS['table_scale'], as driver.py does,
    or None if there is no table.

    Parameters
    ----------
    speeds : array of shape (points,), or None
        Target speeds [m/s].

    Returns
    -------
    jax.Array of shape (points,), or None
        The scaled speeds [m/s].
    """
    return None if speeds is None else jnp.asarray(speeds) * DEFAULTS['table_scale']


def scan_slice():
    """Pick out the LiDAR beams the driver actually uses.

    The race's beams that the driver's "ahead" cone can see, plus two
    spare beams each side. The beam angles are exactly the race's.

    Returns
    -------
    first : float
        Angle of the first beam [rad].
    last : float
        Angle of the last beam [rad].
    beams : int
        Number of beams from first to last.
    """
    increment = (SCAN_MAX_DEG - SCAN_MIN_DEG) / (SCAN_BEAMS - 1)
    half = DEFAULTS['ahead_angle']
    first = math.floor((-half - SCAN_MIN_DEG) / increment) - 2
    last = math.ceil((half - SCAN_MIN_DEG) / increment) + 2
    return (math.radians(SCAN_MIN_DEG + first * increment),
            math.radians(SCAN_MIN_DEG + last * increment),
            last - first + 1)


def make_simulator(device, training, episode_steps=3000, collision_penalty=10.0):
    """Create one simulator for a whole batch of cars on icra26.

    training=True: a crash ends the episode, episodes last at most
    `episode_steps`, and the built-in reward is the metres of progress along
    the centreline each step, minus `collision_penalty` on a crash.
    training=False (evaluation): crashes do not end anything, like the race.

    Parameters
    ----------
    device : jax.Device
        The GPU or CPU to simulate on.
    training : bool
        Training rules (True) or race rules (False), as above.
    episode_steps : int, optional
        Time limit per training episode [steps of 0.01 s]; 3000 = 30 s.
    collision_penalty : float, optional
        Reward taken away on touching a wall [metres of progress].

    Returns
    -------
    JaxSimulator
        f1tenth_gym's batched simulator (reward mode PROGRESS).
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
    """Return controller(observation) for a batch of cars: plain pure pursuit.

    For every car it runs pure_pursuit() exactly as driver.py does when no
    lookahead policy is loaded (lookahead = lookahead_gain * speed), and
    returns (steering, speed), one row per car. evaluate.py uses it when
    no --lookahead file is given.

    Parameters
    ----------
    path : numpy.ndarray, shape (points, 2)
        Racing-line points [m].
    speeds : array of shape (points,), optional
        Speed table [m/s]; scaled by DEFAULTS['table_scale'] before use.
        None = whatever the driver would use.

    Returns
    -------
    callable
        controller(observation) -> (steering (cars,) [rad], speed (cars,) [m/s]).
    """
    _, curvature = path_geometry(path)
    speeds = scaled(driver_speed_table(path, curvature) if speeds is None else speeds)
    path = jnp.asarray(path)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def one_car(car, yaw, speed, ranges):
        """Pure pursuit for one car."""
        steer, v, _, _ = pure_pursuit(
            path, car, yaw, speed, ranges, angles, DEFAULTS, xp=jnp, speed_table=speeds)
        return steer, v

    def controller(observation):
        """Run one_car on every car at once (jax.vmap)."""
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(one_car)(car, yaw, speed, ranges)

    return controller


def make_lookahead_controller(path, speeds=None):
    """Build the two batched functions the dynamic-lookahead policy needs.

    For the dynamic-lookahead policy. Returns two functions, each taking
    a batch of cars:
        inputs(observation)        -> (cars, 5) the policy's inputs s
        drive(observation, L)      -> (steering, speed) from pure pursuit
                                      aiming L metres ahead (one L per car)
    `speeds`: a speed table to drive by; None = whatever the driver would use.

    Parameters
    ----------
    path : numpy.ndarray, shape (points, 2)
        Racing-line points [m].
    speeds : array of shape (points,), optional
        Speed table [m/s]; scaled by DEFAULTS['table_scale'] before use.

    Returns
    -------
    inputs : callable
        As above; s = (speed [m/s], |k0|, |k1|, |k2|, dk), the curvatures
        of the line ahead.
    drive : callable
        As above; steering [rad] and speed [m/s], each shape (cars,).
    """
    _, curvature = path_geometry(path)
    speeds = scaled(driver_speed_table(path, curvature) if speeds is None else speeds)
    path, curvature = jnp.asarray(path), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def inputs_one_car(car, speed):
        """The 5 lookahead inputs for one car: position (2,) [m], speed [m/s] -> (5,)."""
        nearest = jnp.argmin(jnp.linalg.norm(path - car, axis=1))
        return lookahead_inputs(curvature, nearest, speed, xp=jnp)

    def drive_one_car(car, yaw, speed, ranges, distance):
        """Pure pursuit for one car aiming `distance` [m] ahead -> (steering [rad], speed [m/s])."""
        steer, v, _, _ = pure_pursuit(path, car, yaw, speed, ranges, angles, DEFAULTS,
                                      xp=jnp, lookahead=distance, speed_table=speeds)
        return steer, v

    def inputs(observation):
        """Run inputs_one_car on every car at once (jax.vmap) -> (cars, 5)."""
        car, _, speed, _ = odometry(observation)
        return jax.vmap(inputs_one_car)(car, speed)

    def drive(observation, distance):
        """Run drive_one_car on every car at once; distance has shape (cars,) [m]."""
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(drive_one_car)(car, yaw, speed, ranges, distance)

    return inputs, drive


def make_speed_table_controller(path):
    """Build the two batched functions that training the speed table needs.

    For training the speed table. Returns two functions, each taking a
    batch of cars:
        observe(observation) -> (critic inputs (cars, 7), nearest path point (cars,))
        steer(observation)   -> steering from pure pursuit with its own
                                lookahead rule (the speed is the policy's choice)

    Parameters
    ----------
    path : numpy.ndarray, shape (points, 2)
        Racing-line points [m].

    Returns
    -------
    observe : callable
        As above. The 7 critic inputs are policies.speed_table_inputs:
        v, |k0|, |k1|, |k2|, dk, and the car's place on the lap as
        sin/cos of an angle.
    steer : callable
        As above; steering [rad], shape (cars,). Pure pursuit's own speed
        is thrown away.
    """
    _, curvature = path_geometry(path)
    path, curvature = jnp.asarray(path), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def observe_one_car(car, speed):
        """Critic inputs (7,) and nearest racing-line point (int) for one car."""
        nearest = jnp.argmin(jnp.linalg.norm(path - car, axis=1))
        return speed_table_inputs(curvature, nearest, speed, xp=jnp), nearest

    def steer_one_car(car, yaw, speed, ranges):
        """Pure pursuit's steering [rad] for one car, with its own lookahead rule."""
        steer, _, _, _ = pure_pursuit(path, car, yaw, speed, ranges, angles, DEFAULTS, xp=jnp)
        return steer

    def observe(observation):
        """Run observe_one_car on every car at once -> ((cars, 7), (cars,))."""
        car, _, speed, _ = odometry(observation)
        return jax.vmap(observe_one_car)(car, speed)

    def steer(observation):
        """Run steer_one_car on every car at once -> steering (cars,) [rad]."""
        car, yaw, speed, _ = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(steer_one_car)(car, yaw, speed, ranges)

    return observe, steer


def odometry(observation):
    """Read off what /ego_racecar/odom would report for each car.

    That is: rear-axle position, heading, speed and yaw rate. The simulator
    itself tracks the centre of mass, which is LR metres ahead of the rear
    axle.

    Parameters
    ----------
    observation : f1tenth_gym observation
        The simulator's observation of a batch of cars (one agent each).

    Returns
    -------
    car : jax.Array, shape (cars, 2)
        Rear-axle position (x, y) in map coordinates [m].
    yaw : jax.Array, shape (cars,)
        Heading [rad].
    speed : jax.Array, shape (cars,)
        Speed [m/s], always positive (its size only).
    yaw_rate : jax.Array, shape (cars,)
        How fast the heading turns [rad/s].
    """
    state = observation.standard_state[:, 0]   # [x, y, steer, speed, yaw, yaw rate, slip]
    yaw = state[:, 4]
    car = state[:, :2] - LR * jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=1)
    return car, yaw, jnp.abs(state[:, 3]), state[:, 5]


def delay_commands(queue, command, delay):
    """Simulate reaction delay with a queue of recent commands.

    `queue` holds each car's last MAX_DELAY + 1 commands, newest first,
    shape (cars, MAX_DELAY + 1, 2). The new command goes in at the front
    and each car gets the one from `delay` steps ago. Delay 0 applies the
    new command at once; delay 2 applies the one chosen 2 steps (0.02 s)
    ago. The queue starts as zeros, so for its first `delay` steps a car
    gets (steering 0, speed 0).

    Parameters
    ----------
    queue : jax.Array, shape (cars, MAX_DELAY + 1, 2)
        Recent (steering [rad], speed [m/s]) commands, newest first.
    command : jax.Array, shape (cars, 2)
        This step's (steering, speed) command for each car.
    delay : jax.Array, shape (cars,), int
        Each car's delay [steps], 0..MAX_DELAY.

    Returns
    -------
    queue : jax.Array, shape (cars, MAX_DELAY + 1, 2)
        The queue with `command` added at the front and the oldest dropped.
    applied : jax.Array, shape (cars, 2)
        The command each car's simulator should use this step.
    """
    queue = jnp.concatenate([command[:, None, :], queue[:, :-1]], axis=1)
    return queue, queue[jnp.arange(queue.shape[0]), delay]


def start_poses(count):
    """Give the race's grid slot as the simulator wants it: centre of mass.

    Parameters
    ----------
    count : int
        Number of cars.

    Returns
    -------
    jax.Array, shape (count, 1, 3), float32
        (x [m], y [m], yaw [rad]) of the centre of mass, LR ahead of the
        rear-axle START_POSE, the same for every car (the 1 is the one
        agent per simulated world).
    """
    x, y, yaw = START_POSE
    pose = [x + LR * math.cos(yaw), y + LR * math.sin(yaw), yaw]
    return jnp.tile(jnp.asarray(pose, dtype=jnp.float32), (count, 1, 1))
