# Unused: kept for reference

Nothing in this folder is used by the driver (`race_ws/src/team_driver`) or by
training (`training/`). It is kept so the work is not lost, not to be run.

| What | Why it is here |
| --- | --- |
| `residual/residual_network.py` | The residual network: our first learned method. Built and tested, never trained to a model good enough to race. |
| `models/lookahead_old.npz.bak` | An earlier lookahead policy (training run `20261012-014600`), trained before the speed table existed. Replaced on 2026-10-13. |

## The residual network

**The idea.** Pure pursuit decides steering and speed as usual, and a small
neural network adds a correction on top. The correction is capped (`LIMITS`:
at most 0.10 rad of steering and 1.5 m/s of speed), so the worst a bad network
can do is nudge a controller that already laps. This is called *residual
policy learning*.

**The network.** 14 inputs (`FEATURE_NAMES`: speed, yaw rate, last steering,
offset and heading error from the racing line, pure pursuit's own command,
free distance ahead, and the line's curvature 0.6 to 6 m ahead), two hidden
layers of 128 tanh neurons, 2 outputs squashed into -1..1 by tanh and scaled by
`LIMITS`.

**Why it is not used.** The pipeline worked: numpy and JAX computed the same
inputs and outputs, and weights saved and loaded correctly. But we never
trained it to a model that beat plain pure pursuit, and the lap time came
instead from the racing line, the PPO-tuned speed table and the lookahead
policy. So on 2026-10-17 it was moved here, and the driver and training lost
every trace of it: `residual.py` became `policies.py`, and
`train.py --method residual`, `evaluate.py --weights` and the driver's
`use_residual` / `residual_weights` parameters were removed.

## How it was wired in

To see or run the fully wired version, use git: commit `f9e6c43` has it, with
the network in `race_ws/src/team_driver/team_driver/residual.py`.

```sh
git show f9e6c43:race_ws/src/team_driver/team_driver/residual.py
git checkout f9e6c43 -- race_ws/src/team_driver training   # (overwrites your current files)
```

The pieces that connected it, as they were just before removal:

**Driver** (`driver.py`). `load_policy()` loaded `models/residual.npz` when
there was no `lookahead.npz`:

```python
path = self.policy_file('residual')
if path:
    self.weights, self.limits, trained_with = load_weights(path)
    self.get_logger().info(f'residual network loaded from {path}, limits {self.limits}')
    self.warn_if_retuned(trained_with)
    return
```

and `plan()` added the correction after pure pursuit:

```python
# 3. The learned correction, if a residual network is loaded: describe
#    the situation as numbers, ask the network, add its capped answer.
if self.weights is not None:
    x = features(self.path, self.heading, self.curvature, car, self.yaw,
                 self.speed, self.yaw_rate, self.last_steer,
                 steering, speed, nearest, ahead, self.params)
    squashed = np.tanh(actor(self.weights, x))         # each output in -1..1
    steering, speed = apply_residual(steering, speed, squashed, self.limits, self.params)
```

**Training simulator** (`training/sim.py`). The controller also built the
network's inputs for every car:

```python
def make_controller(path, speeds=None):
    """Return controller(observation, last_steer) for a batch of cars.

    For every car it runs the driver's pure pursuit and builds the residual
    network's inputs, exactly as driver.py does, and returns
    (base_steer, base_speed, features), each with one row per car.
    `speeds`: a speed table to drive by; None = whatever the driver would use.

    Parameters
    ----------
    path : numpy.ndarray, shape (points, 2)
        Racing-line points [m].
    speeds : array of shape (points,), optional
        Speed table [m/s]; scaled by DEFAULTS['table_scale'] before use.

    Returns
    -------
    callable
        controller(observation, last_steer) -> (base_steer (cars,) [rad],
        base_speed (cars,) [m/s], features (cars, NUM_FEATURES)).
    """
    heading, curvature = path_geometry(path)
    speeds = scaled(driver_speed_table(path, curvature) if speeds is None else speeds)
    path, heading, curvature = jnp.asarray(path), jnp.asarray(heading), jnp.asarray(curvature)
    first, last, beams = scan_slice()
    angles = jnp.asarray(first + np.arange(beams) * (last - first) / (beams - 1))

    def one_car(car, yaw, speed, yaw_rate, ranges, last_steer):
        """Pure pursuit and the residual's inputs for one car."""
        steer, v, nearest, ahead = pure_pursuit(
            path, car, yaw, speed, ranges, angles, DEFAULTS, xp=jnp, speed_table=speeds)
        x = features(path, heading, curvature, car, yaw, speed, yaw_rate, last_steer,
                     steer, v, nearest, ahead, DEFAULTS, xp=jnp)
        return steer, v, x

    def controller(observation, last_steer):
        """Run one_car on every car at once (jax.vmap)."""
        car, yaw, speed, yaw_rate = odometry(observation)
        ranges = preprocess(observation.scans[:, 0], DEFAULTS['max_range'], xp=jnp)
        return jax.vmap(one_car)(car, yaw, speed, yaw_rate, ranges, last_steer)

    return controller
```

**Training** (`training/train.py`). Its PPO settings, reward cost and starting
weights:

```python
'residual': dict(steps=50e6, epochs=4, minibatches=8, lr=3e-4, lr_decay=False,
                 gamma=0.995, gae_lambda=0.95, entropy=0.0, value_coef=0.5, target_kl=None),
```

```python
RESIDUAL_COST = 0.002      # residual only: per step, times the squared size of the correction
```

```python
return {
    # A tiny last layer means the corrections start out almost zero, so
    # training begins from plain pure pursuit.
    'actor': network(k_actor, NUM_FEATURES, hidden, 2, 0.01),
    'critic': network(k_critic, NUM_FEATURES, hidden, 1, 1.0),
    'log_std': jnp.full(2, -1.0),                   # noise 0.37 before tanh
}
```

one simulation step with the network in charge:

```python
def drive_residual(params, carry, key, window):
    """Drive every car one step, with the residual network correcting pure pursuit.

    Pure pursuit gives a command and the NUM_FEATURES (14) inputs; the
    network's output plus noise is squashed into -1..1 by tanh, and
    residual.apply_residual scales it by LIMITS (at most 0.10 rad of
    steering and 1.5 m/s of speed) and adds it on. The reward is the
    simulator's minus RESIDUAL_COST x the squared size of the
    correction. (This method has never been trained for real.)

    Parameters
    ----------
    params, carry, key, window
        As in drive_lookahead().

    Returns
    -------
    carry : tuple
        The new 7-item carry (items 5 and 6 unchanged).
    sample : Sample
        This step's data, one row per car.
    """
    k_noise, k_sim = jax.random.split(key)
    base_steer, base_speed, x = controller(carry[1], carry[2])
    mean = actor(params['actor'], x, xp=jnp)
    action = mean + jnp.exp(params['log_std']) * jax.random.normal(k_noise, mean.shape)
    squashed = jnp.tanh(action)
    steer, speed = jax.vmap(
        lambda s, v, a: apply_residual(s, v, a, limits, DEFAULTS, xp=jnp)
    )(base_steer, base_speed, squashed)
    out, core = step_simulator(carry, steer, speed, k_sim)
    reward = out.rewards[:, 0] - RESIDUAL_COST * jnp.sum(squashed ** 2, axis=1)
    _, _, x_end = controller(out.transition_observation, steer)
    log_prob = jnp.sum(gaussian_log_prob(action, mean, params['log_std'], xp=jnp), axis=-1)
    return core + carry[5:], sample(params, x, x_end, action, log_prob, reward, out, squashed)
```

and its branch of the loss (the network gives a mean for each of its two
outputs, so the log-probabilities are summed):

```python
else:
    mean = actor(params['actor'], mb.inputs, xp=jnp)
    log_prob = jnp.sum(gaussian_log_prob(mb.action, mean, params['log_std'], xp=jnp), axis=-1)
    policy_loss, kl = ppo_policy_loss(log_prob, mb.log_prob, mb.advantage,
                                      params['log_std'], args.clip, args.entropy, xp=jnp)
```
