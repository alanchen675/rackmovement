# Copyright 2023 InstaDeep Ltd
#
# Licensed under the Creative Commons BY-NC-SA 4.0 License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Tuple

import chex
import jax
import jax.numpy as jnp
from chex import Array
from jumanji import specs
from jumanji.environments.routing.tsp.env import TSP
from jumanji.environments.routing.rack.types import State

from compass.environments.rack.utils import (
    DEPOT_IDX,
    compute_tour_length,
    generate_problem,
    get_coordinates_augmentations
)
from jumanji.types import TimeStep, termination, transition, restart

from compass.environments.poppy_env import PoppyEnv
from compass.environments.rack.types import Observation
from compass.environments.rack.inner_env import RackSystem
from compass.environments.rack.config import Config

class PoppyTSP(TSP, PoppyEnv):
    def __init__(self, args):
        self.inner_env = RackSystem(args)

    def step(self, state: State, action: chex.Numeric) -> Tuple[State, TimeStep]:
        """
        Run one timestep of the environment's dynamics. Unlike the Jumanji environment
        it assumes that the action taken is legal, which should be if the action masking
        is properly done and the agent respects it.

        Args:
            state: State object containing the dynamics of the environment.
            action: Array containing the index of the next position to visit.

        Returns:
            state: the next state of the environment.
            timestep: the timestep to be observed.
        """
        state = self._update_state(state, action)
        timestep = self._state_to_timestep(state, True)
        return state, timestep

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep]:
        """
        Resets the environment.

        Args:
            key: used to randomly generate the problem and the start node.

        Returns:
             state: State object corresponding to the new state of the environment.
             timestep: TimeStep object corresponding to the first timestep returned by the environment.
             extra: Not used.
        """
        env_key, start_key = random.split(key)
        seed = int(random.randint(env_key, (), 0, jnp.iinfo(jnp.int32).max))
        self.inner_env.reset(seed)
        self.num_cities = len(self.inner_env.rack_df)
        coords = self.inner_env.get_coordinates_jax()
        costs = self.inner_env.get_demand_jax()
        problem = jnp.hstack((coords, costs))
        start_node = generate_start_position(start_key, self.num_cities)
        return self.reset_from_state(problem, start_node)

    def reset_from_state(
            self, problem: Array, start_position: jnp.int32
    ) -> Tuple[State, TimeStep]:
        """
        Resets the environment from a given problem instance and start position.
        Args:
            problem: jax array (float32) of shape (problem_size, 2)
                the coordinates of each city
            start_position: int32
                the identifier (index) of the first city
        Returns:
            state: State object corresponding to the new state of the environment.
            timestep: TimeStep object corresponding to the first timestep returned by the
                environment.
        """
        state = State(
            coordinates=problem[:, :2],
            demands=problem[:, -1],
            capacity=self.inner_env.get_action_limit_jax(),
            position=jnp.array(-1, jnp.int32),
            visited_mask=jnp.zeros(self.num_cities, dtype=jnp.int8),
            trajectory=-1 * jnp.ones(self.num_cities, jnp.int32),
            num_visited=jnp.int32(0),
            key=jax.random.PRNGKey(0),
        )
        state = self._update_state(state, start_position)
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.
        Returns:
            Spec for the `Observation` whose fields are:
            - coordinates: BoundedArray (float) of shape (num_cities,).
            - position: DiscreteArray (num_values = num_cities) of shape ().
            - trajectory: BoundedArray (int32) of shape (num_cities,).
            - action_mask: BoundedArray (bool) of shape (num_cities,).
        """
        problem = specs.BoundedArray(
            shape=(self.num_cities, 2),
            minimum=0.0,
            maximum=1.0,
            dtype=float,
            name="problem",
        )
        start_position = specs.DiscreteArray(
            self.num_cities, dtype=jnp.int32, name="start_position"
        )
        position = specs.DiscreteArray(
            self.num_cities, dtype=jnp.int32, name="position"
        )
        capacity = specs.BoundedArray(
            shape=(), minimum=0.0, maximum=Config.action_limit_range[1], dtype=int, name="capacity"
        )
        trajectory = specs.BoundedArray(
            shape=(self.num_cities,),
            dtype=jnp.int32,
            minimum=-1,
            maximum=self.num_cities - 1,
            name="trajectory",
        )
        action_mask = specs.BoundedArray(
            shape=(self.num_cities,),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            problem=problem,
            start_position=start_position,
            position=position,
            capacity=capacity,
            trajectory=trajectory,
            action_mask=action_mask,
            is_done=specs.DiscreteArray(1, dtype=jnp.int8, name="is_done"),
        )

    def _update_state(self, state: State, next_node: jnp.int32) -> State:
        """
        Updates the state of the environment.

        Args:
            state: State object containing the dynamics of the environment.
            next_node: int, index of the next node to visit.

        Returns:
            state: State object corresponding to the new state of the environment.
        """
        self.inner_env.sub_step(next_node)

        return State(
            coordinates=state.coordinates,
            demands = state.demands,
            capacity=self.inner_env.get_action_limit_jax(),
            position=next_node,
            visited_mask=state.visited_mask.at[next_node].set(1),
            trajectory=state.trajectory.at[state.num_visited].set(next_node),
            num_visited=state.num_visited+1,
            key=state.key,
        )

    def _state_to_observation(self, state: State) -> Observation:
        """
        Converts a state into an observation.

        Args:
            state: State object containing the dynamics of the environment.

        Returns:
            observation: Observation object containing the observation of the environment.
        """
        return Observation(
            problem=state.coordinates,
            start_position=state.trajectory[0],
            position=state.position,
            capacity=state.capacity,
            trajectory=state.trajectory,
            action_mask=state.visited_mask,
            is_done=(state.num_visited == self.num_cities).astype(int),
        )

    def _state_to_timestep(self, state: State, is_valid: bool) -> TimeStep:
        """
        Checks if the state is terminal and converts it into a timestep. The episode terminates if
        there is no legal action to take (i.e., all cities have been visited) or if the last
        action was not valid.

        Args:
            state: State object containing the dynamics of the environment.
            is_valid: Boolean indicating whether the last action was valid.

        Returns:
            timestep: TimeStep object containing the timestep of the environment.
        """

        def make_termination_timestep(state: State) -> TimeStep:
            reward = jax.lax.cond(state.visited_mask.sum() == self.num_cities,
                                  #lambda _: -compute_tour_length(state.coordinates, state.trajectory),
                                  lambda _: self.inner_env.get_reward(),
                                  lambda _: jnp.array(-self.num_cities * jnp.sqrt(2), float),
                                  None)
            return termination(
                reward=reward,
                observation=self._state_to_observation(state),
            )

        def make_transition_timestep(state: State) -> TimeStep:
            return transition(
                reward=jnp.float32(0), observation=self._state_to_observation(state)
            )

        return jax.lax.cond(
            state.num_visited - 1 >= self.num_cities,
            make_termination_timestep,
            make_transition_timestep,
            state,
        )

    def render(self, state: State) -> Any:
        raise NotImplementedError

    def get_problem_size(self) -> int:
        return len(self.inner_env.rack_df)

    def get_min_start(self) -> int:
        return 0

    def get_max_start(self) -> int:
        return self.num_cities - 1

    def get_episode_horizon(self) -> int:
        return len(self.inner_env.rack_df)

    @staticmethod
    def generate_problem(*args, **kwargs) -> chex.Array:
        return generate_problem(*args, **kwargs)

    @staticmethod
    def get_augmentations(*args, **kwargs) -> chex.Array:
        return get_coordinates_augmentations(*args, **kwargs)

    @staticmethod
    def make_observation(*args, **kwargs) -> Observation:
        return Observation(*args, **kwargs)

    @staticmethod
    def is_reward_negative() -> bool:
        return True

    @staticmethod
    def get_reward_string() -> str:
        return "Rack movement cost and resource spread metrics"
