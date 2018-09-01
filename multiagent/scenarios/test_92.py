import numpy as np
import itertools
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from random import *


class Scenario(BaseScenario):

    def __init__(self):
        self.one_hot_array = []
        self.colours = []
        self.obstacle_count = 0
        self.occupied_landmarks = []
        self.num_agents = 3

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = self.num_agents
        num_landmarks = self.num_agents
        num_obstacles = 12
        # generate one-hot encoding for unique hidden goals
        self.one_hot_array = list(itertools.product([0, 1], repeat=num_landmarks))
        # generate colours for goal identification
        for _ in range(num_landmarks):
            self.colours.append(np.random.uniform(-1, +1, 3))
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.10
            agent.color = self.colours[i]
            agent.state.p_pos = [0.00, 0.00]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.color = self.colours[i]
            landmark.id = self.one_hot_array[2**i]
            self.occupied_landmarks.append(0)
            landmark.state.p_pos = [0.00, 0.00]
        # add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.40
            obstacle.boundary = False
            obstacle.color = np.array([0.25, 0.25, 0.25])
            # self.create_wall(world, obstacle, 10, -0.2, -1, -0.2, -0.2)
        # make initial conditions
        self.reset_world(world)
        return world

    def assign_goals(self, i, agent):
        # assign each agent to a unique set of goals in one-hot encoding
        agent.hidden_goals = self.one_hot_array[2**i]
        
    def sample_position(self):
        random_val = uniform(-0.80, 0.80)
        axis = randint(0, 1)  # to choose an axis
        if axis == 0: # on x-axis
            return [random_val, 0.00]
        elif axis == 1: # on y-axis
            return [0.00, random_val]

    def check_for_spawn_clash(self, world_entity_list, entity):
        other_entities = world_entity_list.copy()
        other_entities.remove(entity)
        for other in other_entities:
            while self.is_collision(entity, other, 0.1) is True:
                entity.state.p_pos = np.array(self.sample_position())

    def reset_world(self, world):
        # properties for agents
        for i, agent in enumerate(world.agents):
            pass
        # properties for landmarks
        for i, agent in enumerate(world.agents):
            pass
        # properties for obstacles
        for i, obstacle in enumerate(world.obstacles):
            pass
        # set initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array(self.sample_position())
            if i > 0:
                self.check_for_spawn_clash(world.agents, agent)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            self.assign_goals(i, agent)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array(self.sample_position())
            if i > 0:
                self.check_for_spawn_clash(world.landmarks, landmark)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, obstacle in enumerate(world.obstacles):
            if i > 3:
                obstacle.size = 0.2
            if i > 7:
                obstacle.size = 0.1
            positions = [[-0.50, -0.50], [-0.50, 0.50], [0.50, -0.50], [0.50, 0.50],
                         [-0.30, -0.30], [-0.30, 0.30], [0.30, -0.30], [0.30, 0.30],
                         [-0.20, -0.20], [-0.20, 0.20], [0.20, -0.20], [0.20, 0.20]]
            obstacle.state.p_pos = np.array(positions[i])
            obstacle.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        min_dists = 0
        for i, l in enumerate(world.landmarks):
            if l.id == agent.hidden_goals:
                rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                if self.is_collision(l, agent):
                    rew += 0
                    self.occupied_landmarks[i] = 1
                else:
                    self.occupied_landmarks[i] = 0
        # Agent rewarded if all agents occupy their goals
        if sum(self.occupied_landmarks) == self.num_agents:
            rew += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 7
                    collisions += 1
            for o in world.obstacles:
                if self.is_collision(o, agent):
                    rew -= 0
        return (rew, collisions, min_dists, self.occupied_landmarks)

    def is_collision(self, agent1, agent2, gap=None):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        if gap is None:
            dist_min = agent1.size + agent2.size
        else:
            dist_min = 2*gap
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each relevant landmark, penalized for collisions
        rew = 0
        for i, l in enumerate(world.landmarks):
            if l.id == agent.hidden_goals:
                rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                if self.is_collision(l, agent):
                    rew += 0
                    self.occupied_landmarks[i] = 1
                else:
                    self.occupied_landmarks[i] = 0
        # Agent rewarded if all agents occupy their goals
        if sum(self.occupied_landmarks) == self.num_agents:
            rew += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 7
            for o in world.obstacles:
                if self.is_collision(o, agent):
                    rew -= 0

        # agents are penalized for exiting the screen, so that they can converge faster
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
 
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        for entity in world.obstacles:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        for entity in world.obstacles:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # get positions of all other agents relative to their goals
        other_goals = []
        for other in world.agents:
            if other is agent: continue
            for l in world.landmarks:
                if l.id == other.hidden_goals:
                    other_goals.append(l.state.p_pos - other.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm + other_goals)

