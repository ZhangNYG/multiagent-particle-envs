import numpy as np
import itertools
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def __init__(self):
        self.agent_size = 0.20
        self.one_hot_array = []
        self.colours = []
        self.game_over = None

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 5
        num_landmarks = 5
        num_obstacles = 0
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
            agent.size = self.agent_size
            agent.color = self.colours[i]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.color = self.colours[i]
            landmark.id = self.one_hot_array[2**i]
        # add obstacles
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.30
            obstacle.boundary = False
            obstacle.color = np.array([0.25, 0.25, 0.25])
        # make initial conditions
        self.reset_world(world)
        return world

    def assign_goals(self, i, agent):
        # assign each agent to a unique set of goals in one-hot encoding
        agent.hidden_goals = self.one_hot_array[2**i]

    def check_for_spawn_clash(self, world, entity):
        for other in world.agents:
            if other is entity: continue
            while self.is_collision(entity, other) is True:
                entity.state.p_pos = np.random.uniform(-0.7, +0.7, world.dim_p)
        for other in world.landmarks:
            if other is entity: continue
            while self.is_collision(entity, other) is True:
                entity.state.p_pos = np.random.uniform(-0.7, +0.7, world.dim_p)

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
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            self.assign_goals(i, agent)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.7, +0.7, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for i, obstacle in enumerate(world.obstacles):
            obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)
        for i, agent in enumerate(world.agents):
            self.check_for_spawn_clash(world, agent)
        for i, landmark in enumerate(world.landmarks):
            self.check_for_spawn_clash(world, landmark)

    def benchmark_data(self, agent, world):
        return 0

    def is_collision(self, entity1, entity2):
        if entity1 == entity2:
            return False
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = self.agent_size * 2
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each relevant landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            if l.id == agent.hidden_goals:
                rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
                if np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) < agent.size:
                    rew += 5
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 20
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
