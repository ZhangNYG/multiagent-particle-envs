import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
import sys
from colorama import init
init(strip=not sys.stdout.isatty())  # strip colors if stdout is redirected
from termcolor import cprint
from pyfiglet import figlet_format
from threading import Timer


class Scenario(BaseScenario):

    def __init__(self):
        self.score = 0
        self.t = Timer(40, self.timeout)  # duration is in seconds
        self.t.start()
        self.game_over = False
        self.obstacle_vel = []

    def timeout(self):
        cprint(figlet_format('Game Over!\nFinal score: ' + str(self.score)), 'red')
        self.game_over = True

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 6
        world.num_agents = num_agents
        num_adversaries = num_agents - 1
        num_landmarks = 1
        num_obstacles = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.08 if i < num_adversaries else 0.10
            agent.max_speed = 0.9 if i < num_adversaries else None
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
        # add obstacle
        world.obstacles = [Landmark() for i in range(num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = True
            obstacle.size = 0.30
        # make initial conditions
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for obstacle in world.obstacles:
            obstacle.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            self.obstacle_vel = np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = self.obstacle_vel
            obstacle.state.c = np.zeros(world.dim_c)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set properties for agents
        world.agents[-1].color = np.array([0.35, 0.65, 0.85])
        for i in range(0, world.num_agents - 1):
            world.agents[i].color = np.array([0.85, 0.85, 0.85])
        # set random initial states
        for agent in world.agents:
            pass
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = np.array([0.35, 0.35, 0.35])
        for obstacle in world.obstacles:
            obstacle.color = np.array([0.70, 0.50, 0.95])

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # restrict movement of agent to inside screen
    def restrict_movement(self, agent, world):
        x_pos = agent.state.p_pos[0]
        y_pos = agent.state.p_pos[1]

        if agent.name == 'obstacle 0':
            if x_pos < -1.0: x_pos = +1.0
            if y_pos < -1.0: y_pos = +1.0
            if x_pos > +1.0: x_pos = -1.0
            if y_pos > +1.0: y_pos = -1.0
        else:
            if abs(x_pos) > 1.0 or abs(y_pos) > 1.0:
                pass  # agent.state.p_vel = np.zeros(world.dim_p)
            if x_pos < -1.0: x_pos = -1.0
            if y_pos < -1.0: y_pos = -1.0
            if x_pos > +1.0: x_pos = +1.0
            if y_pos > +1.0: y_pos = +1.0

        agent.state.p_pos[0] = x_pos
        agent.state.p_pos[1] = y_pos

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Rewarded based on how close the attacker agent is to the landmark
        l = world.landmarks[0]
        rew = -np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        if np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) < agent.size:
            self.score += 1
            cprint(figlet_format('Score: ' + str(self.score)), 'blue')
            self.reset_world(world)
        self.restrict_movement(agent, world)
        return rew

    def adversary_reward(self, agent, world):
        # Punished based on how close attacker agent is to the defender
        rew = -np.sqrt(np.sum(np.square(agent.state.p_pos - world.agents[-1].state.p_pos)))
        return rew

    def observation(self, agent, world):
        for obstacle in world.obstacles:
            obstacle.state.p_vel = self.obstacle_vel
            self.restrict_movement(obstacle, world)
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate(entity_pos + other_pos)

