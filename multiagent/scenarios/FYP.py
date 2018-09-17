import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 42
        world.num_agents = num_agents
        num_adversaries = num_agents - 1
        # add goal locations
        world.goals = [[-0.7, -0.4],
                       [-0.7, -0.3],
                       [-0.7, -0.2],
                       [-0.7, -0.1],
                       [-0.7, 0],
                       [-0.7, 0.1],
                       [-0.7, 0.2],
                       [-0.7, 0.3],

                       [-0.6, -0.05],
                       [-0.6, 0.3],
                       [-0.5, -0.05],
                       [-0.5, 0.3],
                       [-0.4, -0.05],
                       [-0.4, 0.3],

                       [0, -0.4],
                       [0, -0.3],
                       [0, -0.2],
                       [0, -0.1],

                       [-0.05, 0],
                       [0.05, 0],
                       [-0.1, 0.1],
                       [0.1, 0.1],
                       [-0.15, 0.2],
                       [0.15, 0.2],
                       [-0.2, 0.3],
                       [0.2, 0.3],

                       [0.4, -0.4],
                       [0.4, -0.3],
                       [0.4, -0.2],
                       [0.4, -0.1],
                       [0.4, 0],
                       [0.4, 0.1],
                       [0.4, 0.2],
                       [0.4, 0.3],

                       [0.5, 0.3],
                       [0.6, 0.28],
                       [0.68, 0.2],
                       [0.7, 0.1],
                       [0.68, 0],
                       [0.6, -0.08],
                       [0.5, -0.1],

                       [0, 0]]
        # add landmarks
        world.landmarks = []
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.04 if i < num_adversaries else 0.10
            agent.goal = np.array(world.goals[i])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set properties for agents
        world.agents[-1].color = np.array([0.35, 0.35, 0.85])
        for i in range(0, world.num_agents - 1):
            world.agents[i].color = np.array([0.35, 0.35, 0.35])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

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
        # No reward given to human-controlled agent
        return 0

    def adversary_reward(self, agent, world):
        # Punished based on how close attacker agent is to the defender
        rew = -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal)))
        if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal))) < agent.size:
            rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for [x, y] in world.goals:
            entity_pos.append(np.array([x, y]) - agent.state.p_pos)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate(entity_pos + other_pos)

