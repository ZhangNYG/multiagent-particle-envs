import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser("Reading RL output data from .pkl files")
    parser.add_argument("--filename", type=str, default=None, help="filename of .pkl file")
    parser.add_argument("--num-agents", type=int, default=3, help="number of agents")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()

objects = []
filename = arglist.filename
with (open(filename + ".pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

collisions = 0
sum_goals = 0
num_agents = arglist.num_agents
num_episodes = arglist.num_episodes

for i in range(len(objects[0])):
    for j in range(len(objects[0][i])):
        for k in range(len(objects[0][i][j])):
            for l in range(len((objects[0][i][j][k])) - 1):
                collisions += objects[0][i][j][k][l][0]

            if objects[0][i][j][k][-1] is True:
                for l in range(len((objects[0][i][j][k])) - 1):
                    sum_goals += objects[0][i][j][k][l][1]

average_collisions = collisions/num_episodes/num_agents
average_goals_reached = sum_goals/num_episodes/num_agents
print(average_collisions, average_goals_reached)

