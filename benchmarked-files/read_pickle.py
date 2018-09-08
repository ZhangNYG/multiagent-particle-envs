import pickle

objects = []
filename = "test"
with (open(filename + ".pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

collisions = 0
sum_goals = 0
num_episodes = 10
num_agents = 3

for i in range(len(objects[0])):
    for j in range(len(objects[0][i])):
        for k in range(len(objects[0][i][j])):
            for l in range(len((objects[0][i][j][k])) - 1):
                collisions += objects[0][i][j][k][l][0]

            if objects[0][i][j][k][num_agents] is True:
                for l in range(len((objects[0][i][j][k])) - 1):
                    sum_goals += objects[0][i][j][k][l][1]

average_collisions = collisions/num_episodes/num_agents
average_goals_reached = sum_goals/num_episodes/num_agents
print(average_collisions, average_goals_reached)
