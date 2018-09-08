import pickle

objects = []
filename = "test"
with (open(filename + ".pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

for i in range(len(objects[0])):
    for j in range(len(objects[0][i])):
        for k in range(len(objects[0][i][j])):
            print(objects[0][i][j][k])
