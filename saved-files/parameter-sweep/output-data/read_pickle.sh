trap "exit" INT
for i in {0..9}
do
   for j in {0..9}
   do
     python3 read_pickle.py --filename ~/Git/multiagent-particle-envs/saved-files/parameter-sweep/benchmarked-files/sweep-$i$j --num-agents 3 --num-episodes 6
   done
done
