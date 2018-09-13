trap "exit" INT
for i in {0..9}
do
   for j in {0..9}
   do
     python3 /home/dylank/Git/maddpg/experiments/train.py --scenario 3a_parameter_sweep --max-episode-len 50 --num-episodes 100 --indv-rew 0 --coop-rew $i --crash-pun $j --benchmark --benchmark-iters 5000 --benchmark-dir /home/dylank/Git/multiagent-particle-envs/saved-files/parameter-sweep/benchmarked-files/ --plots-dir /home/dylank/Git/multiagent-particle-envs/saved-files/parameter-sweep/saved-plots/ --exp-name "sweep-$i$j" --load-dir /home/dylank/Git/multiagent-particle-envs/saved-files/parameter-sweep/saved-models/sweep-$i$j/
   done
done
