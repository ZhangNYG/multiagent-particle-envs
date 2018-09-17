trap "exit" INT
for i in {0..9}
do
   for j in {0..9}
   do
     python3 /home/dylank/Git/maddpg/experiments/train.py --scenario 3a_parameter_sweep --save-rate 500 --max-episode-len 50 --num-episodes 5000 --indv-rew 0 --coop-rew $i --crash-pun $j --save-dir /home/dylank/Git/multiagent-particle-envs/saved-files/parameter-sweep/saved-models/sweep-$i$j/ --plots-dir /home/dylank/Git/multiagent-particle-envs/saved-files/parameter-sweep/saved-plots/sweep-$i$j/ --exp-name "sweep-$i$j"
   done
done
