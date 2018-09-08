trap "exit" INT
for i in {0..9}
do
   for j in {0..9}
   do
     python3 train.py --scenario 3a_parameter_sweep --save-rate 10 --max-episode-len 10 --num-episodes 20 --indv-rew 0 --coop-rew $i --crash-pun $j --save-dir ./multiagent-particle-envs/saved-files/parameter-sweep/saved-models/sweep-$i$j/ --plots-dir ./multiagent-particle-envs/saved-files/parameter-sweep/saved-plots/sweep-$i$j/ --exp-name "sweep-$i$j"
   done
done
