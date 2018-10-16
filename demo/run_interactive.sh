PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/X11/bin
cd /Users/Dylan/Documents/Git/tensorflow/
source ./bin/activate
export PYTHONPATH='/Users/Dylan/Documents/Git/maddpg'

# RUN INTERACTIVE.PY #

cd /Users/Dylan/Documents/Git/multiagent-particle-envs/bin/
python3 interactive.py --scenario rugby --load-dir /Users/Dylan/Documents/Git/multiagent-particle-envs/saved-files/human-ai/rugby/ --max-episode-len 10000 --display --delay 0.05
