PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/X11/bin
cd /Users/Dylan/Documents/Git/tensorflow/
source ./bin/activate
export PYTHONPATH='/Users/Dylan/Documents/Git/maddpg'

# RUN TRAIN.PY #

cd /Users/Dylan/Documents/Git/maddpg/experiments/
python3 train.py --scenario rugby --load-dir /Users/Dylan/Documents/Git/multiagent-particle-envs/saved-files/human-ai/rugby/ --max-episode-len 10000 --display --delay 0.05


# OPEN A NEW TERMINAL WINDOW TO RUN INTERACTIVE.PY #

osascript \
-e 'tell app "Terminal"' \
-e 'set bounds of front window to {0, 0, 725, 1000}' \
-e 'do script "PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/X11/bin && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ ' && cd /Users/Dylan/Documents/Git/tensorflow/ && source ./bin/activate && export PYTHONPATH='/Users/Dylan/Documents/Git/maddpg' && cd /Users/Dylan/Documents/Git/multiagent-particle-envs/bin/ && python3 interactive.py --scenario rugby --load-dir /Users/Dylan/Documents/Git/multiagent-particle-envs/saved-files/human-ai/rugby/ --display --max-episode-len 10000 --delay 0.05"' \
-e 'set bounds of front window to {725, 0, 1450, 1000}' \
-e 'end tell'
