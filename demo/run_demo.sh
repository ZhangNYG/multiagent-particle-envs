cd /Users/Dylan/Documents/Git/tensorflow/
source ./bin/activate
export PYTHONPATH='/Users/Dylan/Documents/Git/maddpg'

osascript \
-e 'tell app "Terminal"' \
-e 'set bounds of front window to {725, 0, 1450, 1000}' \
-e 'end tell'

osascript -e 'tell app "Terminal"
    do script "PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/X11/bin && cd /Users/Dylan/Documents/Git/tensorflow/ && source ./bin/activate && export PYTHONPATH='/Users/Dylan/Documents/Git/maddpg' && cd /Users/Dylan/Documents/Git/maddpg/experiments/ && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ ' && python3 train.py --scenario rugby --load-dir /Users/Dylan/Documents/Git/multiagent-particle-envs/saved-files/human-ai/rugby/ --max-episode-len 10000 --display --delay 0.05"
end tell'

osascript \
-e 'tell app "Terminal"' \
-e 'set bounds of front window to {0, 0, 725, 1000}' \
-e 'end tell'

cd /Users/Dylan/Documents/Git/multiagent-particle-envs/bin/
echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ ' && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '  && echo -e '\ '
python3 interactive.py --scenario rugby --load-dir /Users/Dylan/Documents/Git/multiagent-particle-envs/saved-files/human-ai/rugby/ --display --max-episode-len 10000 --delay 0.05
