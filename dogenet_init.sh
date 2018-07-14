#!/usr/bin/env bash
### BEGIN INIT INFO
# Provides:          dogenet_init.sh
# Required-Start:    $all
# Required-Stop:     $all
# Default-Start:     3   5
# Default-Stop:      0   1   6
# Short-Description: Twitter bot for a doge classifier
### END INIT INFO


# to run the script at startup on runlevel 3
# this should be added to rc.d on systemctl systems
python3 twitter.py
