#!/bin/sh

cd ~/mjpg-streamer/mjpg-streamer-experimental/
export LD_LIBRARY_PATH=.
chromium-browser "http://127.0.0.1:8080/?action=stream" --start-fullscreen &
./mjpg_streamer -o "output_http.so -w ./www" -i "input_uvc.so -d /dev/video0 -r 1280x720 -f 60"