#!/usr/bin/bash

echo "-- killing nginx --"
sudo systemctl stop nginx
sudo systemctl disable nginx

echo "-- killing dce-viewer service --"
sudo systemctl stop dce-viewer
sudo systemctl disable dce-viewer

echo "-- removing server files --"
sudo rm /mnt/home/russell.burdt/dce-viewer.info
sudo rm /etc/nginx/sites-enabled/dce-viewer
sudo rm /etc/nginx/sites-available/dce-viewer
sudo rm /etc/systemd/system/dce-viewer.service
sudo rm /tmp/bokeh.access.log
sudo rm /tmp/bokeh.error.log
