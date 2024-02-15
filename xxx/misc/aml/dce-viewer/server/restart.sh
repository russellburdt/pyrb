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

echo "-- copying server files --"
sudo cp /mnt/home/russell.burdt/Lytx.AML.RussellB/aml/dce-viewer/server/dce-viewer.info /mnt/home/russell.burdt
sudo cp /mnt/home/russell.burdt/Lytx.AML.RussellB/aml/dce-viewer/server/dce-viewer /etc/nginx/sites-available
sudo ln -s /etc/nginx/sites-available/dce-viewer /etc/nginx/sites-enabled/
sudo cp /mnt/home/russell.burdt/Lytx.AML.RussellB/aml/dce-viewer/server/dce-viewer.service /etc/systemd/system

echo "-- restarting dce-viewer --"
sudo systemctl daemon-reload
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl start dce-viewer
sudo systemctl enable dce-viewer
