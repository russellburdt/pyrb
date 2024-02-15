#!/usr/bin/bash

echo "-- killing nginx --"
sudo systemctl stop nginx
sudo systemctl disable nginx

echo "-- killing dce-viewer service --"
sudo systemctl stop dce-viewer
sudo systemctl disable dce-viewer

echo "-- copying server files --"
sudo cp /mnt/home/russell.burdt/Lytx.AML.RussellB/aml/dce-viewer/server/dce-viewer.info /mnt/home/russell.burdt
sudo cp /mnt/home/russell.burdt/Lytx.AML.RussellB/aml/dce-viewer/server/dce-viewer /etc/nginx/sites-available
sudo ln -s /etc/nginx/sites-available/dce-viewer /etc/nginx/sites-enabled/
sudo cp /mnt/home/russell.burdt/Lytx.AML.RussellB/aml/dce-viewer/server/dce-viewer.service /etc/systemd/system

# echo "-- restarting dce-viewer --"
sudo systemctl daemon-reload
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl start dce-viewer
sudo systemctl enable dce-viewer
