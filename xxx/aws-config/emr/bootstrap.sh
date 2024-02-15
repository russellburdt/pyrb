#!/bin/bash
if grep isMaster /mnt/var/lib/info/instance.json | grep false;
then
    sudo yum update -y
    sudo yum install postgresql htop unixODBC postgresql-odbc -y
fi
