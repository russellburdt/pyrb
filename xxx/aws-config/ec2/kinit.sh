#!/bin/bash -e

echo "********************************************"

[[ "" == "" ]] && CCACHE_DIR="/dev/shm"

if [[ -z "/etc/krb5.keytab" ]]; then
  echo "*** Warning default keytab (/etc/krb5.keytab) not found"
  exit 1
fi

# report to stdout the time the kinit was being run
echo "*** kinit at $(date)"

# run kinit with passed options. The verbose option is selected
kinit -V -k svc_MLDev2@DRIVECAM.NET
# kinit -V -k svc_MLDev@DRIVECAM.NET

# report the valid tokens
klist

echo "********************************************"
