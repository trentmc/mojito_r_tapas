#!/bin/bash
#
# start a synth run on a remote machine
#
#  $1 = base directory of the synth engine
#  $2 = cluster id

renice 15 $$ 2> /dev/null 1> /dev/null
source ~/.bashrc  2> /dev/null 1> /dev/null
source ~/.bash_profile 2> /dev/null 1> /dev/null
source ~/scripts/hspice.rc 2> /dev/null 1> /dev/null
cd $1 2> /dev/null 1> /dev/null

HOST=`hostname`

CURRENT_LOAD=$(uptime | gawk -F, '{print $3}' | gawk '{print $3}')
LOAD_TST=$(echo "$CURRENT_LOAD" | sed -e 's/\([0-9]*\)\.[0-9]*/\1/')
LOAD_TST=0

if [ "$LOAD_TST" = "0" ]
then 
    python slave.py "$2" 2>&1 
#    PID=$!
#    echo "$HOST $PID" > $1/last_slave_pid
else
    echo "Skip $HOST since it has load $CURRENT_LOAD"

fi

