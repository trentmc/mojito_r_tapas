#!/bin/bash
#
# start a set of slaves on remote machines
#
HOSTS="$HOSTS ambleve durme demer dommel viroin dijle zenne nete warche semois lomme vesder herk jeker rupel leie"
HOSTS="$HOSTS chertal maaseik lixhe vise herstal seraing amay ampsin huy andenne jambes wepion yvoir dinant anseremme hastiere"
HOSTS="$HOSTS oudenaarde mariekerke burcht lillo eke kallo temse stamands melle hoboken gavere doel zevergem wetteren hemiksem baasrode"

# if the pooler exits, all processes should be stopped
# rather brute-force...
DIR=`pwd`
for HOST in $HOSTS
do
    echo "Grabbing $HOST..."
    ssh -x -f $HOST "python $DIR/loop.py"
done

