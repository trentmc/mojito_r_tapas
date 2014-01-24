#!/bin/bash
#
# start a set of slaves on remote machines
#

SYNTH_DIR=$1
CLUSTER_ID=$2

START_COMMAND="sh $SYNTH_DIR/clustertools/startsim.sh"

DUAL_PROCESSOR_HOSTS=""
SINGLE_PROCESSOR_HOSTS=""

#DUAL_PROCESSOR_HOSTS="scapa portellen dalwhinnie"
#SINGLE_PROCESSOR_HOSTS="titan hyperion iapetus"

# MICAS machines
# single core
#SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS scapa portellen dalwhinnie benrinnes bushmills lagavulin ileach longmorn tobermory caolila"
# the dual cores
#DUAL_PROCESSOR_HOSTS="$DUAL_PROCESSOR_HOSTS titan hyperion iapetus ananke deimos rhea callisto phobos tethys ganymedes" 
  # these are to be used with care ;)
  #SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS benrinnes jack daniels "
  #DUAL_PROCESSOR_HOSTS="$DUAL_PROCESSOR_HOSTS tethys tomatin"

# these are some computer class machines

# class 1 (91.56)
  SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS durme demer dommel viroin dijle zenne nete warche semois lomme vesder herk jeker rupel leie"

    #all of them
    #SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS ambleve durme demer dommel viroin dijle zenne nete warche semois lomme vesder herk jeker rupel leie"

# class 2 (00.91)
  
  SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS chertal maaseik lixhe vise herstal seraing amay ampsin huy andenne jambes wepion yvoir dinant anseremme hastiere"

    #all of them
    #SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS chertal maaseik lixhe vise herstal seraing amay ampsin huy andenne jambes wepion yvoir dinant anseremme hastiere"

# class 3 (02.54)

  SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS oudenaarde mariekerke burcht lillo eke kallo temse stamands melle hoboken gavere doel zevergem wetteren hemiksem baasrode"

    #all of them
    #SINGLE_PROCESSOR_HOSTS="$SINGLE_PROCESSOR_HOSTS oudenaarde mariekerke burcht lillo eke kallo temse stamands melle hoboken gavere doel  zevergem wetteren hemiksem baasrode"

# start the engine on the single processors
for HOST in $SINGLE_PROCESSOR_HOSTS
do
	ssh -x -f $HOST $START_COMMAND " $SYNTH_DIR $CLUSTER_ID"
done
# start the engine on the dual processors
for HOST in $DUAL_PROCESSOR_HOSTS
do
	ssh -x -f $HOST $START_COMMAND " $SYNTH_DIR $CLUSTER_ID"

	ssh -x -f $HOST $START_COMMAND " $SYNTH_DIR $CLUSTER_ID"

done

# wait for user input to terminate slaves
echo -n "Press return to terminate the slaves..."
read terminate

# if the pooler exits, all processes should be stopped
# rather brute-force...
for HOST in $SINGLE_PROCESSOR_HOSTS
do
	# a little rude
	ssh -f $HOST "killall python;killall hspice;killall $START_COMMAND"
done

for HOST in $DUAL_PROCESSOR_HOSTS
do
	# a little rude
	ssh -f $HOST "killall python;killall hspice;killall $START_COMMAND"
done

