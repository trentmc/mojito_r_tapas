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
#
#CORE2_DUO="sinope himalia phoebe enceladus elara prometheus janus mimas pasiphae pandora"
#P4_WORKSTATIONS="glenkeith longmorn dufftown benrinnes glendeveron bushmills laphroaig scapa "\
#P4_WORKSTATIONS="$P4_WORKSTATIONS epimetheus helene metis atlas telesto calypso oberon thebe tobermory strathisla "
#P4_WORKSTATIONS="$P4_WORKSTATIONS glenspey caolila portellen jameson linkwood macallan dalwhinnie ileach lagavulin ladyburn"
#
#P4_SERVERS="jack"
#SERVERS_32BIT="tomatin"
#SERVERS_64BIT="micopt01 micopt02"

CORE2_DUO="himalia phoebe enceladus elara prometheus janus mimas pandora"
P4_WORKSTATIONS="glenkeith longmorn dufftown benrinnes glendeveron bushmills laphroaig scapa "
P4_WORKSTATIONS="$P4_WORKSTATIONS epimetheus helene metis atlas telesto calypso oberon thebe tobermory strathisla "
P4_WORKSTATIONS="$P4_WORKSTATIONS glenspey caolila portellen jameson linkwood macallan dalwhinnie ileach lagavulin ladyburn"

PENTIUM_D="ananke tethys titan callisto deimos hyperion phobos iapetus rhea"

P4_SERVERS="jack"
SERVERS_32BIT="" #"tomatin" doesn't run hspice since no SSE2
SERVERS_64BIT="micopt01 micopt02"

SINGLE_PROCESSOR_HOSTS="$P4_WORKSTATIONS"
DUAL_PROCESSOR_HOSTS="$CORE2_DUO $PENTIUM_D $SERVERS_32BIT"

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

