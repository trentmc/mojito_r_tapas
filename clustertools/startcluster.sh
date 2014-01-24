#!/bin/bash
#
# start a set of slaves on remote machines
#

SYNTH_DIR=$1
CLUSTER_ID=$2

sh startslaves_micas.sh $@
#sh startslaves_pcroom1.sh $@
