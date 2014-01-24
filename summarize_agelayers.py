#!/usr/bin/env python 

##!/usr/bin/env python2.4

import os
import sys
import numpy

def printMOEADLayer(state, layer_i, max_inds_per_weight=1):
    N = state.ss.num_inds_per_age_layer
    cands = state.R_per_age_layer[layer_i]
    print "Displaying MOEA/D layer %d..." % layer_i
    
    # find min and max values in this R for each metric
    print " Estimating metric bounds..."
    metric_bounds = EngineUtils.minMaxMetrics(state.ps, cands)
    dbg = " metric bounds: \n"
    for obj in state.ps.metricsWithObjectives():
        dbg += "  %20s: %10s -> %10s\n" % (obj.name, metric_bounds[obj.name][0], metric_bounds[obj.name][1])
    print dbg
    
    topos_seen = {}
    topo_costs = {}
    inds_for_wi = {}
    weight_topo_strings = []
    # find best ind for each weight
    for w_i in range(N):
        neighbor_I = state.indices_of_neighbors[w_i]
        costs = [ind.scalarCost(1, state.W[w_i,:], state.ss.metric_weights, metric_bounds)
                    for ind in cands]
        max_cost = max(costs)
        min_cost = min(costs)

        best_I = numpy.argsort(costs)
        inds_for_wi[w_i] = []
        topos_seen[w_i] = []
        topo_costs[w_i] = []
        for idx in best_I:
            topo = cands[idx].topoSummary()
            if topo not in topos_seen[w_i]:
                # normalized cost: 0 = best ind for this weight, 1 = worst ind for this weight
                normalized_cost = (costs[idx] - min_cost) / (max_cost - min_cost + 1e-20)
                if normalized_cost > 0.5: # ind is not in top 50% for this weight vector
                    break # don't add it

                inds_for_wi[w_i].append(cands[idx])
                topos_seen[w_i].append(topo)
                topo_costs[w_i].append(normalized_cost)
                if len(inds_for_wi[w_i]) == max_inds_per_weight:
                    break

        # check uniqueness
        assert len(set(topos_seen[w_i])) == len(topos_seen[w_i])

        s = ""
        for (idx, topo) in enumerate(topos_seen[w_i]):
            s += "[%20s %3.5f] " % (topo, topo_costs[w_i][idx])
        weight_topo_strings.append(s)

    for w_i in range(N):
        s = " w_i: %03d (%02d)" % (w_i, len(topos_seen[w_i]))
        print s + weight_topo_strings[w_i]
        neighbor_I = state.indices_of_neighbors[w_i]
        for idx in neighbor_I:
            print "   w_i: %03d    %s" % (idx, weight_topo_strings[idx])

if __name__== '__main__':
    #set up logging
    import logging
    logging.basicConfig()
    logging.getLogger('engine_utils').setLevel(logging.DEBUG)
    logging.getLogger('analysis').setLevel(logging.DEBUG)
    logging.getLogger('master').setLevel(logging.INFO)

    #set help message
    help = """
Usage: summarize_agelayers DB_FILE [MAX_INDS_PER_WEIGHT]

Prints a summary of db age layer contents:

Details:
 DB_FILE -- string -- e.g. ~/synth_results/state_genXXXX.db or pooled_db.db
 MAX_INDS_PER_WEIGHT -- int -- nb of inds to select per weight (default: loaded from db)
"""

    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [2, 3]:
        print help
        sys.exit(0)

    #yank out the args into usable values
    db_file = sys.argv[1]

    #do the work
    import engine.EngineUtils as EngineUtils
    from util import mathutil
    from engine.Channel import ChannelStrategy
    import engine.EngineUtils
    import engine.Evaluator

    # -load data
    if not os.path.exists(db_file):
        print "Cannot find file with name %s" % db_file
        sys.exit(0)
    state = EngineUtils.loadSynthState(db_file, None)
    ps = state.ps

    #if num_args > 2:
        #max_inds_per_weight = eval(sys.argv[2])
    #else:
        #max_inds_per_weight = state.ss.topology_layers_per_weight

    #for i in range(len(state.R_per_age_layer)):
        #printMOEADLayer(state, i, max_inds_per_weight)
    inds = list(set(state.R_per_age_layer[0]))
    (layer, kicked_inds, layer_cost) = EngineUtils.prepareMOEADLayer(ps, inds, state.W, state.ss.metric_weights, state.ss.topology_layers_per_weight)
    clusters = EngineUtils.clusterPerTopology(layer, layer_cost, state.W, state.indices_of_neighbors)
    for cluster in clusters:
        (ind, weight_vector) = cluster.getBestIndAndWeight()
        print "cluster %3d gives ind %15s for weight %s" % (cluster.ID, ind.shortID(), weight_vector)

    #inds = state.R_per_age_layer[6]
    #state.R_per_age_layer = EngineUtils.AgeLayeredPop()
    #state.R_per_age_layer.append(inds)

    #state.nominal_nondom_inds = EngineUtils.nondominatedFilter(inds)
    #state.nominal_nondom_inds_current = inds

    #state.save('test.db')
    #done!
    print "Done summarize_agelayers.py"
