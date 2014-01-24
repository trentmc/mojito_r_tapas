import re
import sys

# common variables

cost_re = r"""INFO:master:Gen=([0-9]*) / age_layer=([0-9]*).*front cost: ([0-9.-]*).*tot_#_evals=([0-9]*).*\(([0-9]*) on funcs, ([0-9]*) on sim\).*evals_per_func_analysis=(.*);.*evals_per_sim_analysis=(.*)"""
#INFO:master:Gen=0 (# age layers=0): begin
generation_re = "INFO:master:Gen=([0-9]*) \(# age layers=([0-9]*)\): begin"
#INFO:master:Of 200 random inds, 45 inds were used in the initial layer.
randomused_re = "INFO:master:Of ([0-9]*) random inds, ([0-9]*) inds were used in the initial layer."
#  Of the 45 individuals (45 unique ones), there are 42 unique topologies.
topocount_re = "Of the ([0-9]*) individuals \([0-9]* unique ones\), there are ([0-9]*) unique topologies."

# method 1: using a compile object
cost_obj = re.compile(cost_re)
generation_obj = re.compile(generation_re)
randomused_obj = re.compile(randomused_re)
topocount_obj = re.compile(topocount_re)

fid = open(sys.argv[1])
fid_out = open(sys.argv[2], "w")

front_cost_per_gen = {}
topo_count_per_gen = {}
ind_count_per_gen = {}
topo_count_rnd_per_gen = {}
ind_count_rnd_per_gen = {}

max_age_layers = 0

last_gen_seen = 0
last_age_layer_seen = 0
just_saw_rnd = False

line = fid.readline()
while line:
    match_obj = generation_obj.search(line)
    if match_obj:
        generation = eval(match_obj.group(1))
        nb_age_layers = eval(match_obj.group(2))
        print " GEN: %s %s" % (generation, nb_age_layers)
        last_gen_seen = generation
        last_age_layer_seen = 0

    match_obj = cost_obj.search(line)
    if match_obj:
        # Retrieve group(s) by index
        generation = eval(match_obj.group(1))
        age_layer = eval(match_obj.group(2))
        max_age_layers = max(age_layer, max_age_layers)
        front_cost = match_obj.group(3)
        nb_evals = match_obj.group(4)
        nb_evals_on_funcs = match_obj.group(5)
        nb_evals_on_sim = match_obj.group(6)
        evals_per_func_analysis = match_obj.group(7)
        evals_per_sim_analysis = match_obj.group(8)
    
        s = "%s %s %s %s %s %s" % (generation, age_layer, front_cost, nb_evals, nb_evals_on_funcs, nb_evals_on_sim)
        print "   COST: %s" % s
        #fid_out.write(s)
        last_age_layer_seen = age_layer

        if not generation in front_cost_per_gen.keys():
            front_cost_per_gen[generation] = []
    
        front_cost_per_gen[generation].append(front_cost)
        
    match_obj = randomused_obj.search(line)
    if match_obj:
        rnd_total = eval(match_obj.group(1))
        rnd_used = eval(match_obj.group(2))
        print "   RND: %s %s" % (rnd_total, rnd_used)
        just_saw_rnd = True

    match_obj = topocount_obj.search(line)
    if match_obj:
        nb_inds = eval(match_obj.group(1))
        nb_topos = eval(match_obj.group(2))
        print "   TOPO: %s %s" % (nb_inds, nb_topos)

        if just_saw_rnd:
            just_saw_rnd = False
            if not last_gen_seen in topo_count_rnd_per_gen.keys():
                topo_count_rnd_per_gen[last_gen_seen] = []
            topo_count_rnd_per_gen[last_gen_seen].append(nb_topos)

            if not last_gen_seen in ind_count_rnd_per_gen.keys():
                ind_count_rnd_per_gen[last_gen_seen] = []
            ind_count_rnd_per_gen[last_gen_seen].append(nb_inds)

        else:
            if not last_gen_seen in topo_count_per_gen.keys():
                topo_count_per_gen[last_gen_seen] = []
            topo_count_per_gen[last_gen_seen].append(nb_topos)
    
            if not last_gen_seen in ind_count_per_gen.keys():
                ind_count_per_gen[last_gen_seen] = []
            ind_count_per_gen[last_gen_seen].append(nb_inds)

    # next line
    line = fid.readline()

fid_out.close()
fid.close()


# output
print "\n"
print "-- Cost per layer per generation"
for gen in range(min(front_cost_per_gen.keys()), max(front_cost_per_gen.keys())+1):
    front_costs = front_cost_per_gen[gen]
    front_costs.reverse() # make top layer cost first entry
    s = "%4d" % gen
    for age_layer in range(max_age_layers+1):
        if age_layer < len(front_costs): # is actually backward
          cost = eval(front_costs[age_layer])
        else:
          cost = 0.0
        s += "%20f " % cost
    print s
print "-- Topology count per layer per generation"
for gen in sorted(topo_count_per_gen.keys()):
    topo_counts = topo_count_per_gen[gen]
    topo_counts.reverse() # make top layer cost first entry
    s = "%4d" % gen
    for age_layer in range(max_age_layers+1):
        if age_layer < len(topo_counts): # is actually backward
          count = topo_counts[age_layer]
        else:
          count = 0
        s += "%5d " % count
    print s
print "-- Topology count for initial random layers"
for gen in sorted(topo_count_rnd_per_gen.keys()):
    topo_counts = topo_count_rnd_per_gen[gen]
    s = "%4d %5d" % (gen, topo_counts[0])
    print s
print "-- Ind count per layer per generation"
for gen in sorted(ind_count_per_gen.keys()):
    counts = ind_count_per_gen[gen]
    counts.reverse() # make top layer cost first entry
    s = "%4d" % gen
    for age_layer in range(max_age_layers+1):
        if age_layer < len(counts): # is actually backward
          count = counts[age_layer]
        else:
          count = 0
        s += "%5d " % count
    print s
print "-- Ind count for initial random layers"
for gen in sorted(ind_count_rnd_per_gen.keys()):
    counts = ind_count_rnd_per_gen[gen]
    s = "%4d %5d" % (gen, counts[0])
    print s

# INFO:master:Gen=0 (# age layers=0): begin