#!/usr/bin/env python 

"""Use this script to quickly generate datasets for nondominated filtering / sorting,
which can then be used for performance profiling.
"""

import random

import sys


if __name__== '__main__':
    help = """
Usage: generate_nondom_datasets NUM_SETS NUM_OBJS NUM_INDS

Will generate a file nondom_data_XXd_YYsamples.py
-where XX is NUM_OBJECTIVES, and YY is NUM_INDS
-the .py file has routines to conveniently use the data.
"""
    #got the right number of args?  If not, output help
    num_args = len(sys.argv)
    if num_args not in [4]:
        print help
        sys.exit(0)
        
    num_sets = eval(sys.argv[1])
    num_objectives = eval(sys.argv[2])
    num_inds = eval(sys.argv[3])

    filename = "nondom_data_%dd_%dsamples.py" % (num_objectives, num_inds)

    print "\nBegin generating nondominated test file '%s' ..." % filename
    f = open(filename, 'w')

    f.write('"""%s\n' % filename)
    f.write("This file was auto-generated via generate_nondom_datasets.py\n")
    f.write("\n")
    f.write("Routines supplied:\n")
    f.write("-nondomTestSet_%dd_%dsamples(set_number)\n" % (num_objectives, num_inds))
    f.write("-nondomTestSets_%dd_%dsamples()\n" % (num_objectives, num_inds))
    f.write('"""\n')
    f.write("\n")
    f.write("\n")

    f.write("def nondomTestSet_%dd_%dsamples(set_number):\n" % (num_objectives, num_inds))
    for set_i in range(num_sets):
        print "Doing set %d / %d" % (set_i+1, num_sets)
        if set_i == 0:
            f.write("    if set_number == %d:\n" % set_i)
        else:
            f.write("    elif set_number == %d:\n" % set_i)

        tuple_of_values = []
        for obj_i in range(num_objectives):
            values_for_objective = [random.random() for ind_i in range(num_inds)]
            tuple_of_values.append(values_for_objective)
        f.write("        return zip%s\n" % str(tuple(tuple_of_values)))
        f.write("\n")

    f.write("    else:    \n")
    f.write("        raise AssertionError\n")
    f.write("    \n")

    f.write("\n")
    f.write("def nondomTestSets_%dd_%dsamples():\n" % (num_objectives, num_inds))
    f.write("    return [nondomTestSet_%dd_%dsamples(set_number) for set_number in range(%d)]\n" %
            (num_objectives, num_inds, num_sets))
    f.write("\n")

    f.close()

    print "Done generating file."

