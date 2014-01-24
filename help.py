#!/usr/bin/env python 

if __name__== '__main__':            

    #set help message
    help = """
Usage: help

TOOLS:

=== To generate results ===
1. dispatcher.py - the daemon that manages messages between master and slaves. Run first.
2. master.py - controls the overall synthesis algorithmically
3. slave.py - start >=1 of these, which do the master's bidding (e.g. invoke sim)

=== To manipulate DBs ===
-resimulate.py - re-simulates every individual in DB; useful for changed or added TBs
-catdb.py -- concatenates >=1 input DBs into a new DB.  Can extract a subset of 1st DB.
-get_ind.py -- extract one ind from a DB and put into a just-ind pickle file.
  Use netlister2.py to analyze these inds.

=== To analyze results ===
-summarize_db.py - list nondominated inds and their performances in a DB
-netlister.py - netlist a single ind from DB_FILE
-netlister2.py - netlist a single ind from IND_FILE (from get_ind.py)
-modeltest.py - get relative influences of metrics=>topo_var, or topo/all_vars=>metric

=== Utilities ===
-listproblems.py - lists the problems and their associated numbers.
-doprune_lut_data - shrinks the size of a lookup table (lut)
-help.py - this file.  To get help for other files just type filename with no args.
-runtests.py - run unit tests (can also run individual tests from their dirs)
-call repeat_plot_nmse() from octave to analyze nmse results on-the-fly (prob 81)
"""
    print help
