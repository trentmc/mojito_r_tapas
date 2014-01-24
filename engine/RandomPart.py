"""RandomPart.py

Has routines related to doing topological mutations of a library, for
novelty-generating synthesis.

Classes:
-RandomPartFactory -- adds a new random part to library, helped by...
-MutateManager -- copies and mutates an embedded part until successful
-MutateWorker -- does in-place mutation of an embedded part.  Core work here.
"""

import copy
import logging
import random
import string
import types

from adts import *
from util.constants import AGGR_TEST
from util import mathutil

log = logging.getLogger('slave')

class RandomPartFactory:
    """High-level control class functionality"""
    def __init__(self, library):
        self._library = library
        self.mutate_manager = None
    
    def build(self, emb_part, unscaled_opt_point):
        """
        @description

          Try to add a random part somewhere in emb_part.  If first try
          was unsuccessful, keep trying until successful (or hit max tries).

        @arguments

          emb_part -- EmbeddedPart --
          unscaled_opt_point -- Point

        @return

          var_name -- string -- name of the optvar at the highest level of 'self'
            which is the choice_var needed to choose the new part
          var_value -- int -- corresponds to the value we need to set to the
            var in order to choose the new part

        @exceptions

        @notes
        """
        #preconditions
        assert isinstance(emb_part, EmbeddedPart)
        assert isinstance(unscaled_opt_point, Point)
        assert not unscaled_opt_point.is_scaled

        #main work...
        max_num_tries = 20 #magic number alert
        num_tries = 0
        changed = False
        while True:
            #log.info("Before a mutate try: tot # topologies = %d" %
            #         emb_part.part.numSubpartPermutations())
            (changed, var_name, var_value) = self._tryOnce(
                emb_part, unscaled_opt_point)
            #log.info("After a mutate try: tot # topologies = %d" %
            #         emb_part.part.numSubpartPermutations())

            #successful stop?
            if changed:
                break

            #prevent infinite loops and pointless trying
            # (should usually get this with 1 or a few tries!)
            num_tries += 1
            if num_tries > max_num_tries:
                log.info('Ran out of re-tries, so no change')
                break
            else:
                log.info('Did not make a change, so do re-try #%d' % num_tries)
            
        return (var_name, var_value)
        

    def _tryOnce(self, emb_part, unscaled_opt_point):
        """
        @description

          1. Randomly choose one of the FlexPart parts in emb_part's hierarchy
             of Parts
          2. That FlexPart gets a new part_choice, which is an altered copy
          of one of its other part_choices.
          3. Update PointMetas of all Parts which contain that FlexPart
          within their hierarchy

        @arguments

          emb_part -- EmbeddedPart --
          unscaled_opt_point -- Point

        @return

          changed -- bool -- was a change succesfully made?
          var_name -- string -- name of the optvar at the highest level of 'self'
            which is the choice_var needed to choose the new part
          var_value -- int -- corresponds to the value we need to set to the
            var in order to choose the new part

        @exceptions

        @notes
        """
        self.mutate_manager = MutateManager(self._library, emb_part)
        
        #base info
        pm = emb_part.part.point_meta
        vars_before = pm.keys()
        
        #preconditions
        assert isinstance(emb_part, EmbeddedPart)

        #choose an embedded FlexPart instance, and also retrieve the
        # the corresponding toplevel var 'var_name' that uses it
        scaled_point = pm.scale(unscaled_opt_point)            
        tuples = emb_part.mutatableFlexTuples(scaled_point)   
        (emb_flex_part, var_name) = random.choice(tuples)
        flex_part = emb_flex_part.part

        #give flex part a new part choice
        varmeta_before = copy.copy(flex_part.choiceVarMeta())
        num_choices_before = len(flex_part.part_choices)
        changed = self._addRandomPartChoice(flex_part)
        if changed:
            assert len(flex_part.part_choices) == (num_choices_before + 1)
        
        var_value = len(flex_part.part_choices) - 1

        #update opt point metas of all parts which contain that flex part
        # within their hierarchy
        if changed:
            emb_part.updateOptPointMetaFlexPartChoices(
                broadening_means_novel = True)

        #postconditions
        if AGGR_TEST and changed:
            try:
                validateVarLists(vars_before, emb_part.part.point_meta.keys(),
                                 'vars_before', 'vars_after')
                varmeta_after = flex_part.choiceVarMeta()
                assert var_name == varmeta_before.name == varmeta_after.name
                assert var_name in vars_before
                flex_part.validateChoices()
                for other_flex_part in emb_part.flexParts():
                    other_flex_part.validateChoices()
                assert (varmeta_before.possible_values + [var_value]) == \
                       varmeta_after.possible_values
                assert set(varmeta_before.nonNovelValues()) == \
                       set(varmeta_after.nonNovelValues())
                assert (varmeta_before.novel_values + [var_value]) == \
                       varmeta_after.novel_values
            except:
                import pdb;pdb.set_trace()

        #return
        return (changed, var_name, var_value)

    def _addRandomPartChoice(self, flex_part):
        """
        @description

          Randomly choose one of self's current part_choices, copy and
          alter it, and make that new parts a new part_choice.

        @arguments

          flex_part -- FlexPart object

        @return

          changed -- bool -- was a change made?
          PLUS alters the internals of input argument flex_part

        @exceptions

        @notes
        """
        #preconditions
        assert isinstance(flex_part, FlexPart)
        assert len(flex_part.part_choices) > 0

        #work: choose the embedded part to mutate
        # -it can't be novel (too mean), or flex (don't have operators on flex)
        nonnovel_I = flex_part.choiceVarMeta().nonNovelValues()
        cand_emb_parts = [flex_part.part_choices[i]
                          for i in nonnovel_I
                          if flex_part.part_choices[i].part.parttype != \
                          FLEX_PART_TYPE]
        if cand_emb_parts:
            old_emb_part = random.choice(cand_emb_parts)
        else:
            return False
        
        #for validate
        bkp_old_emb_emb_parts = [copy.copy(p) for p in
                                 old_emb_part.part.possibleEmbeddedParts()]
        
        #work: create a mutated copy
        tabu_ID = flex_part.ID
        (changed, new_emb_part) = self.mutate_manager.createMutatedPart(
            old_emb_part, tabu_ID)

        #validate
        self._validateEmbPartsAreDifferent(new_emb_part, old_emb_part)
        self._validateEmbPartUnchanged(old_emb_part, bkp_old_emb_emb_parts)
        
        if changed:
            #work: add the choice
            flex_part.addPartChoice(new_emb_part.part,
                                    new_emb_part.connections,
                                    new_emb_part.functions,
                                    is_novel = True)
            #work: report and exit
            log.info("Added a new random part choice named '%s' to '%s'" %
                     (new_emb_part.part.name, flex_part.name))
        else:
            log.info("No changes made by MutateManager")

        #force a clear the backup parts (help memory)
        while bkp_old_emb_emb_parts:
            del bkp_old_emb_emb_parts[0]

        #postconditions
        assert not (changed and (new_emb_part is None))
                
        return changed

    def _validateEmbPartsAreDifferent(self, new_emb_part, old_emb_part):
        """
        Validate: are dangerous references excorcised?
         (implicitly covers that the old_emb_part didn't change at
          its level, and at its .part level)
        """
        if AGGR_TEST:
            assert id(new_emb_part) != id(old_emb_part)
            assert id(new_emb_part.part) != id(old_emb_part.part)
            assert id(new_emb_part.connections) != id(old_emb_part.connections)
            assert id(new_emb_part.functions) != id(old_emb_part.functions)
            assert id(new_emb_part.part.point_meta) != \
                   id(old_emb_part.part.point_meta)
            for varname in new_emb_part.part.point_meta.keys():
                assert id(new_emb_part.part.point_meta[varname]) != \
                       id(old_emb_part.part.point_meta[varname])
            assert set(new_emb_part.part.point_meta.keys()) == \
                   set(old_emb_part.part.point_meta.keys())
            old_emb_part.part.validate()

    def _validateEmbPartUnchanged(self, old_emb_part, bkp_old_emb_emb_parts):
        """
        Validate: cover that old_emb_part didn't change at level of
          each  of its .part.possibleEmbeddedParts() level
        """
        if AGGR_TEST:
            old_emb_emb_parts = old_emb_part.part.possibleEmbeddedParts()
            assert len(old_emb_emb_parts) == len(bkp_old_emb_emb_parts)
            for (p, bkp_p) in zip(old_emb_emb_parts, bkp_old_emb_emb_parts):
                assert p == bkp_p

    
class MutateManager:
    """Knows how to mutate an embedded part via copying (i.e. not in-place)"""

    def __init__(self, library, toplevel_emb_part):
        self._library = library
        self._worker = MutateWorker(library, toplevel_emb_part)
    
    def createMutatedPart(self, emb_part, tabu_ID):
        """
        @description

          Copy and mutate emb_part's part and/or connections.

        @arguments

          emb_part -- EmbeddedPart -- what to mutate.  Note that this structure
            should NOT get altered within this routine
          tabu_ID -- int -- whatever parts get added, they cannot have this
            ID (if they do, recursion will result)

        @return

          changed -- bool -- was a change successfully made?
          mutated_emb_part -- EmbeddedPart -- like emb_part, but with different
            components and/or connections

        @exceptions

        @notes

          Implemented as merely a pass-through to respective mutations per part
          type;but remember that those parts can be converted to different types!
        """
        #main work
        if isinstance(emb_part.part, AtomicPart):
            (changed, mutated_emb_part) = self._mutateEmbeddedAtomicPart(
                emb_part, tabu_ID)
        elif isinstance(emb_part.part, CompoundPart):
            (changed, mutated_emb_part) = self._mutateEmbeddedCompoundPart(
                emb_part,tabu_ID)
        elif isinstance(emb_part.part, FlexPart):
            #don't mutate flex parts - not worth payoff vs. risk
            return (False, None)
        else:
            raise 'unknown part type'

        #postconditions
        if AGGR_TEST:
            emb_part.part.validate()
            emb_part.validate()
            mutated_emb_part.validate()
            mutated_emb_part.part.validate()

        #temporary postcondition - call to subPartsInfo can break
        if False: #_really_ slow
            for round_i in range(20):
                scaled_point = emb_part.part.point_meta.createRandomScaledPoint(
                    True)
                info = emb_part.subPartsInfo(scaled_point)

        #return
        return (changed, mutated_emb_part)

    def _mutateEmbeddedAtomicPart(self, orig_emb_part, tabu_ID):
        """Like mutateEmbeddedPart, except specifically for AtomicParts."""
        #for simplicity and more flexibility, merely convert to Compound
        # and hand off
        new_part = atomicPartToCompoundPart(orig_emb_part.part)
        new_emb_part = EmbeddedPart(new_part,
                                    copy.copy(orig_emb_part.connections),
                                    copy.copy(orig_emb_part.functions))

        #call main work
        changed = self._worker.inPlaceMutate(new_emb_part, tabu_ID)

        return (changed, new_emb_part)

    def _mutateEmbeddedCompoundPart(self, orig_emb_part, tabu_ID):
        """Like mutateEmbeddedPart, except specifically for CompoundParts"""
        new_part = copyCompoundPart(orig_emb_part.part)
        new_emb_part = EmbeddedPart(new_part,
                                    copy.copy(orig_emb_part.connections),
                                    copy.copy(orig_emb_part.functions))

        #call main work
        changed = self._worker.inPlaceMutate(new_emb_part, tabu_ID)

        #posconditions and return
        return (changed, new_emb_part)

class MutateWorker:
    """Mutates input-argument embedded compound parts in-place.
    Has all the actual 'worker' routines for mutation"""
    def __init__(self, library, toplevel_emb_part):
        self._library = library
        self._toplevel_emb_part = toplevel_emb_part

        
    def inPlaceMutate(self, emb_part, tabu_ID, forced_op=None):
        """Does an in-place mutate of emb_part.  Returns: changed=True/False."""
        log.info("Try to mutate '%s'" % emb_part.part.name)

        bias_per_op = {
            #alter 'connections'
            'shuffleConnections' : 0.25, #lower because easier for success

            #
            #'modifyFixedWeightsOfNovelPart' : 5.0,
            
            #add parts
            'addTwoPortSeries' : 1.0,
            'addTwoPortParallel' : 1.0,
            #'addDcvs' : 1.0, #tlm -- this could make a HUGE speed difference
            #'addMosChannelSeries' : 1.0,
            #'addMosChannelParallel' : 1.0,
            #'addEmbPartCopyInSeries' : 1.0,

            #delete parts
            #'deleteAndShortTwoPart' : 1.0,
            #'deleteAndOpenTwoPart' : 1.0,
            #'deleteAndShortMos' : 1.0,
            #'deleteAndOpenMos' : 1.0,

            #change parts
            #'changeBiasedMosTo3TerminalMos' : 1.0,      
            #'change3TerminalMosToBiasedMos' : 1.0,            

            #alter functions
            #(don't need any)
            }
        
        changed = False
        num_tries = 0
        max_tries = 200 #magic number
        while (not changed) and (num_tries < max_tries):
            num_tries += 1
            
            if forced_op is None:
                op_name = mathutil.randIndexFromDict(bias_per_op)
            else:
                op_name = forced_op
            
            log.info("Mutate try #%d, apply operator: %s" %
                     (num_tries, op_name))
            exec('changed = self.' + op_name + '(emb_part, tabu_ID)')
            assert isinstance(changed, types.BooleanType)

        log.info("Changed in mutate? %s" % changed)
        return changed
    
    def shuffleConnections(self, emb_part, tabu_ID):
        """Scramble emb_part's connections.
        In-place modifies emb_part; returns 'changed'
        """
        upper_name = string.upper(emb_part.part.name)
        #corner case: shuffle impossible
        if len(emb_part.connections) <= 1:
            return False

        #corner case: shuffle meaningless because ports are symmetrical
        # magic number alert (b/c dependent on library part names)
        else:
            tabu_subnames = ['WIRE','RES','ESISTOR','CAP','APACITOR',
                             'SHORT','OPEN','RC_SERIES', 'DIODE']
            for tabu_subname in tabu_subnames:
                if tabu_subname in upper_name:
                    return False

        #main case
        old_ports = emb_part.connections.keys()

        #anything with 'mos' in name means that shuffling
        # S and D is meaningless (assuming that S and D are symmetrical).
        # magic number alert (b/c dependent on library part names)
        if ('MOS' in upper_name) and (len(old_ports) > 2) and \
               ('D' in old_ports) and ('S' in old_ports):
            new_ports = self._scrambleDSsymmetricUntilDifferent(old_ports)

            #corner case: shuffle impossible
            if new_ports == old_ports:
                return False

        #non-DS symmetric.  Ensure that the ordering is different, via 'while'.
        else:
            new_ports = old_ports[:]
            random.shuffle(new_ports)
            while new_ports == old_ports:
                random.shuffle(new_ports)

        old_conn = copy.copy(emb_part.connections)
        emb_part.connections = dict(zip(new_ports,emb_part.connections.values()))
        assert emb_part.connections != old_conn
        log.info("Shuffle connections from %s, to %s" %
                 (old_conn, emb_part.connections))
        return True

    def _scrambleDSsymmetricUntilDifferent(self, old_ports):
        """Helper to shuffleConnections' handling of DS ports"""
        agn_old_ports = self._DSagnostic(old_ports)
        new_ports = old_ports[:]
        random.shuffle(new_ports)
        num_loops = 0
        while True:
            #are new ports different?
            agn_new_ports = self._DSagnostic(new_ports)
            if agn_old_ports != agn_old_ports: 
                break

            #avoid infinite loops
            num_loops += 1
            if num_loops > 100:
                break

            #re-try to make different
            random.shuffle(new_ports)
        return new_ports

    def _DSagnostic(self, ports):
        """Replaces any 'D' or 'S' in list of port names with 'DS_XXX'
        Helper to _scrambleDSsymmetricUntilDifferent."""
        new_ports = []
        for port in ports:
            if port == 'D' or port == 'S':
                new_ports.append('DS_XXX')
            else:
                new_ports.append(port)
        return new_ports

    def addTwoPortParallel(self, emb_part, tabu_ID):
        """Choose a two-port part from within parts library.  Then
        add it in parallel.
        In-place modifies emb_part; returns 'changed'

        Disallows open circuits because they are meaningless.
        """
        cands = [part
                 for part in self._allTwoPortPartsForAdd()
                 if not part.mayContainPartWithID(tabu_ID)
                 and 'OPEN' not in string.upper(part.name)]

        s = "Can add these parts: %s" % str([part.name for part in cands])
        log.info(s)

        if len(cands) == 0:
            return False
        part_to_add = random.choice(cands)
        changed = self._addTwoPortParallel(emb_part.part, part_to_add)
        return changed
    
    def addTwoPortSeries(self, emb_part, tabu_ID):
        """Choose a two-port part from within parts library.  Then
        add it in series.
        In-place modifies emb_part; returns 'changed'
        
        Disallows wires / short circuits because they are meaningless.
        """
        cands = [part
                 for part in self._allTwoPortPartsForAdd()
                 if not part.mayContainPartWithID(tabu_ID)
                 and 'SHORT' not in string.upper(part.name)
                 and 'WIRE' not in string.upper(part.name)]

        s = "Can add these parts: %s" % str([part.name for part in cands])
        log.info(s)
        
        if len(cands) == 0:
            return False
        part_to_add = random.choice(cands)
        changed = self._addTwoPortSeries(emb_part.part, part_to_add)

        #postconditions (though this _really_ should be taken
        # care of below)
        emb_part.part.validate()
        
        return changed
    
    def addMosChannelSeries(self, emb_part, tabu_ID):
        pass
    
    def addMosChannelParallel(self, emb_part, tabu_ID):
        pass

    #====================================================================
    #helper functions

    def _allTwoPortPartsForAdd(self):
        """Returns a list of two-port parts that are suitable for add"""
        
        lib = self._library
        cand_parts = [
            #lib.wire(),
            #lib.openCircuit(),
            #lib.resistor(),         #why not: not interesting, R=>0 == wire
            lib.capacitor(),
            lib.mosDiode(),
            lib.biasedMos(),
            #lib.RC_series(),        #why not: don't want R alone, have C
            #lib.twoBiasedMoses(),   #why not: too complicated
            #lib.stackedCascodeMos(),#why not: too complicated
            ]
        
        #only use the cand_parts that are in toplevel_emb_part's hierarchy
        # (lots of defect-type issues if we don't make this restriction)
        allowed_names = [part.name for part in self._toplevel_emb_part.parts()]
        parts = [part for part in cand_parts
                 if part.name in allowed_names]#for some reason IDs don't line up

        return parts
    
    def _addTwoPortParallel(self, part, part_to_add):
        """Add the two-port 'part_to_add' in a parallel connection
        to 'part'.  In-place modifies 'part'.
        Returns 'changed' (bool).
        """
        log.info("_add '%s' (ID=%d) to '%s' (ID=%d)" %
                 (part_to_add.name, part.ID, part.name, part.ID))
        if len(part.portNames()) < 2:
            return False

        #set connections
        chosen_nodes = random.sample(part.portNames(), 2)
        connections = {part_to_add.externalPortnames()[0] : chosen_nodes[0],
                       part_to_add.externalPortnames()[1] : chosen_nodes[1]}

        #functions are set up so that new vars are not introduced, by
        # merely using randomly chosen values.
        functions = part_to_add.point_meta.createRandomScaledPoint(False)
        for var, val in functions.iteritems(): functions[var] = str(val)

        #actually add the part
        part.addPart(part_to_add, connections, functions)

        #temporary postconditions (remove later and rely on higher-level postc.)
        part.validate()

        #done
        return True
    
    def _addTwoPortSeries(self, part, part_to_add):
        """Add the two-port 'part_to_add' in a series connection
        within 'part'.  In-place modifies 'part'.

        Details:
        -create a new internal node
        -randomly choose either an internal or external node of 'part'
        -reassign half the connections of chosen node to new internal node
        -connect part_to_add to the 2 ports_to_reconnect reported by random-split
        
        Returns 'changed' (bool).
        """
        log.info("_add '%s' (ID=%d) to '%s' (ID=%d)" %
                 (part_to_add.name, part.ID, part.name, part.ID))
        (changed, ports_to_reconnect) = self._splitRandomlyChosenNode(part)
        if not changed:
            return False
        elif ports_to_reconnect[0] == ports_to_reconnect[1]: #dcvs causes this
            return False

        #preconditions
        assert len(ports_to_reconnect) == 2
        external_ports_before = copy.copy(part.externalPortnames())

        #set connections
        random.shuffle(ports_to_reconnect)
        connections = {part_to_add.externalPortnames()[0]: ports_to_reconnect[0],
                       part_to_add.externalPortnames()[1]: ports_to_reconnect[1]}

        #functions are set up so that new vars are not introduced, by
        # merely using randomly chosen values.
        functions = part_to_add.point_meta.createRandomScaledPoint(False)
        for var, val in functions.iteritems(): functions[var] = str(val)
        
        #actually add the part
        part.addPart(part_to_add, connections, functions)

        #temporary postconditions (remove later and rely on higher-level postc.)
        validateVarLists(external_ports_before, part.externalPortnames(),
                         'ext. ports before', 'ext. ports after')
        part.validate()

        #done
        return True
    
    def _splitRandomlyChosenNode(self, part):
        """
        Helper to _addTwoPortSeries()
        
        Returns: (changed, ports_to_connect or None)
        
        This will leave danglers, but those will be recovered by a series
        connection.
        """
        if len(part.portNames()) < 2:
            return (False, None)

        old_port = random.choice(part.portNames())
        old_is_external = (old_port in part.externalPortnames())
        new_port = part.addInternalNode()
        
        #what connections are possibly switchable?
        all_connections = [] # list of (emb_part, emb_port)
        for emb_part in part.embedded_parts:
            for emb_port, main_port in emb_part.connections.iteritems():
                if main_port == old_port:
                    all_connections.append((emb_part, emb_port))

        #-there's a chance that the old port doesn't connect to anything
        # internally.  This is perfectly ok for certain parts.
        if old_is_external and len(all_connections) == 0:
            return (False, None)

        if (not old_is_external) and len(all_connections) <= 1:
            raise AssertionError("each internal node should have >1 connections")

        #-now actually rewire the connections based on the constraints of:
        #  -external_ports need >=1 connections
        #  -internal ports need >=2
        #-the new_port is an internal port, therefore ultimately >=2 connections
        #-rewiring to series part will supply 1 connection to old_part and
        # 1 connection to new_part
        #-therefore, here we need to ensure that after the switch:
        #  -new_port gets >=1 switches
        #  -if old_port is external, it retains >=0 after switch
        #  -if old_port is internal, it retains >=1 after switch
        min_num_switch = 1
        if old_is_external:
            max_num_switch = len(all_connections)
        else:
            max_num_switch = len(all_connections) - 1
        num_switch = random.randint(min_num_switch, max_num_switch)

        switch_connections = random.sample(all_connections, num_switch)
        for (emb_part, emb_port) in switch_connections:
            if AGGR_TEST:
                assert emb_port in emb_part.connections.keys()
            emb_part.connections[emb_port] = new_port

        #done
        return (True, [old_port, new_port])
            
        
