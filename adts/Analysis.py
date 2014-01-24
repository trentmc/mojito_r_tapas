"""Analysis.py

Holds:
-Analysis (abstract parent class)
-FunctionAnalysis (child class)
-CircuitAnalysis (child class)
-Simulator class -- a key attribute of CircuitAnalysis

"""

import os
import string
import time
import types

from adts import EvalUtils
from adts.Metric import Metric
from adts.Part import EmbeddedPart
from adts.Point import Point, RndPoint, EnvPoint
from util import mathutil
from util.constants import AGGR_TEST, DOCS_METRIC_NAME, BAD_METRIC_VALUE, \
     REGION_LINEAR, REGION_SATURATION, REGION_CUTOFF, SIMFILE_DIR, WAVEFORMS_FILE_EXTENSIONS

import logging
log = logging.getLogger('analysis')

region_token_to_value = {'Linear'   : REGION_LINEAR,
                         'Saturat' : REGION_SATURATION,
                         'Cutoff'   : REGION_CUTOFF}
region_value_to_str = {REGION_LINEAR : 'LINEAR',
                       REGION_SATURATION : 'SATURATION',
                       REGION_CUTOFF : 'CUTOFF'}

def startSimulationServer():
    """Start up hspice server (simulations will be client requests).  If it has already been started
    on the given machine, then it won't do it more.

    We need to call this before making calls to simulate().

    tlm - Note that I tried to use the advanced mode '-CC' but it was not responding as expected.
    """
    ## PP: the server is auto-started by the first call to simulate
    #command = "hspice -C"
    #log.info("Attempt to start simulation server with command: '%s'" % command)
    #os.system(command)
    

def stopSimulationServer():
    """Stop hspice server.
    If it does not exist, then it won't do it more"""
    command = "hspice -C -K"
    log.info("Attempt to stop simulation server with command: '%s'" % command)
    os.system(command)

class Analysis:
    """
    @description

      One invokation of a 'simulation' or a function call.  Results
      in >=1 metrics.
      
    @attributes

      ID -- int -- unique ID for this Analysis
      env_points -- list of EnvPoint -- to be thorough, need to sim at
        all of these
        
      
    @notes

      Each of the env_points coming in must be scaled.
    """
    
    # Each analysis created get a unique ID
    _ID_counter = 0L
    
    def __init__(self, env_points):
        """
        @description
        
        @arguments
        
          env_points -- list of EnvPoint
        
        @return

          Analysis object
    
        @exceptions
    
        @notes

          Abstract class.
          
        """
        #manage 'ID'
        self._ID = Analysis._ID_counter
        Analysis._ID_counter += 1
        
        #validate inputs
        if len(env_points) == 0:
            raise ValueError("Need >0 env points for each analysis")
        for env_point in env_points:
            if not env_point.is_scaled:
                raise ValueError('Env point needs to be scaled')

        #set values
        self.env_points = env_points
        self.reference_waveforms = None
        self.relative_cost = 1.0

    ID = property(lambda s: s._ID)

    def hasSimulationType(self, sim_type):
        raise NotImplementedError("Implement in child")

    def envPoint(self, target_env_ID):
        """Returns the env point having target_env_ID; raises exception if not found"""
        for env_point in self.env_points:
            if env_point.ID == target_env_ID:
                return env_point
        raise AssertionError, target_env_ID

    def __str__(self):
        """
        @description

          Abstract.
          Override str()
          
        """ 
        raise NotImplementedError('Implement in child')
        
        
class FunctionAnalysis(Analysis):
    """
    @description

      An analysis used for simulating on _functions_.  Holds one metric,
      which gets defined here (at the same time).

      Evaluation is:
        INPUTS: scaled_opt_point, env_point, rnd_point
        opt_env_point = union of scaled_opt_point & env_point
        scalar_value = self.function(opt_env_point) OR self.function(opt_env_point, rnd_point) if rnd_point not nom
        sim_results = {self.metric.name : scalar_value}
      
    @attributes

      env_points -- list of EnvPoint 
      function -- function -- a call to function(opt_env_point) OR function(opt_env_point, rnd_point)
        is considered to be the 'running' of this Analysis.  It will return the value of the _metric_
        at that (scaled) point.
      metric -- Metric object -- describes the metric that running this Analysis at a scaled_point
        will produce a measurement of
      
    @notes
      
    """
    
    def __init__(self, function, env_points, min_metric_threshold, max_metric_threshold,
                 metric_is_objective, rough_minval, rough_maxval, name_override=None):
        """
        @description

          Constructor.
        
        @arguments

          function -- see class description
          env_points -- see class description
          min_metric_threshold -- float/ int -- lower bound; helps define metric 
          max_metric_threshold -- float/ int -- upper bound; helps define metric
          metric_is_objective -- bool -- is metric an objective (vs. constraint(s) ?)
          rough_minval, rough_maxval -- float, float -- see Metric.py
          name_override -- string or None -- usually, this is None so that
            the name gets set to 'metric_' + function.func_name, but if one wants
            to give the metric a different name, then this is how.
        
        @return

          FunctionAnalysis object
    
        @exceptions
    
        @notes

        """
        Analysis.__init__(self, env_points)

        self.function = function
        
        if name_override is None:
            metric_name = 'metric_' + function.func_name
        else:
            metric_name = name_override
            
        self.metric = Metric(metric_name,
                             min_metric_threshold, max_metric_threshold,
                             metric_is_objective, rough_minval, rough_maxval)

    #make it such that we can access the single metric as if it were a list
    def _metrics(self):
        return [self.metric]
    metrics = property(_metrics)

    def hasDOC(self):
        """Returns True if string 'DOC' is in any of self's metric names.
        Helper function to stripAllButDOCs()."""
        return "DOC" in self.metric.name
    
    def getDOCmetrics(self):
        if self.hasDOC():
            return [self.metric]
        else:
            return []
    
    def hasSimulationType(self, sim_type):
        """Can't have any simulation type (e.g. 'tran') because it is not a simulator analysis."""
        return False
        
    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += 'FunctionAnalysis={'
        s += ' function=%s' % self.function
        s += '; # env_points=%d' % len(self.env_points)
        s += '; metric=%s' % self.metric
        s += ' /FunctionAnalysis}'
        return s
        
class CircuitAnalysis(Analysis):
    """
    @description

      An analysis used for simulating on _circuits_.  Holds >1 metrics,
      which gets defined here (at the same time).

      Evaluation is:
        INPUTS: ps, scaled_opt_point, env_point
        emb_part = ps.embedded_part
        emb_part.functions = scaled_opt_point
        netlist = emb_part.spiceNetlistStr(annotate_bb_info=False)
        (sim_results, lis_results, waveforms_per_ext) = analysis.simulate(netlist)
      
    @attributes

      env_points -- list of EnvPoint 
      metrics -- list of Metric objects -- describes the metrics that
        running this Analysis at a scaled_point will produce measurements of.
      simulator -- Simulator object -- knows how to call SPICE and
        retrieve corresponding output data
      relative_cost -- float -- rough simulation cost, relative to other analyses.
        (Used for structural homotopy)
      reference_waveforms -- None or 2d array -- if non-None, can be
        useful to use these to compare against the ind's output waveforms
      
    @notes

      Each of the env_points coming in must be scaled.
    """
    
    def __init__(self, env_points, metrics, simulator, relative_cost, reference_waveforms=None):
        """
        @description

          Constructor.
        
        @arguments

          env_points -- see class description
          metrics -- list of Metric objects
          simulator -- see class description
          relative_cost -- see class description
          reference_waveforms -- see class description
        
        @return

          CircuitAnalysis object
    
        @exceptions
    
        @notes

        """
        #preconditions
        if not isinstance(env_points[0], EnvPoint): raise ValueError
        if not isinstance(metrics[0], Metric): raise ValueError
        metrics_metricnames = sorted([m.name for m in metrics])
        sim_metricnames = sorted(simulator.metricNames())
        if metrics_metricnames != sim_metricnames:
            raise ValueError("These should match:\n%s\n%s" %
                             (metrics_metricnames, sim_metricnames))
        if simulator.metrics_calculator is not None:
            calc_metricnames = sorted(simulator.metrics_calculator.metricNames())
            if metrics_metricnames != calc_metricnames:
                raise ValueError("These should match:\n%s\n%s" %
                                 (metrics_metricnames, calc_metricnames))
        if not isinstance(simulator, Simulator): raise ValueError
        assert mathutil.isNumber(relative_cost)
        assert relative_cost > 0.0
        assert (reference_waveforms is None) or \
               (len(reference_waveforms.shape) == 2)

        #set values
        Analysis.__init__(self, env_points)
        self.metrics = metrics
        self.simulator = simulator
        self.relative_cost = relative_cost
        self.reference_waveforms = reference_waveforms

    def relativeCost(self):
        return self.relative_cost

    def hasDOC(self):
        """Returns True if string 'DOC' is in any of self's metric names.
        Helper function to stripAllButDOCs()."""
        for metric in self.metrics:
            if "DOC" in metric.name:
                return True
        return False

    def getDOCmetrics(self):
        doc_metrics = []
        for metric in self.metrics:
            if "DOC" in metric.name:
                doc_metrics.append(metric)
        return doc_metrics

    def hasSimulationType(self, sim_type):
        """Returns True if self's simulator has the specified sim_type of 'dc', 'op', etc."""
        return self.simulator.hasSimulationType(sim_type)

    def createFullNetlist(self, toplevel_embedded_part, scaled_opt_point, variation_data):
        return self.simulator.createFullNetlist(toplevel_embedded_part, scaled_opt_point, variation_data)
        
    def simulate(self, full_netlist):
        return self.simulator.simulate(full_netlist)

    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += 'CircuitAnalysis={'
        s += ' # env_points=%d' % len(self.env_points)
        s += '; # metrics=%d' % len(self.metrics)
        s += '; relative_cost=%.2f' % self.relative_cost
        s += '; metric names=%s' % [metric.name for metric in self.metrics]
        s += '; CircuitAnalysis.metrics={'
        for (i, metric) in enumerate(self.metrics):
            s += '%s' % metric
            if i < len(self.metrics) - 2:
                s += ', '
        s += ' /CircuitAnalysis.metrics}'
        s += '; simulator=%s' % self.simulator
        s += ' /CircuitAnalysis}'
        return s
    
class Simulator(Analysis):
    """
    @description

      Knows how to call SPICE and retrieve corresponding output data

      Holds a lot of data specifically for yanking info out
      of a simulator output file and converting it into metric
      value information.

      Used for CircuitAnalysis objects, but quite a bit more low-level.
      
    @attributes

      metrics_per_outfile -- dict of output_filetype : list_of_metric_names --
        Tells which output files to look for, for which metrics.
        Output_filetypes can include lis; ms0, ma0, mt0; sw0; ic0.
        If we want waveforms but no metrics from a particular output file,
        it still must be included here.  E.g. have 'sw0':[] entry.
        
      cir_file_path -- string -- path where to find 'base' circuit files
        (but not the ones that are temporarily generated)
                                
      max_simulation_time -- int -- max time to spend on a simulation, in
        seconds
        
      simulator_options_string -- string -- info about simulator options
        which will be embedded directly in auto-generated .cir netlist
      test_fixture_string -- string --info about test fixture
        which will be embedded directly in auto-generated .cir netlist.
        Includes:
             -input waveform generation, biases, etc
             -call to an analysis
             -'.print' commands
      lis_measures -- list of string -- which values to measure in .lis.  E.g.
        ['region','vgs'].  Will measure these on every mos device.

      The following attributes are only needed when one of the output
        filetypes is a waveform file with extensions of 'sw0':
      output_file_num_vars -- dict of extension_string : num_vars_int --
        number of variables expected in each waveform output file (eases parsing)
        Extension string can be one of WAVEFORMS_FILE_EXTENSIONS
      output_file_start_line -- dict of extension_string : start_line_int --
        where to start parsing each waveform output file.  Starts counting
        at line 0.
      number_width -- int -- number of characters taken up by a number in the
        output file (eases parsing)

      metrics_calculator -- MetricsCalculator object --
        knows how to convert a 'waveforms' 2d array to a set of metric(s)
      
    @notes

    """
    def __init__(self,
                 metrics_per_outfile,
                 cir_file_path,
                 max_simulation_time,
                 simulator_options_string,
                 test_fixture_string,
                 lis_measures,
                 output_file_num_vars=None,
                 output_file_start_line=None,
                 number_width=None,
                 metrics_calculator=None):
        """
        @description

          Constructor.  Fills attributes based on arguments.
          See class description for details about arguments.

        """
        #preconditions
        for outfile, metrics in metrics_per_outfile.items():

            if outfile not in ['lis', 'ms0', 'ma0', 'mt0', 'ic0'] + \
               WAVEFORMS_FILE_EXTENSIONS:
                raise ValueError
            
            if outfile == 'lis': #only DOC and pole-zero metrics allowed
                for metric in metrics:
                    if 'pole' in metric: pass
                    elif metric == DOCS_METRIC_NAME: pass
                    elif ')/' in metric: pass # .TF statements 
                    else: raise ValueError

            else:
                if DOCS_METRIC_NAME in metrics: raise ValueError
            
        assert isinstance(lis_measures, types.ListType)
        for metric_names in metrics_per_outfile.values():
            if not isinstance(metric_names, types.ListType): raise ValueError
        if cir_file_path[-1] != '/':
            raise ValueError("cir_file_path must end in '/'; it's: now %s" %
                             cir_file_path)

        #set values
        self.metrics_per_outfile = metrics_per_outfile
        self.cir_file_path = cir_file_path
        self.max_simulation_time = max_simulation_time
        self.simulator_options_string = simulator_options_string
        self.test_fixture_string = test_fixture_string
        self.lis_measure_names = lis_measures
        self.output_file_num_vars = output_file_num_vars
        self.output_file_start_line = output_file_start_line
        self.number_width = number_width
        self.metrics_calculator = metrics_calculator
        
    def hasSimulationType(self, sim_type):
        """Returns True if self's text_fixture includes 'sim_type', e.g. 'tran'
        """
        assert sim_type in ['dc', 'op', 'ac', 'tran', 'noise']
        for line in self.test_fixture_string.splitlines():
            if line.startswith('.' + sim_type):
                return True
        return False

    def metricNames(self):
        """List of the metric names that this analysis measures"""
        names = []
        for next_names in self.metrics_per_outfile.values():
            names.extend(next_names)
        return names

    def simulate(self, netlist):
        """
        @description

          Simulates the (full) netlist which already has design netlist, testbench code, model info, etc
          -calls self.simulator.simulate()
          -calls the simulator
          -yanks out the results
          
        @arguments

          design_netlist -- string -- describes the design.  Cannot
            simulate on its own, however; it needs testbench code around it.
          env_point -- EnvPoint object
        
        @return

           sim_results -- dict of metric_name : metric_value -- has
             there is one entry for every metric in self.analysis.
             Never includes the DOCs metric
           lis_results -- dict of 'lis__device_name__measure_name' : lis_value --
             used to compute the DOCs metric
           waveforms_per_ext -- dict of file_extension : 2d_array_of_waveforms
             For each of the waveforms outputs like .sw0
    
        @exceptions
    
        @notes

          Currently we use os.system a lot, but subprocess.call
          is easier to use; we should consider changing (warning:
          Trent's python 2.4 doesn't properly support subprocess yet)
        """
        simfile_dir = SIMFILE_DIR
        if not os.path.exists(simfile_dir):
            os.mkdir(simfile_dir)
            
        #Make sure no previous output files
        outbase = 'autogen_cirfile'
        os.system('rm ' + simfile_dir + outbase + '*;')
        if os.path.exists(simfile_dir + 'ps_temp.txt'):
            os.remove(simfile_dir + 'ps_temp.txt')

        #Create netlist, write it to file
        cirfile = simfile_dir + outbase + '.cir'
        f = open(cirfile, 'w'); f.write(netlist); f.close()

        #Call simulator; error check
        #old command = ['cd ' simfile_dir '; nice hspice -i ' cirfile ' -o ' outbase '; cd -'];

        #hspice
        # -note that the "-C" means use advanced client/server mode (works in conjunction with
        # startSimulationServer / stopSimulationServer above)
        psc = "ps ax |grep hspice|grep -v 'cd '|grep " + cirfile + \
              " 1> " + simfile_dir + "ps_temp.txt"
        command = "cd " + simfile_dir + "; nice hspice -C -i " + cirfile + \
                  " -o " + outbase + "& cd -; " + psc

        #eldo
        #psc = ['ps ax |grep eldo|grep -v ''cd ''|grep ' cirfile ' 1> ps_temp.txt'];
        #command = ['cd ' simfile_dir '; nice eldo -i ' cirfile ' -o ' outbase '& cd -; ' psc];

        #log.debug("Call with comand: '%s'" % command)
        nb_syscall_tries = 0 # allow multiple tries since sometimes the syscall fails
        while nb_syscall_tries < 5:
            nb_syscall_tries += 1
            status = os.system(command)

            output_filetypes = self.metrics_per_outfile.keys()
            result_files = [simfile_dir + outbase + '.' + output_filetype
                                  for output_filetype in output_filetypes]

            got_results = False
            bad_result = False

            if status != 0:
                got_results = True;
                bad_result = True;
                log.error('System call with bad result (try %d).  Command was: %s' %
                          ( nb_syscall_tries, command ) )
                
                #pause for 0.25 seconds (such that the system can fix whatever was causing this)
                time.sleep(0.25)
                
            else:
                break
          
        #loop until we get results, or until timeout
        t0 = time.time()
        
        while not got_results:
            
            if self._filesExist(result_files):
                #log.debug('\nSuccessfully got result file')
                got_results = True
                bad_result = False
            
            elif (time.time() - t0) > self.max_simulation_time:
                log.debug('\nExceeded max sim time of %d s, so kill' %
                          self.max_simulation_time)
                got_results = True
                bad_result = True
                      
                #kill the process
                t = EvalUtils.file2tokens(simfile_dir + 'ps_temp.txt', 0)
                log.debug('ps_temp.txt was:%s' %
                          EvalUtils.file2str(simfile_dir + 'ps_temp.txt'))
                pid = t[0]
                log.debug('fid was: %s' % pid)
                if not t[0] == 'Done':
                    os.system('kill -9 %s' % pid)

            #pause for 0.25 seconds (to avoid taking cpu time while waiting)
            time.sleep(0.25) 

        #we may have had to do a timeout kill, but there still
        # may be good results
        if self._filesExist(result_files):
            #log.debug('\nSuccessfully got result file')
            got_results = True
            bad_result = False

        if bad_result:
            log.debug('Bad result: did not successfully generate simdata')
            return self._badSimResults()

        #initialize 'sim_results' and 'all_successful_metrics' (and 'metric_names')
        all_successful_metrics = [] #list of metric names that have been successfully extracted
        sim_results = {} # dict of metric_name : metric_value.  Unsuccessful metrics have BAD_VALUE.
        metric_names = self.metricNames()
        for metric_name in metric_names:
            if metric_name != DOCS_METRIC_NAME: #don't fill this in
                sim_results[metric_name] = None

        #Add metric values measured from each result file: lis, tr0, ma0, ic0

        # -lis: from .lis file (which is like stdout)
        if 'lis' in output_filetypes:
            lis_file = simfile_dir + outbase + '.lis'
            (success, lis_results) = self._extractLisResults(lis_file)
            if not success:
                log.debug('Bad result: could not extract Lis values from .lis file')
                return self._badSimResults()
                
            all_successful_metrics.append(DOCS_METRIC_NAME)
            
            # if pz results are found
            for k in lis_results.keys():
                ks = k.split('__')
                measure_name = ks[1] + ks[2]
                if measure_name in self.metrics_per_outfile['lis']:
                    sim_results[measure_name] = lis_results[k]
                    all_successful_metrics.append(measure_name)
            (success, tf_results) = self._extractTFResults(lis_file)

            #do NOT exit if not successful here (because we may not need tf_results)
            
            for measure_name in tf_results.keys():
                sim_results[measure_name] = tf_results[measure_name]
                all_successful_metrics.append(measure_name)
            
        else:
            lis_results = {}

        # -ms0, ma0, mt0 -- .measure outputs for dc, ac, tran respectively
        extensions = [ext for ext in ['ms0','ma0','mt0'] if ext in output_filetypes]
        for extension in extensions:
            sorted_target_metrics = sorted(self.metrics_per_outfile[extension])
            start_time = time.time()
            success = False
            while not success:
                #Algorithm summary:
                # try extracting
                # if successfully have all, stop (success)
                # elif known-bad, then exit-bad
                # else reloop

                #try extracting
                output_file = simfile_dir + outbase + '.' + extension
                tokens = EvalUtils.file2tokens(output_file, 2)
                num_measures = len(tokens) / 2

                # -extract each metric will fall into one of the following categories:
                #   -successful -- metric was found, and successfully extracted a numeric value
                #   -failed -- metric was found, but its value was 'failed'
                #   -bad -- metric was not found, but its value could not be extracted
                #   -missing -- metric was not found at all
                successful_metrics, failed_metrics, missing_metrics, bad_metrics  = [], [], [], []
                for measure_i in range(num_measures):
                    measure_name = tokens[measure_i]
                    if measure_name in sorted_target_metrics:
                        try:
                            measure_value_str = tokens[num_measures + measure_i]
                            if measure_value_str == 'failed':
                                #case: failed
                                failed_metrics.append(measure_name)
                            else:
                                measure_value = eval(measure_value_str)
                                assert mathutil.isNumber(measure_value)

                                #case: successful
                                sim_results[measure_name] = measure_value
                                successful_metrics.append(measure_name)
                                if measure_name not in all_successful_metrics:
                                    all_successful_metrics.append(measure_name)
                        except:
                            #   #case: bad
                            bad_metrics.append(measure_name)

                #               #case: missing
                missing_metrics = mathutil.listDiff(sorted_target_metrics,
                                                    successful_metrics + failed_metrics + bad_metrics)
                assert sorted(successful_metrics + failed_metrics + bad_metrics + missing_metrics) == \
                       sorted_target_metrics

                # case: successfully have all, so stop
                success = (sorted(successful_metrics) == sorted_target_metrics)
                if success:
                    pass
                else:
                    metrics_s = "target_metrics=%s; successful=%s; failed=%s, bad=%s, missing=%s" % \
                                (sorted_target_metrics, successful_metrics, failed_metrics,
                                 bad_metrics, missing_metrics)
                    all_extracted = (sorted(successful_metrics + failed_metrics) == sorted_target_metrics)
                    time_exceeded = ((time.time() - start_time) > self.max_simulation_time)

                    #case: exit-BAD
                    if all_extracted or time_exceeded:
                        log.debug("Could not extract fully successful results from .%s, "
                                  "so return BAD result; %s; tokens=%s; num_measures=%d" %
                                  (extension, metrics_s, tokens, num_measures))
                        return self._badSimResults()
                    
                    #case: re-try.  Pause first.
                    else: 
                        log.info("Could not extract sim_results from .%s yet, so re-loop; %s" %
                                 (extension, metrics_s))
                        time.sleep(0.5)
                
                        
        #fill in 'waveforms_per_ext' -- dict of extension_str :2d_waveforms_array
        # and 'sim_results' related to waveforms.
        # -sw0, st0 -- waveform outputs for dc, tran respectively.  Useful in:
        #   -calculating metrics in python code rather than directly in SPICE
        #   -return waveforms for visualization
        #   -other later-in-the-flow calcs for waveforms calcs
        waveforms_per_ext = {}
        for extension in WAVEFORMS_FILE_EXTENSIONS:
            if extension not in output_filetypes: continue
            output_file = simfile_dir + outbase + '.' + extension
            try:
                start_line = self.output_file_start_line[extension]
                num_vars = self.output_file_num_vars[extension]
                waveforms_array = EvalUtils.getSpiceData(
                    output_file, self.number_width, start_line, num_vars)
            except:
                log.debug('Bad result: could not retrieve %s waveforms' % extension)
                return self._badSimResults()

            expected_num_vars = num_vars + int('tr0' in extension)
            if waveforms_array.shape[0] != expected_num_vars:
                log.debug('Bad result: # waveforms back (%d) != exp num vars' %
                          (waveforms_array.shape[0], expected_num_vars))
                return self._badSimResults()

            if waveforms_array.shape[1] <= 2:
                log.debug('Bad result: <=2 points per waveform (!?)')
                return self._badSimResults()

            if self.metrics_calculator is not None:
                val_per_metric = self.metrics_calculator.compute(waveforms_array)
                for (metric_name, metric_value) in val_per_metric.iteritems():
                    if metric_value == BAD_METRIC_VALUE:
                        log.debug("Bad result in calculating %s: BAD_METRIC_VALUE "
                                  "returned" % metric_name)
                        return self._badSimResults()
                    
                sim_results.update(val_per_metric)
                all_successful_metrics.extend(val_per_metric.keys())

            waveforms_per_ext[extension] = waveforms_array

        # -ic0: comes from .op sim
        if 'ic0' in output_filetypes:
            ic0_file = simfile_dir + outbase + '.ic0'
            tokens = EvalUtils.file2tokens(ic0_file, 2)
            for metric_name in self.metrics_per_outfile['ic0']:
                #find the token and value corresponding to 'metric_name'
                # and fill it
                found = False
                for token_i, token in enumerate(tokens):
                    if token == metric_name:
                        sim_results[token] = eval(tokens[token_i+2])
                        found = True
                        break
                    
                if not found:
                    log.debug('Bad result 3: did not find metric %s' %
                              metric_name)
                    return self._badSimResults()
                else:
                    all_successful_metrics.append(metric_name)

        # -special: pole-zero (pz) measures are a function of 'gbw' and other
        # (note: problem setup may request a subset, or none, of the following)
        # (note: somewhat HACK-like because of our dependence on special
        #  names, but it will do until we have a more general way to
        #  compute some metrics as functions of other metrics w/ error check)
        pz_dict = {'pole0fr':'pole0_margin',
                   'pole1fr':'pole1_margin',
                   'pole2fr':'pole2_margin',
                   'zero0fr':'zero0_margin',
                   'zero1fr':'zero1_margin',
                   'zero2fr':'zero2_margin',
                   }
        have_pzmeasure = mathutil.listsOverlap(pz_dict.values(), metric_names)
        if have_pzmeasure:
            #all pz measures need gbw, so find it (incl. catching error cases)
            if not sim_results.has_key('gbw'):
                log.debug("Bad result: could not find gbw")
                return self._badSimResults()
            gbw = sim_results['gbw']
            if not mathutil.isNumber(gbw) or gbw <= 0.0:
                log.debug("Bad result: gbw is not a number or is <=0")
                return self._badSimResults()

            #now fill in all the pz metrics that we care about
            for pzmeasure_name, pzmetric_name in pz_dict.items():
                if pzmetric_name not in metric_names: continue
                
                if pzmeasure_name in sim_results.keys():
                    pzmeasure = sim_results[pzmeasure_name]
                    if not mathutil.isNumber(pzmeasure):
                        log.debug("Bad result: '%s' is not a number")
                        return self._badSimResults()
                    # note that we've already caught divide-by-zero above
                    sim_results[pzmetric_name] = pzmeasure / gbw
                    all_successful_metrics.append(pzmetric_name)
                else:
                    log.debug("Bad result: '%s' not found" % pzmeasure_name)
                    return self._badSimResults()                

        #have we got all the metrics we expected?
        if sorted(all_successful_metrics) != sorted(metric_names):
            missing = sorted(mathutil.listDiff(metric_names, all_successful_metrics))
            log.debug('Bad result 4: missed metrics: %s' % missing)            
            return self._badSimResults()

        #are all metric values numbers?
        for metric_name, metric_value in sim_results.items():
            if not mathutil.isNumber(metric_value):
                log.debug('Bad result 5: some metrics are numbers: %s' % str(sim_results))
                return self._badSimResults()
        
        #Hooray, everything simulated ok! Return the results.
        s = 'Got fully good results: {'
        for metric_name, metric_value in sim_results.items():
            if metric_name[:4] != 'lis.':
                s += '%s=%g, ' % (metric_name, metric_value)
        s += '}'
        s += ' (plus %d lis values)' % len(lis_results)
        log.debug(s)
        return (sim_results, lis_results, waveforms_per_ext)


    def _extractLisResults(self, lis_file):
        """
        @description

          Helper file for simulate().

          Extracts the simulation results from a .lis file
          (e.g. for later finding out if DOCs are met)
          
        @arguments

          lis_file -- string -- should end in '.lis'
        
        @return

           success -- bool -- was extraction successful?
           lis_results -- dict of 'lis__device_name__measure_name' : lis_value
    
        @exceptions
    
        @notes
        """
        assert self.metrics_per_outfile.has_key('lis'), 'only call if want lis'
        assert lis_file[-4:] == '.lis', lis_file

        if not os.path.exists(lis_file):
            log.debug("_extractLisResults failed: couldn't find file: %s" %
                      lis_file)
            return (False, {})
        
        lis_results = {}
            
        #extract subset of 'lis' file that starts with ***mosfets
        # and ends with the next ***
        lines = EvalUtils.subfile2strings(lis_file, '**** mosfets','***')
        if len(lines) == 0:
            log.debug("_extractLisResults failed: '**** mosfets' section "
                      "was not found")
            return (False, {})
            

        #strip leading whitespace
        lines = [string.lstrip(line) for line in lines]

        #extract (ordered) list of transistor names            
        device_names = []
        for line in lines:
            if line[:len('element')] == 'element':
                #'token' examples are '0:m2', '0:m16', we don't want the 0: part
                tokens = EvalUtils.string2tokens(line[len('element'):])
                device_names += [token[2:] for token in tokens]
        
        #extract list of each measure of interest
        # lis_measure_values maps measure_name : list_of_values, where
        #  the order of values is same as that of transistor names
        lis_measures = {} 
        for measure_name in self.lis_measure_names:
            if measure_name == 'sat_violation': continue #handle 'sat_violation' lower down!
            values = []
            for line in lines:
                if line and (line.split()[0] == measure_name):
                    tokens = EvalUtils.string2tokens(line[len(measure_name):])
                    #resolve: usually it's Saturati, but sometimes Saturat
                    for i,token in enumerate(tokens):
                        if 'Saturat' in token:
                            tokens[i] = 'Saturat'
                    if measure_name == 'region':
                        values += [region_token_to_value[token]
                                   for token in tokens]
                    elif measure_name == 'model':
                        values += tokens
                    else:
                        values += [eval(token) for token in tokens]
            
            #validate
            if len(values) != len(device_names):
                s = 'Found %d values for measure=%s but found %d devices (%s)'% \
                    (len(values), measure_name, len(device_names), device_names)
                log.debug(s)
                return (False, {})

            #output the region measures to help runtime analysis
            if measure_name == 'region':
                s = "'region' measures:"
                for device_name, value in zip(device_names, values):
                    s += "%5s=%16s, " % (device_name, region_value_to_str[value])
                log.debug(s)

            #good, so update lis_measures
            lis_measures[measure_name] = values

        #'sat_violation' is a special measure for"make violation of saturation constraint to be 0.0"
        #Note: this is not as strict / accurate as REGION_SATURATION, which can calculate
        # it precisely.  But it is very useful as an extra guide to the optimizer so that
        # the optimizer can differentiate between nearly-met DOCs and less-met DOCs.
        if 'sat_violation' in self.lis_measure_names:
            #compute values for sat_violation
            values = []
            for (device_index, device_name) in enumerate(device_names):
                vgs = lis_measures['vgs'][device_index]
                vth = lis_measures['vth'][device_index]
                vds = lis_measures['vds'][device_index]
                vdsat = lis_measures['vdsat'][device_index]
                model = lis_measures['model'][device_index]

                if self._isPmos(model):
                    vgs_vth_violation = max(0.0, vgs - vth)
                    if vgs_vth_violation > 0.0:
                        sat_violation = 10.0 * vgs_vth_violation
                    else:
                        vds_vdsat_violation = max(0.0, vds - vdsat)
                        sat_violation = 1.0 * vds_vdsat_violation
                else:
                    vgs_vth_violation = max(0.0, vth - vgs)
                    if vgs_vth_violation > 0.0:
                        sat_violation = 10.0 * vgs_vth_violation
                    else:
                        vds_vdsat_violation = max(0.0, vdsat - vds)
                        sat_violation = 1.0 * vds_vdsat_violation

                values.append(sat_violation)

            #fill values
            lis_measures['sat_violation'] = values

            #output 
            s = "'sat_violation'  :"
            for device_name, value in zip(device_names, values):
                s += "%5s=%16s, " % (device_name, "%.5e" % value)
            log.debug(s)
        
        #update lis_results
        for (device_index, device_name) in enumerate(device_names):
            for measure_name in self.lis_measure_names:
                lis_name = 'lis' + '__' + device_name + '__' + measure_name
                lis_value = lis_measures[measure_name][device_index]
                lis_results[lis_name] = lis_value

        #extract subset of 'lis' file that starts with '  ******   pole/zero analysis'
        # and ends with ' ***** constant factor'
        lines = EvalUtils.subfile2strings(lis_file, '  ******   pole/zero analysis',
                                          ' ***** constant factor')
 
        # now fill the real values if present
        if len(lines) == 0:
            log.info("_extractLisResults failed: '**** pole/zero analysis' section "
                      "was not found")                            
        else:
            # there are pole-zero analysis results

            #strip leading whitespace
            lines = [string.lstrip(line) for line in lines]
            
            # find the start of the poles section
            poles_start_idx=0
            for line in lines:
                if line[:5] == 'poles':
                    break
                else:
                    poles_start_idx += 1
                    
            poles_stop_idx=poles_start_idx+3
            for line in lines[poles_start_idx+3:]:
                if len(line)==0 or not line[0] in ['0','1','2','3','4','5','6','7','8','9','-']:
                    break
                else:
                    poles_stop_idx += 1
                                     
            zeros_start_idx=poles_stop_idx
            for line in lines[poles_stop_idx:]:
                if line[:5] == 'zeros':
                    break
                else:
                    zeros_start_idx += 1
                    
            zeros_stop_idx=zeros_start_idx+3
            for line in lines[zeros_start_idx+3:]:
                if len(line)==0 or not line[0] in ['0','1','2','3','4','5','6','7','8','9','-']:
                    break
                else:
                    zeros_stop_idx += 1
            
            nb_poles=poles_stop_idx - poles_start_idx - 3
            nb_zeros=zeros_stop_idx - zeros_start_idx - 3
                            
            if nb_poles > 0:
            
                # the poles are extracted
                for n in range(0,nb_poles):                
                    line = lines[poles_start_idx+n+3]
                    values = EvalUtils.string2tokens(line)
                    
                    if (len(values) < 4):
                        log.debug("_extractLisResults failed: 'values' bad: %s" % str(values))
                        return (False, {})
                    
                    lis_name = 'lis' + '__pole' + str(n) + '__real'
                    lis_results[lis_name] = eval(values[2])
                    lis_name = 'lis' + '__pole' + str(n) + '__imag'
                    lis_results[lis_name] = eval(values[3])

                    try:
                        fr=((eval(values[2])**2 + eval(values[3])**2)**0.5)
                    except:
                        log.debug("_extractLisResults failed: 'values' bad: %s" % str(values))
                        return (False, {})
                        
                    
                    if fr == 0:
                        lis_name = 'lis' + '__pole' + str(n) + '__fr'
                        lis_results[lis_name] = 'failed'
                        lis_name = 'lis' + '__pole' + str(n) + '__zeta'
                        lis_results[lis_name] = 'failed'
                    else:
                        lis_name = 'lis' + '__pole' + str(n) + '__fr'
                        lis_results[lis_name] = fr
                        lis_name = 'lis' + '__pole' + str(n) + '__zeta'
                        lis_results[lis_name] = eval(values[2]) / fr
                        
            if nb_zeros > 0:
                # the zeros are extracted
                for n in range(0,nb_zeros):
                    line = lines[zeros_start_idx+n+3]
                    values = EvalUtils.string2tokens(line)
                    
                    if (len(values) < 4):
                        log.debug("_extractLisResults failed: 'values' bad: %s" % str(values))
                        return (False, {})
                    
                    lis_name = 'lis' + '__zero' + str(n) + '__real'
                    lis_results[lis_name] = values[2]
                    lis_name = 'lis' + '__zero' + str(n) + '__imag'
                    lis_results[lis_name] = values[3]
                    
                    try:
                        fr=((eval(values[2])**2 + eval(values[3])**2)**0.5)
                    except:
                        log.debug("_extractLisResults failed: 'values' bad: %s" % str(values))
                        return (False, {})
                    
                    if fr == 0:
                        lis_name = 'lis' + '__zero' + str(n) + '__fr'
                        lis_results[lis_name] = 'failed'
                        lis_name = 'lis' + '__zero' + str(n) + '__zeta'
                        lis_results[lis_name] = 'failed'
                    else:
                        lis_name = 'lis' + '__zero' + str(n) + '__fr'
                        lis_results[lis_name] = fr
                        lis_name = 'lis' + '__zero' + str(n) + '__zeta'
                        lis_results[lis_name] = eval(values[2]) / fr

        # nothing else to extract                
        return (True, lis_results)
    
    def _extractTFResults(self,lis_file):
        """
        @description
          Helper file for simulate().
          Extracts the .tf analysis results from a .lis file
        @arguments
          lis_file -- string -- should end in '.lis'
        @return
          success -- bool
          tf_results -- dict of 'measure_name' : lis_value
        """
        assert self.metrics_per_outfile.has_key('lis'), 'only call if want lis'
        assert lis_file[-4:] == '.lis', lis_file

        if not os.path.exists(lis_file):
            log.debug("_extractTFResults failed: couldn't find file: %s" % lis_file)
            return (False, {})
            
        #extract subset of 'lis' file that starts with ****     small-signal transfer characteristics
        # and ends with the next ***
        lines = EvalUtils.subfile2strings(lis_file, '****     small-signal transfer characteristics','***')
        if len(lines) == 0:
            log.debug("_extractLisResults failed: '****     small-signal transfer characteristics' "
                      "section was not found")
            return (False, {})
        
        tf_results={}
        for line in lines:
            if ')/' in line:
                tokens = line.split()
                tf_results[tokens[0]]=eval(tokens[-1])              
        return (True, tf_results)

    def _filesExist(self, filenames):
        for filename in filenames:
            if not os.path.exists(filename):
                return False
        return True

    def _isPmos(self, instance_model_name):
        """Returns True if the instance_model_name is a PMOS (vs. NMOS).
        Detection is based on whether the first character after the ':' is p or n.
        
        Examples:
        '0:p_18_mm' returns True
        '0:n_18_mm' returns False
        'foo' raises an exception.
        """
        max_i = len(instance_model_name) - 1
        for (char_i, char) in enumerate(instance_model_name):
            if (char == ':') and (char_i < max_i):
                next_char = instance_model_name[char_i+1]
                if next_char in ['p','P']:
                    return True
                elif next_char in ['n','N']:
                    return False
                break
        raise AssertionError("Could not detect if pmos from '%s'" % instance_model_name)

    def _badSimResults(self):
        """
        @description

          Returns (sim_results, lis_results, waveforms) where each entry in
          the tuple is 'bad'.  A bad 'sim_results' is a
          dict of metric_name : BAD_METRIC_VALUE whereas
          bad lis_results are merely an empty dict.
          
        @arguments

          <<none>>
        
        @return

          tuple -- see description
    
        @exceptions
    
        @notes

          Does not have an entry in sim_results for DOCs
        """
        sim_results = {}
        for metric_name in self.metricNames():
            if metric_name != DOCS_METRIC_NAME:
                sim_results[metric_name] = BAD_METRIC_VALUE
        return (sim_results, {}, {})

    def createFullNetlist(self, toplevel_embedded_part, scaled_opt_point, variation_data):
        """
        @description

          Builds up a full netlist having the following components:
          -design
          -env variables and test structures
          -model info
             
        @arguments

          design_netlist -- string -- describes the design.
          scaled_opt_point -- Point object --
          variation_data -- None or (RndPoint, EnvPoint, DevicesSetup) --
        
        @return

          full_netlist -- string -- simulation-ready netlist
    
        @exceptions
    
        @notes
        """
        #condition inputs
        (rnd_point, env_point, devices_setup) = variation_data
        
        #preconditions
        if AGGR_TEST:
            assert isinstance(toplevel_embedded_part, EmbeddedPart)
            assert isinstance(scaled_opt_point, Point)
            assert scaled_opt_point.is_scaled
            assert isinstance(rnd_point, RndPoint)
            assert isinstance(env_point, EnvPoint)
            #assert isinstance(devices_setup, DevicesSetup) #don't check, to avoid import dependencies...
            assert devices_setup is not None                   #...but at least have a low-dependency check

        #main work...
        toplevel_embedded_part.functions = scaled_opt_point        
        (design_netlist, models_string) = toplevel_embedded_part.spiceNetlistStr(
            annotate_bb_info=False, add_infostring=False, variation_data=variation_data,
            models_too=True)

        #build up s
        s = [] 
        
        s += ['\n*SPICE netlist, auto-generated by Simulator.createFullNetlist()']
        s += ['\n']

        s += ['\n*------Design---------' ]
        s += ['\n' + design_netlist]
        s += ['\n']

        s += ['\n*------Env Variables and Test Fixture---------' ]
        s += ['\n']
        for (envvar_name, envvar_val) in env_point.items():
            s += ['\n.param %s = %5.3e' % (envvar_name, envvar_val)]
        s += ['\n' + self.test_fixture_string]
        s += ['\n']
 
        s += ['\n*------Simulator Options---------' ]
        s += ['\n' + self.simulator_options_string]
        s += ['\n' ]

        s += ['\n*------Models---------' ]
        s += ['\n' + models_string]
        s += ['\n' ]

        s += ['\n.end' ]
        s += ['\n']
        
        netlist = "".join(s)
        return netlist

    def __str__(self):
        """
        @description

          Override str()
          
        """ 
        s = ''
        s += 'Simulator={'
        #s += '; # xxx=%s' % xxx
        s += ' /Simulator}'
        return s
