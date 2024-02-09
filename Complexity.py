#!/usr/bin/env python3.8
# Necessary imports
import os
import pandas as pd
import subprocess
import math
import pm4py
from argparse import ArgumentParser
import statistics
from datetime import datetime
from dateutil import rrule
from dateutil.relativedelta import relativedelta
import calendar
from lempel_ziv_complexity import lempel_ziv_complexity, lempel_ziv_decomposition # pip install lempel_ziv_complexity
import time
from BitVector import BitVector
import matplotlib.pyplot as plt
import functools
from itertools import count
import operator

# Classes
class Event:
	__slots__ = 'case_id', 'activity', 'timestamp', 'predecessor', 'event_id'
	#_ids = count(0)
	_counter = 0

	def __init__(self, id, a, ts, p = None):
		self.case_id = id
		self.activity = a
		self.timestamp = ts
		self.predecessor = p
		#global event_id_counter
		#event_id_counter += 1
		Event._counter += 1
		self.event_id = Event._counter
		#self.event_id = next(self._ids)

class Node:
	def __init__(self, name):
		self.name = name
		self.successors = {}
		self.c = 0
		self.j = 0

class ActivityType(Node):
	def __init__(self, activity, predecessor, c, accepting=True):
		self.activity = activity
		self.sequence = []
		self.predecessor = predecessor
		self.successors = {}
		self.c = c
		self.j = predecessor.j + 1
		self.label = "<" + "s" + "<sup>" + str(c) + "</sup>" + "<sub>" + str(self.j) + "</sub>" + ">"
		self.name = activity+ "Type" + str(c) + "_" + str(self.j)
		self.accepting = accepting

	def getPrefix(self):
		prefix = self.activity
		if self.predecessor.name != "root":
			prefix = self.predecessor.getPrefix() + "," + prefix
		return prefix

class Graph:
	def __init__(self):
		self.c = 1
		self.root = Node("root")
		self.nodes = [self.root]
		self.activity_types = dict()
		# not part of formal definition,
		# implementation decision to improve construction speed
		self.last_at = dict()
		# implementatio decision to improve entropy measurement speed
		# must always be constructed anew before use
		self.c_index = {}

	def addNode(self, activity, predecessor, c, accepting=True, verbose=False):
		node = ActivityType(activity, predecessor, c, accepting)
		node.predecessor.successors[activity] = node
		self.nodes.append(node)
		if activity not in self.activity_types:
			self.activity_types[activity] = []
		self.activity_types[activity].append(node)
		if(verbose):
			print("Adding activity type: "+node.name)
		# Extremely expensive -> affecting all
		#self.nodes.sort(key = lambda node: (node.c, node.j))
		return node

	#node [label=""];
	def draw(self, subg=False, accepting=False):
		dot_string = """digraph G {
	rankdir=LR;
	node [shape=circle fontsize=30.0];
	edge [fontsize=30.0];
	subgraph Rel1 {
"""
		for node in self.nodes:
			if node != self.root:
				dot_string += "\t\t\"" + node.name + "\" [label=" + node.label + "];\n"
				dot_string += "\t\t\""+node.predecessor.name+"\" -> \""+node.name+"\" [label=\"" + node.activity + "\"]" + ";\n"
		dot_string += "\t}"
		if(subg):
			dot_string += """\n\tsubgraph Rel2 {
		edge [dir=none]
		node [shape=rectangle]
"""
			for node in self.nodes:
				if node != self.root:
					dot_string += "\t\t" + "\"" + str(node.sequence[0].event_id) + "\""  #"_".join([ str(event.event_id) for event in node.sequence]) + "\"" #",".join #event.activity+str(event.event_id)
					dot_string += " [label=<" + ",".join([event.activity + "<sup>" + str(event.case_id) + "</sup>" + "<sub>" + str(event.event_id) + "</sub>" for event in node.sequence]) + ">];\n"
					dot_string += "\t\t\""+node.name+"\" -> "+"\""+ str(node.sequence[0].event_id) + "\";\n"  #"_".join([ str(event.event_id) for event in node.sequence])+"\";\n" #event.activity+str(event.event_id)
			dot_string += "\t}\n"
		if(accepting):
			for node in self.nodes:
				if node != self.root and hasattr(node, "accepting") and node.accepting:
					dot_string += "\n\t\""+node.name+"\" [shape=doublecircle]"
		dot_string += "\n}"
		return dot_string

	def get_first_timestamp(self):
		return min([min([event.timestamp for event in node.sequence]) for node in self.nodes if node != self.root])
	def get_last_timestamp(self):
		return max([max([event.timestamp for event in node.sequence]) for node in self.nodes if node != self.root])
	def get_timespan(self):
		return (self.get_last_timestamp() - self.get_first_timestamp()).total_seconds()

	def to_plain_log(self):
		return sorted(flatten([node.sequence for node in self.nodes if node != self.root]), key = lambda event: event.timestamp)
	def to_pm4py_log(self):
		traces = {}
		for event in sorted(self.to_plain_log(), key=lambda event: (event.case_id, event.timestamp)):
			if event.predecessor and (event.case_id not in traces):
				raise Exception("Could not convert to PM4Py log")
			if event.case_id not in traces:
				traces[event.case_id] = pm4py.objects.log.obj.Trace()
			pm4py_event = pm4py.objects.log.obj.Event()
			pm4py_event["concept:name"] = event.activity
			pm4py_event["time:timestamp"] = event.timestamp
			traces[event.case_id].append(pm4py_event)
		return pm4py.objects.log.obj.EventLog(traces.values())

#2. Read the event log

# Functions

def extract_base_filename(filename):
	base_filename = ".".join(os.path.basename(filename).split('.')[0:-1])
	return base_filename

# Takes in file name
# Returns pm4py_log
def generate_pm4py_log(filename=None, verbose=False):
	if filename==None:
		raise Exception("No file specified")


	if filename.split(".")[-1]=="xes":
		input_file = filename  #"/home/max/Downloads/Sepsis Cases - Event Log.xes"
		from pm4py.objects.log.importer.xes import importer as xes_importer
		pm4py_log = xes_importer.apply(input_file)
	elif (filename.split(".")[-1]=="csv"):
		subprocess.call(["head", filename])
		i_h = input("Does the file have a header? [y/N]:") or "n"
		h = 0 if i_h != "n" else None
		i_d = input("What is the delimiter? [,]:") or ","
		i_c = input("What is the column number of case ID? [0]:")
		i_c = 0 if i_c == "" else int(i_c)
		i_a = input("What is the column number of activity name? [1]:")
		i_a = 1 if i_a == "" else int(i_a)
		i_t = input("What is the column number of timestamp? [2]:")
		i_t = 2 if i_t == "" else int(i_t)

		from pm4py.objects.conversion.log import converter as log_converter
		from pm4py.objects.log.util import dataframe_utils

		log_csv = pd.read_csv(filename, sep=i_d, header=h)
		log_csv.rename(columns={log_csv.columns[i_c]:'case', log_csv.columns[i_a]:'concept:name', log_csv.columns[i_t]:'time:timestamp'}, inplace=True)
		for col in log_csv.columns:
			if isinstance(col, int):
				log_csv.rename(columns={col:'column'+str(col)}, inplace=True)
		log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
		log_csv = log_csv.sort_values('time:timestamp')
		parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case'}
		if(verbose):
			print(log_csv)
		pm4py_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

	else:
		raise Exception("File type not recognized, should be xes or csv")

	return pm4py_log

# Generate log frpm pm4py_log
def generate_log(pm4py_log,verbose=False):
	log = []
	for trace in pm4py_log:
		for event in trace:
			log.append(Event(trace.attributes['concept:name'], event['concept:name'], event['time:timestamp']))

	log.sort(key = lambda event: event.timestamp)

	last_event = {}
	for event in log:
		if event.case_id in last_event:
			event.predecessor = last_event[event.case_id]
		last_event[event.case_id] = event
	#at this point last_event.keys will include all case IDs in the log

	if(verbose):
		print("Case ID, Activity, Timestamp, Predecessor, Event ID")
		for event in log:
			print(",".join([str(event.case_id), event.activity, str(event.timestamp), (event.predecessor.activity + "-" + str(event.predecessor.event_id) if event.predecessor else "-"), str(event.event_id)]))

	return log

def get_timespan(log):
	timespan = (log[-1].timestamp - log[0].timestamp).total_seconds()
	return timespan

def get_last_timestamp(log):
	last_timestamp=log[-1].timestamp
	return last_timestamp

# Define flattening function
def flatten(in_list):
	out_list = []
	for item in in_list:
		if isinstance(item, list):
			out_list.extend(flatten(item))
		else:
			out_list.append(item)
	return out_list

# Build the EPA
def build_graph(log, verbose=False, accepting=False):
	if len(log)==0:
		raise Exception("Cannot build EPA from an empty log")
	if(verbose):
		print("Building the prefix automaton...")

	pa = Graph()
	#last_at = {}
	pa = add_events_to_graph(pa, log, verbose=verbose)
	if accepting:
		pa = mark_accepting_states(pa)
	return pa

def find_predecessor(event, pa, verbose=False):
	if(event.predecessor):
		#Find the ActivityType of the predecessor event
		if event.predecessor != pa.root:
			if (event.case_id in pa.last_at and event.predecessor in pa.last_at[event.case_id].sequence): # doesnt affect speed
				pred_activity_type = pa.last_at[event.case_id]
			else:
				raise Exception("Error")
	else:
		pred_activity_type = pa.root

	return pred_activity_type

def add_event_to_graph(event, pa, verbose=False):
	pred_activity_type = find_predecessor(event, pa, verbose=verbose)
	#Check if the predecessor's ActivityType has a succeeding ActivityType that we need
	current_activity_type = None
	if(event.activity in pred_activity_type.successors): #keys
		current_activity_type = pred_activity_type.successors[event.activity]
	else:
		if len(pred_activity_type.successors) > 0:
			pa.c += 1
			curr_c = pa.c
		else:
			curr_c = pred_activity_type.c if pred_activity_type != pa.root else pa.c
		current_activity_type = pa.addNode(event.activity, pred_activity_type, curr_c, verbose=verbose)

	current_activity_type.sequence.append(event)
	pa.last_at[event.case_id] = current_activity_type

	return current_activity_type

def add_events_to_graph(pa, log, verbose=False):
	for event in log:
		add_event_to_graph(event, pa, verbose=verbose)
	# Moved expensive sort here -- it is done only once when all nodes are added
	pa.nodes.sort(key = lambda node: (node.c, node.j))
	return pa

def mark_accepting_states(pa):
	for node in pa.nodes:
	        if node != pa.root and (len(node.successors)==0 or len(node.sequence) > sum([len(successor.sequence) for successor in node.successors.values()])):
                	node.accepting = True
        	else:
	                node.accepting = False


	return pa

def draw_graph(pa, base_filename, subg, png=False, accepting=False):
	my_spec=open(base_filename+'.gv', 'w')
	my_spec.write(pa.draw(subg, accepting))
	my_spec.close()
	print("Saved DOT specification to "+base_filename+".gv")
	subprocess.call(["dot", "-Tpng" if png else "-Tsvg", base_filename+".gv", "-o", base_filename+"."+("png" if png else "svg")])
	print("Saved graph to "+base_filename+"."+("png" if png else "svg"))


# Calculate complexity measures
# Log complexity metrics
# Measures depending solely on EPA
def measure_pentland_task(pa, quiet=False, verbose=False):
	# root node as 0th task in all variants,
	m_pentland_task = 0
	for n in pa.nodes:
		if len(n.successors)==0:
			m_pentland_task += n.j

	if not quiet:
		print("Pentland's Task complexity: "+str(m_pentland_task))
	return m_pentland_task

# Measures depending solely on log
def measure_time_granularity(log, quiet=False, verbose=False):
	time_granularities = {}
	for event in log:
		if event.predecessor:
			d = (event.timestamp - event.predecessor.timestamp).total_seconds()
			if event.case_id not in time_granularities or d < time_granularities[event.case_id]:
				time_granularities[event.case_id] = d
				if(verbose):
					print("Updating time granularity for trace " + str(event.case_id) + ": " + str(d) + " seconds. Event " + str(event.activity) + " (" + str(event.event_id) + ")")
	try:
		m_time_granularity = statistics.mean([time_granularities[case_id] for case_id in time_granularities])
	except statistics.StatisticsError as err:
		if not time_granularities:
			# If all cases have at most one event
			# Then there are no time differences between events
			# Setting time granularity to 0
			m_time_granularity = 0
		else:
			raise err
	if not quiet:
		print("Time granularity: " + str(m_time_granularity) + " (seconds)")
	return m_time_granularity

def measure_lempel_ziv(log, quiet=False, verbose=False):
	m_l_z = lempel_ziv_complexity(tuple([event.activity for event in log]))
	if not quiet:
		print("Lempel-Ziv complexity: " + str(m_l_z))
	return m_l_z

def measure_magnitude(log, quiet=False, verbose=False):
	m_magnitude = len(log) # magnitude - number of events in a log
	if not quiet:
		print("Magnitude: " + str(m_magnitude))
	return m_magnitude

# Auxiliary variables for measures depending on pm4py_log
def aux_variants(pm4py_log):
	from pm4py.algo.filtering.log.variants import variants_filter
	var = variants_filter.get_variants(pm4py_log)
	return var

def aux_event_classes(pm4py_log):
	event_classes = {}
	for trace in pm4py_log:
		event_classes[trace.attributes['concept:name']] = set()
		for event in trace:
			event_classes[trace.attributes['concept:name']].add(event['concept:name'])
	return event_classes

def aux_hashmap(pm4py_log):
	event_classes=aux_event_classes(pm4py_log)

	hashmap = {}
	evts = list(set.union(*[event_classes[case_id] for case_id in event_classes]))
	num_act = len(evts)
	i=0
	for event in evts:
		for event_follows in evts:
			hashmap[(event, event_follows)]=i
			i += 1
	return hashmap, num_act

def aux_aff(pm4py_log):
	var = aux_variants(pm4py_log)
	hashmap, num_act = aux_hashmap(pm4py_log)

	aff = {}
	for variant in var: #keys
		aff[variant] = [0,0]
		aff[variant][0] = len(var[variant])
		aff[variant][1] = BitVector(size=num_act**2)
		for i in range(1, len(var[variant][0])):
			aff[variant][1][hashmap[(var[variant][0][i-1]['concept:name'], var[variant][0][i]['concept:name'])]] = 1
	return aff

# Measures depending solely on pm4py_log (and auxiliary variables)
def measure_support(pm4py_log, quiet=False, verbose=False):
	m_support = len(pm4py_log) # support - number of traces in a log
	if not quiet:
		print("Support: " + str(m_support))
	return m_support

def measure_trace_length(pm4py_log, quiet=False, verbose=False):
	m_trace_length = {}
	m_trace_length["min"] = min([len(trace) for trace in pm4py_log])
	m_trace_length["avg"] = statistics.mean([len(trace) for trace in pm4py_log])
	m_trace_length["max"] = max([len(trace) for trace in pm4py_log])
	if not quiet:
		print("Trace length: " + "/".join([str(m_trace_length[key]) for key in ["min", "avg", "max"]])  + " (min/avg/max)")
	return m_trace_length

def measure_variety(pm4py_log, quiet=False, verbose=False):
	event_classes = aux_event_classes(pm4py_log)

	m_variety = len(set.union(*[event_classes[case_id] for case_id in event_classes]))
	if not quiet:
		print("Variety: " + str(m_variety))
	return m_variety

def measure_level_of_detail(pm4py_log, quiet=False, verbose=False):
	event_classes = aux_event_classes(pm4py_log)

	m_lod = statistics.mean([len(event_classes[case_id]) for case_id in event_classes])
	if not quiet:
		print("Level of detail: " + str(m_lod))
	return m_lod

def measure_affinity(pm4py_log, quiet=False, verbose=False):
	aff = aux_aff(pm4py_log)

	m_affinity=0
	for v1 in aff:
		for v2 in aff:
			if(v1!=v2):
				if(verbose):
					print(v1+"-"+v2)
				overlap = (aff[v1][1] & aff[v2][1]).count_bits_sparse()
				union = (aff[v1][1] | aff[v2][1]).count_bits_sparse()
				if(verbose):
					 print(str(overlap)+"/"+str(union))
				if(union==0 and overlap==0):
					m_affinity += 0
				else:
					m_affinity += (overlap/union)*aff[v1][0]*aff[v2][0]
			else:
				#relative overlap = 1
				m_affinity += aff[v1][0]*(aff[v1][0]-1)
	try:
		m_affinity /= (sum([aff[v][0] for v in aff]))*(sum([aff[v][0] for v in aff])-1)
	except ZeroDivisionError:
		m_affinity = float("nan")
	if not quiet:
		print("Affinity: "+str(m_affinity))
	return m_affinity

def measure_structure(pm4py_log, quiet=False, verbose=False):
	aff = aux_aff(pm4py_log)
	m_variety = measure_variety(pm4py_log, quiet=True)

	m_structure = 1 - (((functools.reduce(lambda a,b: a | b, [bv[1] for bv in aff.values()])).count_bits_sparse())/(m_variety**2))
	if not quiet:
		print("Structure: " + str(m_structure))
	return m_structure

def measure_distinct_traces(pm4py_log, quiet=False, verbose=False):
	var = aux_variants(pm4py_log)
	m_support = measure_support(pm4py_log, quiet=True)

	m_distinct_traces =  (len(var)/m_support)*100
	if not quiet:
		print("Distinct traces: " + str(m_distinct_traces) + "%")
	return m_distinct_traces

def measure_pentland_process(pm4py_log, quiet=False, verbose=False):
	aff = aux_aff(pm4py_log)
	m_variety = measure_variety(pm4py_log, quiet=True)

	hashmap, num_act = aux_hashmap(pm4py_log)
	event_classes = aux_event_classes(pm4py_log)

	evts = list(set.union(*[event_classes[case_id] for case_id in event_classes]))

	# Only shows if a DF relation between the activities exists
	#binary_transition_matrix = [[0 for j in range(num_act)] for i in range(num_act)]
	vector = functools.reduce(operator.or_,[v[1] for v in aff.values()])
	#for i,act1 in enumerate(evts):
	#	for j,act2 in enumerate(evts):
	#		binary_transition_matrix[i][j] = vector[hashmap[(act1,act2)]]

	# Formula: 10^(0.08*(1+e-v))
	# where v = number of 'vertices', i.e. activities
	# e = number of 'edges', i.e. non-zero cells in transition matrix
	#v = len(evts)
	#e = sum(sum(binary_transition_matrix[i][j] for j in range(num_act)) for i in range(num_act))
	v = m_variety
	e = vector.count_bits_sparse()

	m_pentland_process = 10**(0.08*(1+e-v))
	if verbose:
		print(f"v: {v}")
		print(f"e: {e}")
	if not quiet:
		print(f"Pentland's process complexity: {m_pentland_process}")

# Measures depending on both log and pm4py_log
def measure_deviation_from_random(log, pm4py_log, quiet=False, verbose=False):
	event_classes = aux_event_classes(pm4py_log)
	m_variety = measure_variety(pm4py_log, quiet=True)

	action_network = []
	for i in range (m_variety):
		action_network.append([])
		for j in range(m_variety):
			action_network[i].append(0)
	evt_lexicon = set.union(*[event_classes[case_id] for case_id in event_classes])
	evt_lexicon = list(evt_lexicon)
	n_transitions = 0
	for event in log:
		if event.predecessor:
			action_network[evt_lexicon.index(event.predecessor.activity)][evt_lexicon.index(event.activity)] += 1
			n_transitions +=1
	m_dev_from_rand = 0
	if n_transitions > 0:
		a_mean = n_transitions/(m_variety**2)
		for i in range(len(action_network)):
			for j in range(len(action_network[i])):
				m_dev_from_rand += ((action_network[i][j]-a_mean)/n_transitions)**2
		m_dev_from_rand = math.sqrt(m_dev_from_rand)
		m_dev_from_rand = 1 - m_dev_from_rand
	else:
		# If there are no transitions
		# set deviation from random to NA
		m_dev_from_rand = None
	if not quiet:
		print("Deviation from random: " + str(m_dev_from_rand))
	return m_dev_from_rand

# Nicer interface for complexity measurements
def perform_measurements(desired_measurements, log=None, pm4py_log=None, pa=None, quiet=True,verbose=False):
	def check_measure(*measures):
		return any(element in desired_measurements for element in measures)

	if not quiet:
		print("Selected measures: " + ",".join(desired_measurements))

	#TODO: further optimizations
	# Generating auxiliary variables
	#if check_measure("deviation_from_random","variety","level_of_detail","affinity","structure","all"):
	#	measure_times['event_classes'] = time.perf_counter()-m

	#if check_measure("affinity","structure","distinct_traces","all"):
	#	measure_times['var'] = time.perf_counter()-m

	#if check_measure("affinity","structure","all"):
	#	measure_times['hashmap'] = time.perf_counter()-m

	measurements = {}

	if check_measure("magnitude","all"):
		measurements['Magnitude'] = measure_magnitude(log, quiet, verbose)

	if check_measure("support","distinct_traces","all"):
		measurements['Support'] = measure_support(pm4py_log, quiet, verbose)

	if check_measure("variety","deviation_from_random","structure","all"):
		measurements['Variety'] = measure_variety(pm4py_log, quiet, verbose) # event_classes

	if check_measure("level_of_detail","all"):
		measurements['Level of detail'] = measure_level_of_detail(pm4py_log, quiet, verbose) # event_classes

	if check_measure("time_granularity","all"):
		measurements['Time granularity'] = measure_time_granularity(log, quiet, verbose)

	if check_measure("structure","all"):
		measurements['Structure'] = measure_structure(pm4py_log, quiet, verbose) # event_classes, var, hashmap, variety

	if check_measure("affinity","all"):
		measurements['Affinity'] = measure_affinity(pm4py_log, quiet, verbose) #event_classes, var, hashmap

	if check_measure("trace_length","all"):
		measurements['Trace length'] = measure_trace_length(pm4py_log, quiet, verbose)

	if check_measure("distinct_traces","all"):
		measurements['Distinct traces'] = measure_distinct_traces(pm4py_log, quiet, verbose) # var, support

	if check_measure("deviation_from_random","all"):
		measurements['Deviation from random'] = measure_deviation_from_random(log, pm4py_log, quiet, verbose) # event_classes, variety

	if check_measure("lempel-ziv","all"):
		measurements['Lempel-Ziv complexity'] = measure_lempel_ziv(log, quiet, verbose)

	# Haerem and Pentland task complexity
	if check_measure("pentland","all"):
		measurements["Pentland's task complexity"] = measure_pentland_task(pa, quiet, verbose)

	# Pentland process complexity
	if check_measure("pentland_process","all"):
		measurements["Pentland's process complexity"] = measure_pentland_process(pm4py_log, quiet, verbose)

	return measurements

#  EPA complexity measures
def create_c_index(pa):
	c_index = {}
	for node in pa.nodes:
		if node.c not in c_index:
			c_index[node.c] = []
		c_index[node.c].append(node)

	for key in c_index:
		c_index[key].sort(key = lambda node: node.j)

	return c_index

# Variant entropy
def graph_complexity(pa):
	# Always (re-)create c_index instead of assuming it exists and is up-to-date
	pa.c_index = create_c_index(pa)
	graph_complexity = math.log(len(pa.nodes)-1) * (len(pa.nodes)-1)
	normalize = graph_complexity
	#for i in range(1,pa.c+1):
	for i in list(pa.c_index.keys())[1:]: # ignore c=0 that is always present
		#print(graph_complexity)
		#e = len([AT for AT in pa.nodes if AT.c == i])
		e = len(pa.c_index[i])
		graph_complexity -= math.log(e)*e
	try:
		return graph_complexity,(graph_complexity/normalize)
	except ZeroDivisionError:
		if graph_complexity==0:
			return 0,0
		else:
			raise Exception("Error")
# Sequence entropy
def log_complexity(pa, forgetting = None, k=1):
	# Always (re-)create c_index instead of assuming it exists and is up-to-date
	pa.c_index = create_c_index(pa)
	#normalize = len(log)*math.log(len(log))
	#removed due to dependency on log
	normalize = sum([len(AT.sequence) for AT in flatten(pa.activity_types.values())])
	normalize = normalize*math.log(normalize)
	if(not forgetting):
		length = 0
		for AT in flatten(pa.activity_types.values()):
			length += len(AT.sequence)
		log_complexity = math.log(length)*length
		#for i in range(1,pa.c+1):
		for i in list(pa.c_index.keys())[1:]: # ignore c=0 that is always present
			#print(log_complexity)
			#e = 0
			#for AT in pa.nodes:
			#	if AT.c == i:
			#		e += len(AT.sequence)
			e = sum([len(AT.sequence) for AT in pa.c_index[i]])
			log_complexity -= math.log(e)*e

		try:
			return log_complexity,(log_complexity/normalize)
		except ZeroDivisionError:
			if log_complexity==0:
				return 0,0
			else:
				raise Exception("Error")

	elif(forgetting=="linear"):
		#log complexity with linear forgetting
		last_timestamp = pa.get_last_timestamp()
		timespan = pa.get_timespan()
		log_complexity_linear = 0
		for AT in flatten(pa.activity_types.values()):
			for event in AT.sequence:
				try:
					log_complexity_linear += 1 - (last_timestamp - event.timestamp).total_seconds()/timespan
				except ZeroDivisionError:
					# timespan==0
					# meaning the period has only 1 event
					# its weight is conceptually both 0 and 1
					# choose weight of 1
					log_complexity_linear += 1

		log_complexity_linear = math.log(log_complexity_linear) * log_complexity_linear

		#for i in range(1,pa.c+1):
		for i in list(pa.c_index.keys())[1:]: # ignore c=0 that is always present
			e = 0
			#for AT in pa.nodes:
			#	if AT.c == i:
			#		for event in AT.sequence:
			for AT in pa.c_index[i]:
				for event in AT.sequence:
					try:
						e += 1 - (last_timestamp - event.timestamp).total_seconds()/timespan # used to be 1 more tab
					except ZeroDivisionError:
						# timespan==0
						# meaning the period has only 1 event
						# its weight is conceptually both 0 and 1
						# choose weight of 1
						e += 1
			try:
				log_complexity_linear -= math.log(e)*e
			except ValueError:
				if e==0:
					# if a partition only contains one event
					# that happens to be the first event
					# then it just weights 0 and does not affect entropy
					pass
				else:
					print(f"e= {e}")
					import sys
					sys.exit("E")

		try:
			return log_complexity_linear,(log_complexity_linear/normalize)
		except ZeroDivisionError:
			if log_complexity_linear==0:
				return 0,0
			else:
				raise Exception("Error")


	elif(forgetting=="exp"):
		#log complexity with exponential forgetting
		last_timestamp = pa.get_last_timestamp()
		timespan = pa.get_timespan()
		log_complexity_exp = 0
		for AT in flatten(pa.activity_types.values()):
			for event in AT.sequence:
				try:
					log_complexity_exp += math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k)
				except ZeroDivisionError:
					# timespan==0
					# meaning the period has only 1 event
					# its weight is conceptually both 0 and 1
					# choose weight of 1
					log_complexity_exp += 1


		log_complexity_exp = math.log(log_complexity_exp) * log_complexity_exp

		#for i in range(1,pa.c+1):
		for i in list(pa.c_index.keys())[1:]: # ignore c=0 that is always present
			e = 0
			#for AT in pa.nodes:
			#	if AT.c == i:
			#		for event in AT.sequence:
			for AT in pa.c_index[i]:
				for event in AT.sequence:
					try:
						e += math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k) # used to be 1 more tab
					except ZeroDivisionError:
						# timespan==0
						# meaning the period has only 1 event
						# its weight is conceptually both 0 and 1
						# choose weight of 1
						e += 1
			log_complexity_exp -= math.log(e)*e
		try:
			return log_complexity_exp,(log_complexity_exp/normalize)
		except ZeroDivisionError:
			if log_complexity_exp==0:
				return 0,0
			else:
				raise Exception("Error")

	else:
		return None,None

# EPA Complexity changing over time
# Variant II - gradually increasing complexity
def monthly_complexity(pa, end):
	active_nodes = [node for node in pa.nodes if node != pa.root and len([event for event in node.sequence if event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)])>0] #note: comparing UTC timestamps
	graph_complexity = math.log(len(active_nodes)) * (len(active_nodes))
	normalize1 = graph_complexity
	normalize2 = math.log(len(pa.nodes)-1) * (len(pa.nodes)-1)
	for i in range(1,pa.c+1):
		e = len([AT for AT in active_nodes if AT.c == i])
		if (e > 0):
			graph_complexity -= math.log(e)*e

	return graph_complexity,(graph_complexity/normalize1),(graph_complexity/normalize2)


def calculate_variant_entropy(pa, start_timestamp, end_timestamp, figure=False, base_filename=None, verbose=False):
	if figure and not base_filename:
		raise Exception("Filename required to store plots")
	if(verbose):
		print("Monthly complexity")
	dates=[]
	complexities=[]
	complexities_norm1=[]
	complexities_norm2=[]

	for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_timestamp, until=end_timestamp+relativedelta(months=1)):
		dates.append(dt)
		if(verbose):
			print(str(calendar.month_name[dt.month])+" "+str(dt.year))
		complexity, complexity_norm1, complexity_norm2 = monthly_complexity(pa, datetime(dt.year, dt.month,calendar.monthrange(int(dt.year), int(dt.month))[1], 23,59,59 ))
		if(verbose):
			print("Complexity: "+str(complexity))
			print("Complexity_norm1: "+str(complexity_norm1))
			print("Complexity_norm2: "+str(complexity_norm2))
		complexities.append(complexity)
		complexities_norm1.append(complexity_norm1)
		complexities_norm2.append(complexity_norm2)


	df = pd.DataFrame()
	df["Date"]=dates
	df["Variant entropy"]=complexities
	df["Variant entropy(Active)"]=complexities_norm1
	df["Variant entropy(All)"]=complexities_norm2

	if(figure):
		plt.figure(figsize=(1920,1080))
		df.plot("Date", "Variant entropy")
		plt.savefig(base_filename+"_Entropy_growth.png")
		plt.figure(figsize=(1920,1080))
		df.plot("Date", ["Variant entropy(Active)", "Variant entropy(All)"])
		plt.savefig(base_filename+"_Entropy_growth_normalized.png")
		plt.close('all')

	return df

def monthly_log_complexity(pa, end, forgetting=None, k=1):
	normalize1 = sum([len([event for event in AT.sequence if event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)]) for AT in flatten(pa.activity_types.values())])
	normalize1 = normalize1*math.log(normalize1)
	normalize2 = sum([len(AT.sequence) for AT in flatten(pa.activity_types.values())])
	normalize2 = normalize2*math.log(normalize2)

	if(not forgetting):
		length = 0
		for AT in flatten(pa.activity_types.values()):
			length += len([event for event in AT.sequence if event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)])
		log_complexity = math.log(length)*length

		for i in range(1,pa.c+1):
			e = 0
			for AT in pa.nodes:
				if AT.c == i:
					e += len([event for event in AT.sequence if event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)])
			if(e>0):
				log_complexity -= math.log(e)*e
		return log_complexity,(log_complexity/normalize1), (log_complexity/normalize2)

	elif(forgetting=="linear"):
		#log complexity with linear forgetting
		first_timestamp = pa.get_first_timestamp()
		last_timestamp = pa.get_last_timestamp()
		timespan = pa.get_timespan()

		curr_timespan = ((end+first_timestamp.utcoffset()).replace(tzinfo=first_timestamp.tzinfo) - first_timestamp).total_seconds()
		log_complexity_linear1 = 0
		log_complexity_linear2 = 0
		for AT in flatten(pa.activity_types.values()):
			for event in AT.sequence:
				if(event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)):
					log_complexity_linear1 += 1 - ((end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo) - event.timestamp).total_seconds()/curr_timespan 
					log_complexity_linear2 += 1 - (last_timestamp - event.timestamp).total_seconds()/timespan

		log_complexity_linear1 = math.log(log_complexity_linear1) * log_complexity_linear1
		log_complexity_linear2 = math.log(log_complexity_linear2) * log_complexity_linear2

		for i in range(1,pa.c+1):
			e1 = 0
			e2 = 0
			for AT in pa.nodes:
				if AT.c == i:
					for event in AT.sequence:
						if(event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)):
							e1 += 1 - ((end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo) - event.timestamp).total_seconds()/curr_timespan
							e2 += 1 - (last_timestamp - event.timestamp).total_seconds()/timespan
			if(e1>0):
				log_complexity_linear1 -= math.log(e1)*e1
			if(e2>0):
				log_complexity_linear2 -= math.log(e2)*e2

		return log_complexity_linear1,(log_complexity_linear1/normalize1),(log_complexity_linear1/normalize2),log_complexity_linear2,(log_complexity_linear2/normalize1),(log_complexity_linear2/normalize2)

	elif(forgetting=="exp"):
		#log complexity with exponential forgetting
		first_timestamp = pa.get_first_timestamp()
		last_timestamp = pa.get_last_timestamp()
		timespan = pa.get_timespan()

		curr_timespan = ((end+first_timestamp.utcoffset()).replace(tzinfo=first_timestamp.tzinfo) - first_timestamp).total_seconds()
		log_complexity_exp1 = 0
		log_complexity_exp2 = 0
		for AT in flatten(pa.activity_types.values()):
			for event in AT.sequence:
				if(event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)):
					log_complexity_exp1 += math.exp((-((end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo) - event.timestamp).total_seconds()/curr_timespan)*k)
					log_complexity_exp2 += math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k)

		log_complexity_exp1 = math.log(log_complexity_exp1) * log_complexity_exp1
		log_complexity_exp2 = math.log(log_complexity_exp2) * log_complexity_exp2

		for i in range(1,pa.c+1):
			e1 = 0
			e2=0
			for AT in pa.nodes:
				if AT.c == i:
					for event in AT.sequence:
						if(event.timestamp <= (end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo)):
							e1 += math.exp((-((end+event.timestamp.utcoffset()).replace(tzinfo=event.timestamp.tzinfo) - event.timestamp).total_seconds()/curr_timespan)*k)
							e2 += math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k)
			if(e1>0):
				log_complexity_exp1 -= math.log(e1)*e1
			if(e2>0):
				log_complexity_exp2 -= math.log(e2)*e2
		return log_complexity_exp1,(log_complexity_exp1/normalize1),(log_complexity_exp1/normalize2),log_complexity_exp2,(log_complexity_exp2/normalize1),(log_complexity_exp2/normalize2)
	else:
		return None

def calculate_sequence_entropy(pa, start_timestamp, end_timestamp, forgetting=None, k=1 ,figure=False, base_filename=None, verbose=False):
	if forgetting and not (forgetting in ["linear", "exp"]):
		raise Exception("Forgetting can only be linear or exponential")

	if figure and not base_filename:
		raise Exception("Filename required to store plots")

	if(verbose):
		if not forgetting:
			print("Monthly log entropy")
		elif(forgetting=="linear"):
			print("Monthly entropy with linear forgetting")
		elif(forgetting=="exp"):
			print("Monthly entropy with exponential forgetting(k="+str(k)+")")

	dates=[]
	if not forgetting:
		complexities=[]
		complexities_norm1=[]
		complexities_norm2=[]
	else:
		complexities1=[]
		complexities1_norm1=[]
		complexities1_norm2=[]
		complexities2=[]
		complexities2_norm1=[]
		complexities2_norm2=[]


	for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_timestamp, until=end_timestamp+relativedelta(months=1)):
		dates.append(dt)
		if(verbose):
			print(str(calendar.month_name[dt.month])+" "+str(dt.year))
		if not forgetting:
			complexity, complexity_norm1, complexity_norm2 = monthly_log_complexity(pa, datetime(dt.year, dt.month,calendar.monthrange(int(dt.year), int(dt.month))[1], 23,59,59 ))
			if(verbose):
				print("Complexity: "+str(complexity))
				print("Complexity_norm1: "+str(complexity_norm1))
				print("Complexity_norm2: "+str(complexity_norm2))
			complexities.append(complexity)
			complexities_norm1.append(complexity_norm1)
			complexities_norm2.append(complexity_norm2)
		else:
			if forgetting=="linear":
				complexity1, complexity1_norm1, complexity1_norm2,complexity2, complexity2_norm1, complexity2_norm2 = monthly_log_complexity(pa, datetime(dt.year, dt.month,calendar.monthrange(int(dt.year), int(dt.month))[1], 23,59,59 ), forgetting="linear")
			elif forgetting=="exp": #TODO: add k
				complexity1, complexity1_norm1, complexity1_norm2,complexity2, complexity2_norm1, complexity2_norm2 = monthly_log_complexity(pa, datetime(dt.year, dt.month,calendar.monthrange(int(dt.year), int(dt.month))[1], 23,59,59 ), forgetting="exp", k=k)
			else:
				raise Exception("Forgetting can only be linear or exponential")

			if(verbose):
				print("Complexity1: "+str(complexity1))
				print("Complexity1_norm1: "+str(complexity1_norm1))
				print("Complexity1_norm2: "+str(complexity1_norm2))
			complexities1.append(complexity1)
			complexities1_norm1.append(complexity1_norm1)
			complexities1_norm2.append(complexity1_norm2)
			if(verbose):
				print("Complexity2: "+str(complexity2))
				print("Complexity2_norm1: "+str(complexity2_norm1))
				print("Complexity2_norm2: "+str(complexity2_norm2))
			complexities2.append(complexity2)
			complexities2_norm1.append(complexity2_norm1)
			complexities2_norm2.append(complexity2_norm2)

	df = pd.DataFrame()
	df["Date"]=dates
	if not forgetting:
		df["Sequence entropy"]=complexities
		df["Sequence entropy(Active)"]=complexities_norm1
		df["Sequence entropy(All)"]=complexities_norm2
	else:
		df["Sequence entropy(Rel)"]=complexities1
		df["Sequence entropy(Rel,Active)"]=complexities1_norm1
		df["Sequence entropy(Rel,All)"]=complexities1_norm2
		df["Sequence entropy(Abs)"]=complexities2
		df["Sequence entropy(Abs,Active)"]=complexities2_norm1
		df["Sequence entropy(Abs,All)"]=complexities2_norm2

	if(figure):
		if not forgetting:
			plt.figure(figsize=(1920,1080))
			df.plot("Date", "Sequence entropy")
			plt.savefig(base_filename+"_Log_entropy_growth.png")
			plt.figure(figsize=(1920,1080))
			df.plot("Date", ["Sequence entropy(Active)", "Sequence entropy(All)"])
			plt.savefig(base_filename+"_Log_entropy_growth_normalized.png")
		elif(forgetting=="linear"):
			plt.figure(figsize=(1920,1080))
			df.plot("Date", "Sequence entropy(Rel)")
			plt.savefig(base_filename+"_Log_entropy_growth_linear_relative.png")
			plt.figure(figsize=(1920,1080))
			df.plot("Date", ["Sequence entropy(Rel,Active)", "Sequence entropy(Rel,All)"])
			plt.savefig(base_filename+"_Log_entropy_growth_linear_relative_normalized.png")
			plt.figure(figsize=(1920,1080))
			df.plot("Date", "Sequence entropy(Abs)")
			plt.savefig(base_filename+"_Log_entropy_growth_linear_absolute.png")
			plt.figure(figsize=(1920,1080))
			df.plot("Date", ["Sequence entropy(Abs,Active)", "Sequence entropy(Abs,All)"])
			plt.savefig(base_filename+"_Log_entropy_growth_linear_absolute_normalized.png")
		elif(forgetting=="exp"):
			plt.figure(figsize=(1920,1080))
			df.plot("Date", "Sequence entropy(Rel)")
			plt.savefig(base_filename+"_Log_entropy_growth_exp_relative.png")
			plt.figure(figsize=(1920,1080))
			df.plot("Date", ["Sequence entropy(Rel,Active)", "Sequence entropy(Rel,All)"])
			plt.savefig(base_filename+"_Log_entropy_growth_exp_relative_normalized.png")
			plt.figure(figsize=(1920,1080))
			df.plot("Date", "Sequence entropy(Abs)")
			plt.savefig(base_filename+"_Log_entropy_growth_exp_absolute.png")
			plt.figure(figsize=(1920,1080))
			df.plot("Date", ["Sequence entropy(Abs,Active)", "Sequence entropy(Abs,All)"])
			plt.savefig(base_filename+"_Log_entropy_growth_exp_absolute_normalized.png")
		plt.close('all')

	return df

###############


if __name__ == "__main__":
	# Read the command line arguments
	parser = ArgumentParser()
	parser.add_argument("-f", "--file", dest="file", help="input log file")
	parser.add_argument("-d", "--dot", dest="dot", help="create dot specs", default=False, action="store_true")
	parser.add_argument("-g", "--graph", dest="graph", help="draw a graph", default=False, action="store_true")
	parser.add_argument("-p", "--prefix", dest="prefix", help="output prefix for each state", default=False, action="store_true")
	parser.add_argument("-v", "--verbose", dest="verbose", help="verbose output", default=False, action="store_true")
	parser.add_argument("-m", "--measures", dest="measures", help="calculate other complexity measures", default=[], action="append", choices=["magnitude","support","variety","level_of_detail","time_granularity","structure","affinity","trace_length","distinct_traces","deviation_from_random","lempel-ziv","pentland","all"]) #store_true
	parser.add_argument("--hide-event", dest="subg", help="hide event nodes, keep only activity types", default=True, action="store_false")
	parser.add_argument("--png", dest="png", help="draw the graph in PNG (may fail if the graph is too big", default=False, action="store_true")
	parser.add_argument("-e", "--exponential-forgetting", dest="ex_k", help="coefficient for exponential forgetting", default=1)
	parser.add_argument("-t", dest="change", help="calculate complexity growth over time", default=False,action="store_true")
	parser.add_argument("-a", "--accepting", dest="accepting", help="explicitly mark accepting states", default=False, action="store_true")

	args = parser.parse_args()
	times = {}
	measure_times = {}

	# Read and prepare event log
	base_filename = extract_base_filename(args.file)
	pm4py_log = generate_pm4py_log(args.file, verbose=args.verbose)
	log = generate_log(pm4py_log, verbose=args.verbose)
	pa = build_graph(log, verbose=args.verbose, accepting=args.accepting)

	if(args.dot):
		print("DOT specification:")
		print(pa.draw(args.subg))

	if(args.graph):
		draw_graph(pa, base_filename, args.subg, args.png, args.accepting)
	if(args.measures):
		measurements = perform_measurements(args.measures, log, pm4py_log, pa, quiet=False, verbose=args.verbose)

	print("---Entropy measures---")
	var_ent = graph_complexity(pa)
	print("Variant entropy: "+str(var_ent[0]))
	print("Normalized variant entropy: "+str(var_ent[1]))

	seq_ent = log_complexity(pa)
	print("Sequence entropy: "+str(seq_ent[0]))
	print("Normalized equence entropy: "+str(seq_ent[1]))

	seq_ent_lin = log_complexity(pa, "linear")
	print("Sequence entropy with linear forgetting: "+str(seq_ent_lin[0]))
	print("Normalized sequence entropy with linear forgetting: "+str(seq_ent_lin[1]))

	seq_ent_exp = log_complexity(pa, "exp",float(args.ex_k))
	print("Sequence entropy with exponential forgetting (k="+str(args.ex_k)+"): "+str(seq_ent_exp[0]))
	print("Normalized sequence entropy with exponential forgetting (k="+str(args.ex_k)+"): "+str(seq_ent_exp[1]))

	# Change over time
	if(args.change):
		def show(df): # For backward compatibility
			cols = [col for col in df.columns if col != "Date"]
			for i in range(len(df)):
				dt = df["Date"][i]
				print(str(calendar.month_name[dt.month])+" "+str(dt.year))
				for col in cols:
					print(col+": "+str(df[col][i]))

		variant_entropy_change = calculate_variant_entropy(pa,log[0].timestamp,log[-1].timestamp, figure=True, base_filename=base_filename, verbose=args.verbose)
		show(variant_entropy_change)
		sequence_entropy_change = calculate_sequence_entropy(pa,log[0].timestamp,log[-1].timestamp, figure=True, base_filename=base_filename, verbose=args.verbose)
		show(sequence_entropy_change)
		sequence_entropy_change_linear = calculate_sequence_entropy(pa,log[0].timestamp,log[-1].timestamp, forgetting="linear", figure=True, base_filename=base_filename, verbose=args.verbose)
		show(sequence_entropy_change_linear)
		sequence_entropy_change_exponential = calculate_sequence_entropy(pa,log[0].timestamp,log[-1].timestamp, forgetting="exp", k=float(args.ex_k), figure=True, base_filename=base_filename, verbose=args.verbose)
		show(sequence_entropy_change_exponential)

	# Show prefixes of each state
	if(args.prefix):
		print("Prefixes:")
		for node in pa.nodes:
			if node != pa.root:
				print ("s"+"^"+str(node.c)+"_"+str(node.j) + ":" + node.getPrefix())

#TODO:
#Text wrap for event nodes
#Automatically set font size
# Add args.verbose to all addNode calls

#count time
#s = time.perf_counter()
#times['Defining predecessors'] = time.perf_counter()-s
#times["Building prefix automaton"] = time.perf_counter()-s
#times["Drawing the graph"] = time.perf_counter()-s
#times["Calculating log complexity measures"] = time.perf_counter()-s
#times["Calculating variant entropy"] = time.perf_counter()-s
#times["Calculating sequence entropy"] = time.perf_counter()-s
#times["Calculating sequence entropy with linear forgetting"] = time.perf_counter()-s
#times["Calculating sequence entropy with exponential forgetting"] = time.perf_counter()-s

#print("Time measurements:")
#for k,v in times.items():
#	if(k=="Calculating log complexity measures"):
#		for p,m in measure_times.items():
#			if p not in ["event_classes", "hashmap", "var"]:
#				print(p+": "+str(m)+" seconds")
#	print(k+": "+str(v)+" seconds")
#print("Total: "+str(sum(times.values())))
