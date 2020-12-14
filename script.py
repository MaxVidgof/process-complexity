#!/usr/bin/env python3.7
import subprocess
import math
import pm4py
from argparse import ArgumentParser
import statistics
from datetime import datetime
from dateutil import rrule
from lempel_ziv_complexity import lempel_ziv_complexity, lempel_ziv_decomposition # pip install lempel_ziv_complexity
import time

#0. Read the command line arguments
parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file", help="input log file")
parser.add_argument("-d", "--dot", dest="dot", help="create dot specs", default=False, action="store_true")
parser.add_argument("-g", "--graph", dest="graph", help="draw a graph", default=False, action="store_true")
parser.add_argument("-p", "--prefix", dest="prefix", help="output prefix for each state", default=False, action="store_true")
parser.add_argument("-v", "--verbose", dest="verbose", help="verbose output", default=False, action="store_true")
parser.add_argument("-m", "--measures", dest="measures", help="calculate other complexity measures", default=False, action="store_true")
parser.add_argument("--hide-event", dest="subg", help="hide event nodes, keep only activity types", default=True, action="store_false")
parser.add_argument("--png", dest="png", help="draw the graph in PNG (may fail if the graph is too big", default=False, action="store_true")
parser.add_argument("-e", "--exponential-forgetting", dest="ex_k", help="coefficient for exponential forgetting", default=1)

args = parser.parse_args()

times = [0] * 7

#1 Define the event class with slots

event_id_counter = 0
class Event:
	__slots__ = 'case_id', 'activity', 'timestamp', 'predecessor', 'event_id'

	def __init__(self, id, a, ts, p = None):
		self.case_id = id
		self.activity = a
		self.timestamp = ts
		self.predecessor = p
		global event_id_counter
		event_id_counter += 1
		self.event_id = event_id_counter


#2. Read the event log
#TODO:
#Text wrap for event nodes
#Automatically set font size
s = time.perf_counter() # start reading the log
if args.file==None:
	raise Exception("No file specified")

if args.file.split(".")[-1]=="xes":
	input_file = args.file  #"/home/max/Downloads/Sepsis Cases - Event Log.xes"
	from pm4py.objects.log.importer.xes import importer as xes_importer
	pm4py_log = xes_importer.apply(input_file)
elif (args.file.split(".")[-1]=="csv"):
	subprocess.call(["head", args.file])
	i_h = input("Does the file have a header? [y/N]:") or "n"
	h = 0 if i_h != "n" else None
	i_d = input("What is the delimiter? [,]:") or ","
	i_c = input("What is the column number of case ID? [0]:")
	i_c = 0 if i_c == "" else int(i_c)
	i_a = input("What is the column number of activity name? [1]:")
	i_a = 1 if i_a == "" else int(i_a)
	i_t = input("What is the column number of timestamp? [2]:")
	i_t = 2 if i_t == "" else int(i_t)

	import pandas as pd
	from pm4py.objects.conversion.log import converter as log_converter
	from pm4py.objects.log.util import dataframe_utils

	log_csv = pd.read_csv(args.file, sep=i_d, header=h)
	log_csv.rename(columns={i_c:'case', i_a:'concept:name', i_t:'time:timestamp'}, inplace=True)
	for col in log_csv.columns:
		if isinstance(col, int):
			log_csv.rename(columns={col:'column'+str(col)}, inplace=True)
	log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
	log_csv = log_csv.sort_values('time:timestamp')
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case'}
	print(log_csv)
	pm4py_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

	#log = []
	#f = open(args.file, 'r')

	#if i_h=="y":
	#	header = next(f, None)

	#for line in f:
	#	line = line.strip()
	#	if len(line) == 0:
	#		continue
	#	parts = line.split(i_d.strip())
	#	if parts[0] not in log:
	#		log[parts[0]] = []
	#	log.append(Event(parts[i_c], parts[i_a], datetime.fromisoformat(parts[i_t])))
else:
	raise Exception("File type not recognized, should be xes or csv")

log = []
for trace in pm4py_log:
	for event in trace:
		log.append(Event(trace.attributes['concept:name'], event['concept:name'].strip().replace(" ", "_"), event['time:timestamp']))


log.sort(key = lambda event: event.timestamp)

times[0] = time.perf_counter()-s

#2.1 Define the time span
timespan = (log[-1].timestamp - log[0].timestamp).total_seconds()
last_timestamp=log[-1].timestamp
#3. Define the predecessor of each event in a trace
#Time not measred for 2.1
s = time.perf_counter()

last_event = {}

for event in log:
	if event.case_id in last_event:
		event.predecessor = last_event[event.case_id]
	last_event[event.case_id] = event
#at this point last_event.keys will include all case IDs in the log

if(args.verbose):
	print("Case ID, Activity, Timestamp, Predecessor, Event ID")
	for event in log:
		print(",".join([event.case_id, event.activity, str(event.timestamp), (event.predecessor.activity + "-" + str(event.predecessor.event_id) if event.predecessor else "-"), str(event.event_id)]))

times[1] = time.perf_counter()-s

#4. Define the classes for the prefix automaton

class Node:
	def __init__(self, name):
		self.name = name
		self.successors = []
		self.c = 0
		self.j = 0

#class Edge:
#	def __init__(self, activity, start, end):
#		self.activity = activity
#		self.start = start
#		self.end = end

class ActivityType(Node):
	def __init__(self, activity, predecessor, c):
		self.activity = activity
		self.sequence = []
		self.predecessor = predecessor
		self.successors = []
		self.c = c
		self.j = predecessor.j + 1
		self.label = "<" + "s" + "<sup>" + str(c) + "</sup>" + "<sub>" + str(self.j) + "</sub>" + ">"
		self.name = activity+ "Type" + str(c) + "_" + str(self.j)

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
		#self.edges = []
		self.activity_types = dict()

	def addNode(self, activity, predecessor, c):
		node = ActivityType(activity, predecessor, c)
		node.predecessor.successors.append(node)
		self.nodes.append(node)
		if activity not in self.activity_types:
			self.activity_types[activity] = []
		self.activity_types[activity].append(node)
		if(args.verbose):
			print("Adding activity type: "+node.name)
		self.nodes.sort(key = lambda node: (node.c, node.j))
		return node

	#node [label=""];
	def draw(self):
		dot_string = """digraph G {
	rankdir=LR;
	node [shape=circle fontsize=30.0];
	edge [fontsize=30.0];
	subgraph Rel1 {
"""
		for node in self.nodes:
			if node != self.root:
				dot_string += "\t\t" + node.name + " [label=" + node.label + "];\n"
				dot_string += "\t\t"+node.predecessor.name+" -> "+node.name+" [label=" + node.activity + "]" + ";\n"
		dot_string += "\t}"
		if(args.subg):
			dot_string += """\n\tsubgraph Rel2 {
		edge [dir=none]
		node [shape=rectangle]
"""
			for node in self.nodes:
				if node != self.root:
					dot_string += "\t\t" + "\"" + str(node.sequence[0].event_id) + "\""  #"_".join([ str(event.event_id) for event in node.sequence]) + "\"" #",".join #event.activity+str(event.event_id)
					dot_string += " [label=<" + ",".join([event.activity + "<sup>" + event.case_id + "</sup>" + "<sub>" + str(event.event_id) + "</sub>" for event in node.sequence]) + ">];\n"
					dot_string += "\t\t"+node.name+" -> "+"\""+ str(node.sequence[0].event_id) + "\";\n"  #"_".join([ str(event.event_id) for event in node.sequence])+"\";\n" #event.activity+str(event.event_id)
			dot_string += "\t}\n"
		dot_string += "}"
		return dot_string

#4.1 Define flattening function
def flatten(in_list):
	out_list = []
	for item in in_list:
		if isinstance(item, list):
			out_list.extend(flatten(item))
		else:
			out_list.append(item)
	return out_list

#5. Build the graph
s = time.perf_counter()
if(args.verbose):
	print("Building the prefix automaton...")
pa = Graph()

for event in log:
	if(event.predecessor):
		#Find the ActivityType of the predecessor event
		pred_activity_type = None
		if event.predecessor != pa.root:
			for node in pa.activity_types[event.predecessor.activity]:
				if event.predecessor in node.sequence:
					pred_activity_type = node
			if pred_activity_type == None:
				print("Error")
	else:
		pred_activity_type = pa.root

	#Check if the predecessor's ActivityType has a succeeding ActivityType that we need
	current_activity_type = None
	for node in pred_activity_type.successors:
		if node.activity == event.activity:
			current_activity_type = node
	if current_activity_type==None:
		if len(pred_activity_type.successors) > 0:
			pa.c += 1
			curr_c = pa.c
		else:
			curr_c = pred_activity_type.c if pred_activity_type != pa.root else pa.c
		current_activity_type = pa.addNode(event.activity, pred_activity_type, curr_c)
	current_activity_type.sequence.append(event)
	#else:
	#	pa.c += 1
	#	current_activity_type = pa.addNode(event.activity, pa.root, pa.c)
	#	current_activity_type.sequence.append(event)

times[2] = time.perf_counter()-s

s = time.perf_counter()

if(args.dot):
	print("DOT specification:")
	print(pa.draw())

if(args.graph):
	my_spec=open('t.gv', 'w')
	my_spec.write(pa.draw())
	my_spec.close()
	print("Saved DOT specification to t.gv")
	subprocess.call(["dot", "-Tpng" if args.png else "-Tsvg", "t.gv", "-o", "t."+("png" if args.png else "svg")]) #-Tpng
	print("Saved graph to t."+("png" if args.png else "svg"))

times[3] = time.perf_counter()-s

s = time.perf_counter()

#6. Calculate complexity measures
#6.1. Log complexity metrics
if(args.measures):
	m_magnitude = len(log) # magnitude - number of events in a log
	print("Magnitude: " + str(m_magnitude))
	m_support = len(pm4py_log) # support - number of traces in a log
	print("Support: " + str(m_support))

	event_classes = {}
	df_relations = {}
	trace_lengths = {}
	for case_id in last_event.keys():
		event_classes[case_id] = set()
		df_relations[case_id] = set()
		trace_lengths[case_id] = 0
	for event in log:
		event_classes[event.case_id].add(event.activity)
		if event.predecessor:
			df_relations[event.case_id].add((event.predecessor.activity, event.activity))
		trace_lengths[event.case_id] += 1
	m_variety = len(set.union(*[event_classes[case_id] for case_id in event_classes]))
	print("Variety: " + str(m_variety))
	m_lod = statistics.mean([len(event_classes[case_id]) for case_id in event_classes])
	print("Level of detail: " + str(m_lod))

	time_granularities = {}
	for event in log:
		if event.predecessor:
			d = (event.timestamp - event.predecessor.timestamp).total_seconds()
			if event.case_id not in time_granularities or d < time_granularities[event.case_id]:
				time_granularities[event.case_id] = d
				if(args.verbose):
					print("Updating time granularity for trace " + str(event.case_id) + ": " + str(d) + " seconds. Event " + str(event.activity) + " (" + str(event.event_id) + ")")
	m_time_granularity = statistics.mean([time_granularities[case_id] for case_id in time_granularities])
	print("Time granularity: " + str(m_time_granularity) + " (seconds)")
	m_structure = 1 - ((len(set.union(*[df_relations[case_id] for case_id in df_relations])))/(m_variety**2))
	print("Structure: " + str(m_structure))

	#Jan's bit vectors
	#hashmap = {}

	#commented out for a moment
	#affinities = []
	#for case_id_1 in df_relations:
	#	for case_id_2 in df_relations:
	#		if case_id_1 != case_id_2:
	#			# note that affinities are only calculated for distinct traces
	#			# also note that affinity is calculated twice for each pair of traces, e.g. (1,2) and (2,1),
	#			# this does not affect the result but does double the calculation time
	#			# however, it prevents possible errors when comparing event IDs that are strungs
	#			affinity = len(set.intersection(df_relations[case_id_1], df_relations[case_id_2]))/len(set.union(df_relations[case_id_1], df_relations[case_id_2]))
	#			affinities.append(affinity)
	#m_affinity = statistics.mean(affinities)
	#print("Affinity: " + str(m_affinity))
	m_trace_length = {}
	m_trace_length["min"] = min([trace_lengths[case_id] for case_id in trace_lengths])
	m_trace_length["avg"] = statistics.mean([trace_lengths[case_id] for case_id in trace_lengths])
	m_trace_length["max"] = max([trace_lengths[case_id] for case_id in trace_lengths])
	print("Trace length: " + "/".join([str(m_trace_length[key]) for key in ["min", "avg", "max"]])  + " (min/avg/max)")
	print("Distinct traces: " + str((pa.c/m_support)*100) + "%")
	from pm4py.algo.filtering.log.variants import variants_filter
	var = variants_filter.get_variants(pm4py_log)
	#print("Here: "+str(pa.c/m_support==len(var)/len(pm4py_log)))
	print("Graph partitions: "+str(pa.c))
	print("Variants: "+str(len(var)))
	#Pentland's Process Complexity
	#Calculated as number of variants - variants that include loops
	#m_simple_paths = pa.c #all paths
	#for node in pa.nodes:
	#	if len(node.successors) == 0:
	#		p = node.getPrefix()
	#		if len(p.split(",")) > len(set(p.split(","))):
	#			m_simple_paths -= 1
	#print("?Number of simple paths: " + str(m_simple_paths)) #remove this method as it refers to process _model_ not _log_
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
	a_mean = n_transitions/(m_variety**2)
	for i in range(len(action_network)):
		for j in range(len(action_network[i])):
			m_dev_from_rand += ((action_network[i][j]-a_mean)/n_transitions)**2
	m_dev_from_rand = math.sqrt(m_dev_from_rand)
	m_dev_from_rand = 1 - m_dev_from_rand
	print("Deviation from random: " + str(m_dev_from_rand))
	#print("L-Z dec: " + ";".join(lempel_ziv_decomposition("".join([event.activity for event in log]))))
	m_l_z = lempel_ziv_complexity(tuple([event.activity for event in log]))
	print("Lempel-Ziv complexity: " + str(m_l_z))
	# Haerem and Pentland task complexity
	#TODO: of which task???
	# root node as 0th task in all variants,
	m_pentland_task = 0
	for n in pa.nodes:
		if len(n.successors)==0:
			m_pentland_task += n.j
	print("Pentland's Task complexity: "+str(m_pentland_task))

times[4] = time.perf_counter()-s
#6.2. Calculate the graph complexity measure
print("---Entropy measures---")

def graph_complexity(pa):
	graph_complexity = math.log(len(pa.nodes)-1) * (len(pa.nodes)-1)
	normalize = graph_complexity
	for i in range(1,pa.c+1):
		#print(graph_complexity)
		e = len([AT for AT in pa.nodes if AT.c == i])
		graph_complexity -= math.log(e)*e

	return graph_complexity,(graph_complexity/normalize)

# graph entropy
s = time.perf_counter()
var_ent = graph_complexity(pa)
print("Variant entropy: "+str(var_ent[0]))
print("Normalized variant entropy: "+str(var_ent[1]))
times[5] = time.perf_counter()-s

def log_complexity(pa, forgetting = None, k=1):
	normalize = len(log)*math.log(len(log))
	if(not forgetting):
		length = 0
		for AT in flatten(pa.activity_types.values()):
			length += len(AT.sequence)
		log_complexity = math.log(length)*length
		for i in range(1,pa.c+1):
			#print(log_complexity)
			e = 0
			for AT in pa.nodes:
				if AT.c == i:
					e += len(AT.sequence)
			log_complexity -= math.log(e)*e

		return log_complexity,(log_complexity/normalize)
	elif(forgetting=="linear"):
		#log complexity with linear forgetting
		log_complexity_linear = 0
		for AT in flatten(pa.activity_types.values()):
			for event in AT.sequence:
				log_complexity_linear += 1 - (last_timestamp - event.timestamp).total_seconds()/timespan

		log_complexity_linear = math.log(log_complexity_linear) * log_complexity_linear

		for i in range(1,pa.c+1):
			e = 0
			for AT in pa.nodes:
				if AT.c == i:
					for event in AT.sequence:
						e += 1 - (last_timestamp - event.timestamp).total_seconds()/timespan
			log_complexity_linear -= math.log(e)*e

		return log_complexity_linear,(log_complexity_linear/normalize)

	elif(forgetting=="exp"):
		#log complexity with exponential forgetting
		log_complexity_exp = 0
		for AT in flatten(pa.activity_types.values()):
			for event in AT.sequence:
				log_complexity_exp += math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k)

		log_complexity_exp = math.log(log_complexity_exp) * log_complexity_exp

		for i in range(1,pa.c+1):
			e = 0
			for AT in pa.nodes:
				if AT.c == i:
					for event in AT.sequence:
						e += math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k)
			log_complexity_exp -= math.log(e)*e
		return log_complexity_exp,(log_complexity_exp/normalize)
	else:
		return None,None

# log complexity
s = time.perf_counter()

seq_ent = log_complexity(pa)

print("Sequence entropy: "+str(seq_ent[0]))
print("Normalized equence entropy: "+str(seq_ent[1]))

seq_ent_lin = log_complexity(pa, "linear")

print("Sequence entropy with linear forgetting: "+str(seq_ent_lin[0]))
print("Normalized sequence entropy with linear forgetting: "+str(seq_ent_lin[1]))

seq_ent_exp = log_complexity(pa, "exp",float(args.ex_k))
#print("log complexity with exponential forgetting (k=1): "+str(log_complexity(pa, "exp")))
print("Sequence entropy with exponential forgetting (k="+str(args.ex_k)+"): "+str(seq_ent_exp[0]))
print("Normalized sequence entropy with exponential forgetting (k="+str(args.ex_k)+"): "+str(seq_ent_exp[1]))

times[6] = time.perf_counter()-s
#6.3
# TODO calculate monthly 
#for dt in rrule.rrule(rrule.MONTHLY, dtstart=log[0].timestamp, until=log[-1].timestamp):
#	log_rule = list(filter(lambda event: event.timestamp.year==dt.year and event.timestamp.month==dt.month, log))
#7. Show prefixes of each state
if(args.prefix):
	print("Prefixes:")
	for node in pa.nodes:
		if node != pa.root:
			print ("s"+"^"+str(node.c)+"_"+str(node.j) + ":" + node.getPrefix())

print("Time measurements:")
print("Reading log: "+str(times[0])+" seconds")
print("Defining predecessors: "+str(times[1])+" seconds")
print("Building prefix automaton: "+str(times[2])+" seconds")
print("Drawing the graph: "+str(times[3])+" seconds")
print("Calculating log complexity measures: "+str(times[4])+" seconds")
print("Calculating graph entropy: "+str(times[5])+" seconds")
print("Calculating sequence complexity: "+str(times[6])+" seconds")

