import subprocess
import math
import pm4py
from argparse import ArgumentParser

#0. Read the command line arguments
parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file", help="input log file")
parser.add_argument("-d", "--dot", dest="dot", help="create dot specs", default=False, action="store_true")
parser.add_argument("-g", "--graph", dest="graph", help="draw a graph", default=False, action="store_true")
parser.add_argument("-p", "--prefix", dest="prefix", help="output prefix for each state", default=False, action="store_true")
parser.add_argument("-v", "--verbose", dest="verbose", help="verbose output", default=False, action="store_true")
parser.add_argument("--hide-event", dest="subg", help="hide event nodes, keep only activity types", default=True, action="store_false")
parser.add_argument("--png", dest="png", help="draw the graph in PNG (may fail if the graph is too big", default=False, action="store_true")
args = parser.parse_args()
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
if args.file==None:
	raise Exception("No file specified")

if args.file.split(".")[-1]=="xes":
	input_file = args.file  #"/home/max/Downloads/Sepsis Cases - Event Log.xes"
	from pm4py.objects.log.importer.xes import factory as xes_importer
	pm4py_log = xes_importer.apply(input_file)
	log = []
	for trace in pm4py_log:
		for event in trace:
			log.append(Event(trace.attributes['concept:name'], event['concept:name'].strip().replace(" ", "_"), event['time:timestamp']))
elif (args.file.split(".")[-1]=="csv"):
	subprocess.call(["head", args.file])
	i_h = input("Does the file have a header? [y/N]:" or "n")
	i_d = input("What is the delimiter? [,]:") or ","
	i_c = input("What is the column number of case ID? [0]:")
	i_c = 0 if i_c == "" else int(i_c)
	i_a = input("What is the column number of activity name? [1]:")
	i_a = 1 if i_a == "" else int(i_a)
	i_t = input("What is the column number of imestamp? [2]:")
	i_t = 2 if i_t == "" else int(i_t)

	log = []
	f = open(args.file, 'r')

	if i_h=="y":
		header = next(f, None)

	for line in f:
		line = line.strip()
		if len(line) == 0:
			continue
		parts = line.split(i_d.strip())
	#	if parts[0] not in log:
	#		log[parts[0]] = []
		log.append(Event(parts[i_c], parts[i_a], parts[i_t]))
else:
	raise Exception("File type not recognized, should be xes or csv")

log.sort(key = lambda event: event.timestamp)

#3. Define the predecessor of each event in a trace

last_event = {}

for event in log:
	if event.case_id in last_event:
		event.predecessor = last_event[event.case_id]
	last_event[event.case_id] = event

if(args.verbose):
	print("Case ID, Activity, Timestamp, Predecessor, Event ID")
	for event in log:
		print(",".join([event.case_id, event.activity, str(event.timestamp), (event.predecessor.activity + "-" + str(event.predecessor.event_id) if event.predecessor else "-"), str(event.event_id)]))

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

#5. Build the graph
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

#6. Calculate the graph complexity measure
graph_complexity = math.log(len(pa.nodes)-1) * (len(pa.nodes)-1)
for i in range(1,pa.c+1):
	#print(graph_complexity)
	e = len([AT for AT in pa.nodes if AT.c == i])
	graph_complexity -= math.log(e)*e

print("graph complexity = "+str(graph_complexity))

log_complexity = math.log(len(log))*len(log)
for i in range(1,pa.c+1):
	#print(log_complexity)
	e = 0
	for AT in pa.nodes:
		if AT.c == i:
			e += len(AT.sequence)
	log_complexity -= math.log(e)*e

print("log complexity = "+str(log_complexity))

#7. Show prefixes of each state
if(args.prefix):
	print("Prefixes:")
	for node in pa.nodes:
		if node != pa.root:
			print ("s"+"^"+str(node.c)+"_"+str(node.j) + ":" + node.getPrefix())
