import subprocess
import math
#1 Define the event class with slots
class Event:
	__slots__ = 'case_id', 'activity', 'timestamp', 'predecessor'

	def __init__(self, id, a, ts, p = None):
		self.case_id = id
		self.activity = a
		self.timestamp = ts
		self.predecessor = p


#2. Read the event log
#TODO:
#Ask for logfile
#Print the head of the log
#subprocess.call(["head", "log1.csv"])
#Ask about header
#Ask about delimiter
#Ask for case id column number
#Ask for activity name column number
#Ask for timestamp column number

log = []
f = open('log1.csv', 'r')

for line in f:
	line = line.strip()
	if len(line) == 0:
		continue
	parts = line.split(';')
#	if parts[0] not in log:
#		log[parts[0]] = []
	log.append(Event(parts[0], parts[1], parts[3]))

log.sort(key = lambda event: event.timestamp)

#3. Define the predecessor of each event in a trace

last_event = {}

for event in log:
	if event.case_id in last_event:
		event.predecessor = last_event[event.case_id]
	last_event[event.case_id] = event

print("Case ID, Activity, Timestamp, Predecessor")
for event in log:
	print(",".join([event.case_id, event.activity, event.timestamp, (event.predecessor.activity if event.predecessor else "-")]))

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
		self.name = activity+ "Type" + str(c)
		self.sequence = []
		self.predecessor = predecessor
		self.successors = []
		self.c = c
		self.j = predecessor.j + 1
		self.label = "<" + "s" + "<sup>" + str(c) + "</sup>" + "<sub>" + str(self.j) + "</sub>" + ">"

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
		print("Adding activity type: "+node.name)
		self.nodes.sort(key = lambda node: (node.c, node.j))
		return node

	#node [label=""];
	def draw(self):
		dot_string = """digraph G {
	rankdir=LR;
	node [shape=circle];
	subgraph Rel1 {
"""
		for node in self.nodes:
			if node != self.root:
				dot_string += "\t\t" + node.name + " [label=" + node.label + "];\n"
				dot_string += "\t\t"+node.predecessor.name+" -> "+node.name+" [label=" + node.activity + "]" + ";\n"
		dot_string += "\t}"
		dot_string += """\n\tsubgraph Rel2 {
		edge [dir=none]
		node [shape=rectangle]
"""
		for node in self.nodes:
			if node != self.root:
				dot_string += "\t\t" + "\"" + ",".join([ event.activity + event.case_id for event in node.sequence]) + "\""
				dot_string += " [label=<" + ",".join([event.activity + "<sup>" + event.case_id + "</sup>" for event in node.sequence]) + ">];\n"
				dot_string += "\t\t"+node.name+" -> "+"\""+",".join([ event.activity+event.case_id for event in node.sequence])+"\";\n"
		dot_string += "\t}\n"
		dot_string += "}"
		return dot_string

#5. Build the graph
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

print("DOT specification:")
print(pa.draw())

my_spec=open('t.gv', 'w')
my_spec.write(pa.draw())
my_spec.close()
print("Saved DOT specification to t.gv")

subprocess.call(["dot", "-Tpng", "t.gv", "-o", "t.png"])
print("Saved graph to t.png")

#6. Calculate the graph complexity measure
graph_complexity = math.log(len(pa.nodes)-1) * (len(pa.nodes)-1)
for i in range(1,pa.c+1):
	#print(graph_complexity)
	e = len([AT for AT in pa.nodes if AT.c == i])
	graph_complexity += math.log(e)*e

print("graph complexity = "+str(graph_complexity))

log_complexity = math.log(len(log))*len(log)
for i in range(1,pa.c+1):
	#print(log_complexity)
	e = 0
	for AT in pa.nodes:
		if AT.c == i:
			e += len(AT.sequence)
	log_complexity += math.log(e)*e

print("log complexity = "+str(log_complexity))

#7. Show prefixes of each state
print("Prefixes:")
for node in pa.nodes:
	if node != pa.root:
		print ("s"+"^"+str(node.c)+"_"+str(node.j) + ":" + node.getPrefix())
