#!/usr/bin/env python3.8
import Complexity
from argparse import ArgumentParser

class AttributesContainer:
	def __init__(self, attributes):
		if attributes.keys()=={"trace","event"}:
			self.trace = AttributesContainer(attributes['trace'])
			self.event = AttributesContainer(attributes['event'])
		else:
			for k,v in attributes.items():
				self.__setattr__(k,v)

	def __eq__(self, other):
		if not isinstance(other, AttributesContainer):
			return NotImplemented
		return self.__dict__ == other.__dict__

	def __hash__(self):
		return hash(tuple(sorted(self.__dict__.items(), key = lambda kv: kv[0])))


class EnrichedEvent(Complexity.Event):
	__slots__ = 'case_id', 'activity', 'timestamp', 'predecessor', 'event_id', 'attributes'

	def __init__(self, id, a, ts, p = None, attributes={}):
		super().__init__(id, a, ts, p)
		self.attributes = AttributesContainer(attributes) #{'trace': {}, 'event': {}}

class EnrichedActivityType(Complexity.ActivityType):
	def __init__(self, activity, predecessor, c, attributes, accepting=True):
		super().__init__(activity, predecessor, c, accepting)
		if isinstance(attributes, AttributesContainer):
			self.attributes = attributes
		else:
			self.attributes = AttributesContainer(attributes or {})


class EnrichedGraph(Complexity.Graph):
	def addNode(self, activity, predecessor, c, attributes=None, accepting=True, verbose=False):
		node = EnrichedActivityType(activity, predecessor, c, attributes, accepting)
		node.predecessor.successors[(activity, attributes)] = node #
		self.nodes.append(node)
		if activity not in self.activity_types:
			self.activity_types[activity]=[]
		self.activity_types[activity].append(node)
		if(verbose):
			print("Adding activity type: "+node.name)
		#self.nodes.sort(key = lambda node: (node.c, node.j))
		return node

	def draw(self, subg=False, accepting=False):
		dot_string = super().draw(subg, accepting)
		# remove '}' from string
		dot_string = dot_string[:-1]
		dot_string += """\tsubgraph Rel3 {
		edge [dir=none]
		node [shape=rectangle]
"""
		for node in self.nodes:
			if node != self.root:
				dot_string += "\t\t" + "\"" + node.attributes.__repr__().split("at ",1)[1][:-1] + "\""
				dot_string += " [label=<Trace: " + str(node.attributes.trace.__dict__) + "; Event: " + str(node.attributes.event.__dict__)  + ">];\n"
				dot_string += "\t\t\""+node.name+"\" -> \"" + node.attributes.__repr__().split("at ",1)[1][:-1] + "\";\n"
		dot_string += "\n\t}"
		dot_string += "\n}"

		return dot_string

def generate_enriched_log(pm4py_log, verbose=False):
	log = []
	for trace in pm4py_log:
		for event in trace:
			attributes = {'trace':{}, 'event':{}}
			for attr, value in trace.attributes.items():
				attributes['trace'][attr] = value
			attributes['trace'].pop('concept:name', None)
			for attr, value in event.items():
				attributes['event'][attr] = value
			attributes['event'].pop('concept:name', None)
			attributes['event'].pop('time:timestamp', None)
			evt = EnrichedEvent(trace.attributes['concept:name'], event['concept:name'], event['time:timestamp'], attributes=attributes)
			log.append(evt)

	log.sort(key = lambda event: event.timestamp)

	last_event = {}
	for event in log:
		if event.case_id in last_event:
			event.predecessor = last_event[event.case_id]
		last_event[event.case_id] = event

	if(verbose):
		pass #add later

	return log

def build_enriched_graph(log, verbose=False, accepting=False):
	if len(log)==0:
		raise Exception("Cannot build EPA from an empty log")
	if(verbose):
		print("Building the prefix automaton...")

	pa = EnrichedGraph()
	pa = add_enriched_events_to_graph(pa, log, verbose=verbose)
	if accepting:
		pa = Complexity.mark_accepting_states(pa)
	return pa

def add_enriched_events_to_graph(pa, log, verbose=False):
	for event in log:
		if(event.predecessor):
			#Find the ActivityType of the predecessor event
			if event.predecessor != pa.root:
				if (event.case_id in pa.last_at and event.predecessor in pa.last_at[event.case_id].sequence):
					pred_activity_type = pa.last_at[event.case_id]
				else:
					raise Exception("Error")
		else:
			pred_activity_type = pa.root

		#Check if the predecessor's ActivityType has a succeeding ActivityType that we need
		# Change happens here -- we search for required AT by both activity label and attributes
		current_activity_type = None
		if((event.activity,event.attributes) in pred_activity_type.successors): #keys
			current_activity_type = pred_activity_type.successors[(event.activity,event.attributes)]
		else:
			if len(pred_activity_type.successors) > 0:
				pa.c += 1
				curr_c = pa.c
			else:
				curr_c = pred_activity_type.c if pred_activity_type != pa.root else pa.c
			current_activity_type = pa.addNode(event.activity, pred_activity_type, curr_c, attributes=event.attributes, verbose=verbose)

		current_activity_type.sequence.append(event)
		pa.last_at[event.case_id] = current_activity_type

	pa.nodes.sort(key = lambda node: (node.c, node.j))
	return pa

def list_attributes(pm4py_log):
	return {'trace': set.union(*[set(trace.attributes.keys()) for trace in pm4py_log]),
		'event': set.union(*[set(event.keys()) for trace in pm4py_log for event in trace])}

def filter_pm4py_log_attributes(pm4py_log, filter):
		for trace in pm4py_log:
			trace_attributes_list = list(trace.attributes.keys())
			for attribute in trace_attributes_list:
				if attribute not in filter['trace']:
					trace.attributes.pop(attribute, None)
			for event in trace:
				event_attributes_list = list(event.keys())
				for attribute in event_attributes_list:
					if attribute not in filter['event']:
						event.__delitem__(attribute)
		return pm4py_log
