#!/usr/bin/env python3.8
DEBUG = True
# Necessary imports
#import sys
#sys.path.insert(1, "../code")
import Complexity
import enriched
from dateutil import rrule
import statistics
from argparse import ArgumentParser
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.statistics.traces.generic.log import case_statistics

# Helper functions
def get_time_boundaries(pm4py_log):
	first_timestamp = min([event["time:timestamp"] for trace in pm4py_log for event in trace])
	last_timestamp = max([event["time:timestamp"] for trace in pm4py_log for event in trace])
	# Define start and end timestamps
	start_timestamp = Complexity.datetime(first_timestamp.year, first_timestamp.month, 1, 0, 0, 0, tzinfo=None)
	end_timestamp = last_timestamp.replace(tzinfo=None)
	return (start_timestamp, end_timestamp)

# Syntactic sugar to check if a timestamp is within some boundaries
def is_between(ts, start, end):
	if ts >= start and ts <= end:
		return True
	return False

def shortest_and_median_time(full_log):
	times = [(max([event["time:timestamp"] for event in trace])-min([event["time:timestamp"] for event in trace])).total_seconds() for trace in full_log]
	shortest_time = min(times)
	median_time = statistics.median(times)
	return (shortest_time, median_time)

# Split functions
## Split by intersect
def split_intersect(pm4py_log):
	start_timestamp, end_timestamp = get_time_boundaries(pm4py_log)
	# Probably replace with generator and/or list comprehension
	sublogs = []
	for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_timestamp, until=end_timestamp):
		filtered_pm4py_log = timestamp_filter.filter_traces_intersecting(pm4py_log, dt, dt+Complexity.relativedelta(months=1))
		if len(filtered_pm4py_log)==0:
			continue
		#if(args.verbose or DEBUG):
		#	print(str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(len(filtered_pm4py_log))+" cases")
		sublogs.append((dt, filtered_pm4py_log))
	return sublogs

## Split by start time
def split_start(pm4py_log):
	start_timestamp, end_timestamp = get_time_boundaries(pm4py_log)
	sublogs = []
	for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_timestamp, until=end_timestamp):
		# Assuming first event in trace is the earliest
		filtered_pm4py_log = Complexity.pm4py.objects.log.obj.EventLog([trace for trace in pm4py_log if is_between(trace[0]["time:timestamp"], dt, dt+Complexity.relativedelta(months=1))])
		if len(filtered_pm4py_log)==0:
			continue
		if(args.verbose or DEBUG):
			print(str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(len(filtered_pm4py_log))+" cases")
		sublogs.append(filtered_pm4py_log)
	return sublogs


# Measure functions
## Measure generic
def measure_generic(pm4py_log):
	# Create df
	# Measure
	distinct_activities = sorted(set([event['concept:name'] for trace in pm4py_log for event in trace]))
	simple_cycles = [0]*len(pm4py_log)
	activity_repetitions = [0]*len(pm4py_log)
	for i in range(len(pm4py_log)):
		trace_activities = set()
		for j in range(len(pm4py_log[i])-1):
			if pm4py_log[i][j]['concept:name']==pm4py_log[i][j+1]['concept:name']:
				simple_cycles[i] += 1
			if pm4py_log[i][j]['concept:name'] in trace_activities:
				activity_repetitions[i] += 1
			else:
				trace_activities.add(pm4py_log[i][j]['concept:name'])
		if pm4py_log[i][-1]['concept:name'] in trace_activities:
			activity_repetitions[i] += 1
	total_simple_cycles = sum(simple_cycles)
	avg_simple_cycles = statistics.mean(simple_cycles)
	median_simple_cycles = statistics.median(simple_cycles)
	total_activity_repetitions = sum(activity_repetitions)
	avg_activity_repetitions = statistics.mean(activity_repetitions)
	median_activity_repetitions = statistics.median(activity_repetitions)
	# Return df
	if(args.verbose or DEBUG):
		print("Distinct activities: "+str(len(distinct_activities)))
		print("Number of simple cycles: "+str(total_simple_cycles))
		print("Average number of simple cycles per trace: "+str(avg_simple_cycles))
		print("Median number of simple cycles per trace: "+str(median_simple_cycles))
		print("Number of activity repetitions: "+str(total_activity_repetitions))
		print("Average number of activity repetitions per trace: "+str(avg_activity_repetitions))
		print("Median number of activity repetitions per trace: "+str(median_activity_repetitions))
	return (len(pm4py_log), distinct_activities, len(distinct_activities), total_simple_cycles, avg_simple_cycles, median_simple_cycles, total_activity_repetitions, avg_activity_repetitions, median_activity_repetitions)

## Measure complexity incl. with EPA
def measure_complexity(pm4py_log):
	# Create df
	# Plain log
	plain_log = Complexity.generate_log(pm4py_log)
	# EPA
	pa = Complexity.build_graph(plain_log)
	# Entropy metrics
	var_ent = Complexity.graph_complexity(pa) # absolute and normalized
	seq_ent = Complexity.log_complexity(pa)
	seq_ent_lin = Complexity.log_complexity(pa, "linear")
	# TODO
	seq_ent_exp = Complexity.log_complexity(pa, "exp", 1) #float(args.ex_k)
	#EEPA
	enriched_plain_log = enriched.generate_enriched_log(pm4py_log)
	eepa = enriched.build_enriched_graph(enriched_plain_log)
	enriched_var_ent = Complexity.graph_complexity(eepa)
	enriched_seq_ent = Complexity.log_complexity(eepa)
	enriched_seq_ent_lin = Complexity.log_complexity(eepa, "linear")
	enriched_seq_ent_exp = Complexity.log_complexity(eepa, "exp", 1)
	# Measure all
	# Return df
	if(args.verbose or DEBUG):
		print("Simple Variant entropy: "+str(var_ent))
		print("Simple Sequence entropy: "+str(seq_ent))
		print("Simple Sequence entropy with linear forgetting: "+str(seq_ent_lin))
		print("Simple Sequence entropy with exponential forgetting (k=1): "+str(seq_ent_exp))
		print("Enriched Variant entropy: "+str(enriched_var_ent))
		print("Enriched Sequence entropy: "+str(enriched_seq_ent))
		print("Enriched Sequence entropy with linear forgetting: "+str(enriched_seq_ent_lin))
		print("Enriched Sequence entropy with exponential forgetting (k=1): "+str(enriched_seq_ent_exp))
	return (*var_ent, *seq_ent, *seq_ent_lin, *seq_ent_exp,
		*enriched_var_ent, *enriched_seq_ent, *enriched_seq_ent_lin, *enriched_seq_ent_exp)

## Measure SNA
def measure_sna(pm4py_log):
	if not all(["org:resource" in event for trace in pm4py_log for event in trace]):
		return tuple([Complexity.pd.NA]*8)
	# Create df
	# Measure
	resources = set([event["org:resource"] for trace in pm4py_log for event in trace if "org:resource" in event.keys()])
	total_resources = len(resources)
	avg_resources = statistics.mean([len(set([event["org:resource"] for event in trace if "org:resource" in event.keys()])) for trace in pm4py_log])
	median_resources = statistics.median([len(set([event["org:resource"] for event in trace if "org:resource" in event.keys()])) for trace in pm4py_log])
	#
	handovers = {resource:dict() for resource in resources}
	handover_traces = []
	for trace in pm4py_log:
		trace_handovers_counter = 0
		for i in range(len(trace)-1):
			handover_from = trace[i]['org:resource']
			handover_to = trace[i+1]['org:resource']
			if handover_from != handover_to:
				if handover_to not in handovers[handover_from].keys():
					handovers[handover_from][handover_to] = 0
				handovers[handover_from][handover_to] += 1
				trace_handovers_counter += 1
		handover_traces.append(trace_handovers_counter)
	#
	total_handovers = sum(handover_traces)
	avg_handovers = statistics.mean(handover_traces)
	median_handovers = statistics.median(handover_traces)
	#
	avg_from_handovers = statistics.mean([sum(h.values()) for h in handovers.values()])
	median_from_handovers = statistics.median([sum(h.values()) for h in handovers.values()])
	#
	if(args.verbose or DEBUG):
		print("Number of participants: "+str(total_resources))
		print("Average number of participants per trace: "+str(avg_resources))
		print("Median number of participants per trace: "+str(median_resources))
		print("Number of work handovers: "+str(total_handovers))
		print("Average number of work handovers per trace: "+str(avg_handovers))
		print("Median number of work handovers per trace: "+str(median_handovers))
		print("Average number of work handovers from participant: "+str(avg_from_handovers))
		print("Median number of work handovers from participant: "+str(median_from_handovers))
	return (total_resources, avg_resources, median_resources, total_handovers, avg_handovers, median_handovers, avg_from_handovers, median_from_handovers)
	# Return df

## Measure performance
def measure_performance(pm4py_log, shortest_full, median_full):
	# Create df
	#time_metrics = Complexity.pd.DataFrame(columns=["Date", "Median throughput time", "Average throughput time", 
	#	"PM4Py median throughput time", "Median cycle time", "Average cycle time"]) # new
	# Measure avg, median performance
	median_time = statistics.median([(max([event["time:timestamp"] for event in trace])-min([event["time:timestamp"] for event in trace])).total_seconds() for trace in pm4py_log])
	avg_time = statistics.mean([(max([event["time:timestamp"] for event in trace])-min([event["time:timestamp"] for event in trace])).total_seconds() for trace in pm4py_log])
	#pm_enriched_log = Complexity.pm4py.objects.log.util.interval_lifecycle.assign_lead_cycle_time(pm4py_log)
	cycle_times = [sum([event['@@duration'] for event in trace]) for trace in pm4py_log] #pm_enriched_log
	median_cycle_time = statistics.median(cycle_times)
	avg_cycle_time = statistics.mean(cycle_times)
	pm_median_time = case_statistics.get_median_case_duration(pm4py_log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
	ratio_to_shortest = median_time/shortest_full if shortest_full > 0 else Complexity.pd.NA
	ratio_to_median = median_time/median_full if median_full > 0 else Complexity.pd.NA
	#Return df
	if(args.verbose or DEBUG):
		print("Median case duration: "+str(median_time)+" seconds")
		print("Average case duration: "+str(avg_time)+" seconds")
		print("Median PM4Py case duration: "+str(pm_median_time)+" seconds")
		print("Median cycle time: "+str(median_cycle_time)+" seconds")
		print("Average cycle time: "+str(avg_cycle_time)+" seconds")
		print("Ratio to shortest duration: "+str(ratio_to_shortest))
		print("Ratio to median duration: "+str(ratio_to_median))
	return (median_time, avg_time, pm_median_time, median_cycle_time, avg_cycle_time, ratio_to_shortest, ratio_to_median)

## Measure other complexity
def measure_other_complexity(pm4py_log):
	plain_log = Complexity.generate_log(pm4py_log)
	pa = Complexity.build_graph(plain_log)
	other_metrics = Complexity.perform_measurements(["all"], log=plain_log, pm4py_log=pm4py_log, pa=pa, quiet=False)

	return (other_metrics["Magnitude"], other_metrics["Support"], other_metrics["Variety"], other_metrics["Level of detail"], other_metrics["Time granularity"],
		other_metrics["Structure"], other_metrics["Affinity"], other_metrics["Trace length"]["min"], other_metrics["Trace length"]["avg"], other_metrics["Trace length"]["max"],
		other_metrics["Distinct traces"]/100, other_metrics["Deviation from random"], other_metrics["Lempel-Ziv complexity"],
		other_metrics["Pentland's task complexity"])
# Main
if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument("-f", "--file", dest="file", help="input log file")
	parser.add_argument("-v", "--verbose", dest="verbose", help="verbose output", default=False, action="store_true")
	args = parser.parse_args()

	base_filename = Complexity.extract_base_filename(args.file)
	pm_log = Complexity.generate_pm4py_log(args.file)

	# enrich here
	pm_log = Complexity.pm4py.objects.log.util.interval_lifecycle.assign_lead_cycle_time(pm_log)
	sublogs = split_intersect(pm_log)
	shortest, med = shortest_and_median_time(pm_log)
	df = Complexity.pd.DataFrame(columns=["Date", 
		"Cases", "Distinct activities", "Number of distinct activities", "Number of simple cycles in period",
		"Average number of simple cycles per trace", "Median number of simple cycles per trace", "Number of activity repetitions in period",
		"Average number of activity repetitions per trace", "Median number of activity repetitions per trace",
		"Number of participants in period", "Average number of participants per trace", "Median number of participants per trace", "Handover of work in period",
		"Average handover of work per trace", "Median handover of work per trace",
		"Average handover of work from participant", "Median handover of work from participant",
		"Simple Variant Entropy", "Simple Normalized Variant Entropy", "Simple Sequence Entropy", "Simple Normalized Sequence Entropy",
		"Simple Sequence Entropy (linear forgetting)", "Simple Normalized Sequence Entropy (linear forgetting)",
		"Simple Sequence Entropy (exponential forgetting, k1)", "Simple Normalized Sequence Entropy (exponential forgetting, k1)",
		"Enriched Variant Entropy", "Enriched Normalized Variant Entropy", "Enriched Sequence Entropy", "Enriched Normalized Sequence Entropy",
		"Enriched Sequence Entropy (linear forgetting)", "Enriched Normalized Sequence Entropy (linear forgetting)",
		"Enriched Sequence Entropy (exponential forgetting, k1)", "Enriched Normalized Sequence Entropy (exponential forgetting, k1)",
		"Median throughput time", "Average throughput time",
		"PM4Py median throughput time", "Median cycle time", "Average cycle time",
		"Ratio to shortest", "Ratio to median",
		"Magnitude", "Support", "Variety", "Level of detail", "Time granularity", "Structure", "Affinity",
		"Minimum trace length", "Average trace length", "Maximum trace length", "Distinct traces", "Deviation from random", "Lempel-Ziv",
		"Pentland task complexity"])
	i = 0
	for dt,filtered_log in sublogs:
		if(args.verbose or DEBUG):
			print(str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(len(filtered_log))+" cases")
		perf = measure_performance(filtered_log, shortest, med)
		generic = measure_generic(filtered_log)
		sna = measure_sna(filtered_log)
		complexity = measure_complexity(filtered_log)
		other_complexity = measure_other_complexity(filtered_log)
		df.loc[i] = [dt, *generic, *sna, *complexity, *perf, *other_complexity]
		i += 1

	df.to_csv(base_filename+"_metrics.csv", index=False)
