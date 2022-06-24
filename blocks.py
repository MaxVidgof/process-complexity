#!/usr/bin/env python3.8
DEBUG = True
# Necessary imports
import Complexity
from pm4py.algo.filtering.log.timestamp import timestamp_filter

from pm4py.statistics.traces.generic.log import case_statistics # new, traces.generic.log in documentation
from argparse import ArgumentParser
from functools import reduce
import statistics

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="file", help="input log file")
parser.add_argument("-v", "--verbose", dest="verbose", help="verbose output", default=False, action="store_true")
parser.add_argument("-e", "--exponential-forgetting", dest="ex_k", help="coefficitne for exponential forgetting", default=1)
parser.add_argument("--save-plots", dest="save_plots", help="save results as plots", default=False, action="store_true")
parser.add_argument("--save-csv", dest="save_csv", help="save results as csv", default=False, action="store_true")
parser.add_argument("-w", "--window-size", dest="window_size", help="Window size (number of traces)", default=100)
args = parser.parse_args()
window_size = int(args.window_size)
# Read log
base_filename = Complexity.extract_base_filename(args.file)
pm4py_log = Complexity.generate_pm4py_log(args.file, verbose=args.verbose)

# TODO: sort by timestamp, sometimes it is not done automatically
sorted_log = pm4py_log

# Prepare
def save_figures(df, metric):
	Complexity.plt.figure(figsize=(1920,1080))
	df.plot(df.columns[0], df.columns[1])
	Complexity.plt.savefig(base_filename+"_"+metric+".png")
	Complexity.plt.figure(figsize=(1920,1080))
	df.plot(df.columns[0], df.columns[2])
	Complexity.plt.savefig(base_filename+"_"+metric+"_Normalized.png")
	Complexity.plt.close("all")

generic_metrics = Complexity.pd.DataFrame(columns=["Block", "Cases", "Distinct activities", "Number of distinct activities", "Number of simple cycles in period", "Average number of simple cycles per trace", "Median number of simple cycles per trace", "Number of activity repetitions in period", "Average number of activity repetitions per trace", "Median number of activity repetitions per trace"])
variant_entropy = Complexity.pd.DataFrame(columns=["Block", "Variant entropy", "Normalized variant entropy"])
sequence_entropy = Complexity.pd.DataFrame(columns=["Block", "Sequence entropy", "Normalized sequence entropy"])
sequence_entropy_linear = Complexity.pd.DataFrame(columns=["Block", "Sequence entropy (linear forgetting)", "Normalized sequence entropy (linear forgetting)"])
sequence_entropy_exponential = Complexity.pd.DataFrame(columns=["Block", "Sequence entropy (exponential forgetting)", "Normalized sequence entropy (exponential forgetting)"])

time_metrics = Complexity.pd.DataFrame(columns=["Block", "Median throughput time", "Average throughput time", "PM4Py median throughput time", "Median cycle time", "Average cycle time"]) # new
sna_metrics = Complexity.pd.DataFrame(columns=["Block", "Number of participants in period", "Average number of participants per trace", "Median number of participants per trace", "Handover of work in period", "Average handover of work per trace", "Median handover of work per trace", "Average handover of work from participant", "Median handover of work from participant"]) # new

# Init indices
start_idx = 0
end_idx = window_size
block_number = 0
blocks = int(Complexity.math.ceil(len(sorted_log)/window_size))
# Iterate over all blocks of window_size traces
while(start_idx < len(sorted_log)):
	# Filter PM4Py log
	# Naive filtering produces a list instead of Log
	# To overcome this, we take Case IDs of the
	# traces at right indices...
	case_ids = [trace.attributes['concept:name'] for trace in sorted_log[start_idx:end_idx]]
	# ... and apply PM4Py filter
	filtered_pm4py_log = Complexity.pm4py.filter_log(lambda x: x.attributes['concept:name'] in case_ids, sorted_log)  #sorted_log[start_idx:end_idx]

	# Print block data
	if(args.verbose or DEBUG):
		print(f"Block {block_number+1}/{blocks}")
		print(f"Traces {start_idx}-{min(end_idx-1,len(sorted_log)-1)}")

	# Increment counters
	start_idx += window_size
	end_idx += window_size
	block_number += 1

	distinct_activities = sorted(set([event['concept:name'] for trace in filtered_pm4py_log for event in trace]))
	simple_cycles = [0]*len(filtered_pm4py_log)
	activity_repetitions = [0]*len(filtered_pm4py_log)
	for i in range(len(filtered_pm4py_log)):
		trace_activities = set()
		for j in range(len(filtered_pm4py_log[i])-1):
			if filtered_pm4py_log[i][j]['concept:name']==filtered_pm4py_log[i][j+1]['concept:name']:
				simple_cycles[i] += 1
			if filtered_pm4py_log[i][j]['concept:name'] in trace_activities:
				activity_repetitions[i] += 1
			else:
				trace_activities.add(filtered_pm4py_log[i][j]['concept:name'])
		if filtered_pm4py_log[i][-1]['concept:name'] in trace_activities:
			activity_repetitions[i] += 1
	total_simple_cycles = sum(simple_cycles)
	avg_simple_cycles = statistics.mean(simple_cycles)
	median_simple_cycles = statistics.median(simple_cycles)
	total_activity_repetitions = sum(activity_repetitions)
	avg_activity_repetitions = statistics.mean(activity_repetitions)
	median_activity_repetitions = statistics.median(activity_repetitions)
	generic_metrics.loc[len(generic_metrics)]=[block_number, len(filtered_pm4py_log), distinct_activities, len(distinct_activities), total_simple_cycles, avg_simple_cycles, median_simple_cycles, total_activity_repetitions, avg_activity_repetitions, median_activity_repetitions]
	if(args.verbose):
		print("Number of simple cycles in "+"block "+str(block_number)+": "+str(total_simple_cycles))
		print("Average number of simple cycles per trace: "+"block "+str(block_number)+": "+str(avg_simple_cycles))
		print("Median number of simple cycles per trace: "+"block "+str(block_number)+": "+str(median_simple_cycles))
		print("Number of activity repetitions in "+"block "+str(block_number)+": "+str(total_activity_repetitions))
		print("Average number of activity repetitions per trace: "+"block "+str(block_number)+": "+str(avg_activity_repetitions))
		print("Median number of activity repetitions per trace: "+"block "+str(block_number)+": "+str(median_activity_repetitions))

	# Build event log
	filtered_log = Complexity.generate_log(filtered_pm4py_log, verbose=False) # args.verbose)
	# Build EPA on the filtered log
	pa = Complexity.build_graph(filtered_log)
	# Calculate variant entropy of the EPA
	var_ent = Complexity.graph_complexity(pa)
	variant_entropy.loc[len(variant_entropy)]=[block_number, *var_ent]
	if(args.verbose):
		print("Variant entropy: "+"block "+str(block_number)+": "+str(var_ent))
	# Calculate sequence entropy of the EPA
	seq_ent = Complexity.log_complexity(pa)
	sequence_entropy.loc[len(sequence_entropy)]=[block_number, *seq_ent]
	if(args.verbose):
		print("Sequence entropy: "+"block "+str(block_number)+": "+str(seq_ent))
	# Calculate sequence entropy of the EPA with linear forgetting
	seq_ent_lin = Complexity.log_complexity(pa, "linear")
	sequence_entropy_linear.loc[len(sequence_entropy_linear)]=[block_number, *seq_ent_lin]
	if(args.verbose):
		print("Sequence entropy with linear forgetting: "+"block "+str(block_number)+": "+str(seq_ent_lin))
	# Calculate sequence entropy of the EPA with exponential forgetting
	seq_ent_exp = Complexity.log_complexity(pa, "exp", float(args.ex_k))
	sequence_entropy_exponential.loc[len(sequence_entropy_exponential)]=[block_number, *seq_ent_exp]
	if(args.verbose):
		print("Sequence entropy with exponential forgetting (k="+str(args.ex_k)+")"+": "+"block "+str(block_number)+": "+str(seq_ent_exp))


	# Calculate median throughput time
	pm_median_time = case_statistics.get_median_case_duration(filtered_pm4py_log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
	#median_time = statistics.median(case_statistics.get_all_case_durations(filtered_pm4py_log,parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"}))
	#avg_time = statistics.mean(case_statistics.get_all_case_durations(filtered_pm4py_log,parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"}))
	# Calculate manually instead of using pm4py methods
	median_time = statistics.median([(max([event["time:timestamp"] for event in trace])-min([event["time:timestamp"] for event in trace])).total_seconds() for trace in filtered_pm4py_log])
	avg_time = statistics.mean([(max([event["time:timestamp"] for event in trace])-min([event["time:timestamp"] for event in trace])).total_seconds() for trace in filtered_pm4py_log])
	enriched_filtered_log = Complexity.pm4py.objects.log.util.interval_lifecycle.assign_lead_cycle_time(filtered_pm4py_log)
	cycle_times = [sum([event['@@duration'] for event in trace]) for trace in enriched_filtered_log]
	median_cycle_time = statistics.median(cycle_times)
	avg_cycle_time = statistics.mean(cycle_times)
	time_metrics.loc[len(time_metrics)] = [block_number, median_time, avg_time, pm_median_time, median_cycle_time, avg_cycle_time]
	if(args.verbose):
		print("Median case duration: "+"block "+str(block_number)+": "+str(median_time)+" seconds")
		print("Average case duration: "+"block "+str(block_number)+": "+str(avg_time)+" seconds")
		print("Median cycle time: "+"block "+str(block_number)+": "+str(median_cycle_time)+" seconds")
		print("Average cycle time: "+"block "+str(block_number)+": "+str(avg_cycle_time)+" seconds")

	# SNA
	# if org:resource attribute exists for event
	#if "org:resource" in filtered_pm4py_log[0][0].keys():
	if False in set(['org:resource' in event.keys() for trace in filtered_pm4py_log for event in trace]):
		sna_metrics.loc[len(sna_metrics)] = [block_number]+ [Complexity.pd.NA]*8
		continue

	resources = []
	#for trace in filtered_pm4py_log:
	#	curr = set()
	#	for event in trace:
	#		if "org:resource" in event.keys():
	#			curr.add(event["org:resource"])
	#	resources.append((curr.copy(), len(curr)))
	#total_resources = len(set(Complexity.flatten([list(val[0]) for val in resources])))
	#avg_resources = statistics.mean([val[1] for val in resources])
	#median_resources = statistics.median([val[1] for val in resources])
	#total_resources, avg_resources, median_resources = (0,0,0)

	#total_resources = len(set([event["org:resource"] for trace in filtered_pm4py_log for event in trace if "org:resource" in event.keys()]))
	resources = set([event["org:resource"] for trace in filtered_pm4py_log for event in trace if "org:resource" in event.keys()])
	total_resources = len(resources)
	avg_resources = statistics.mean([len(set([event["org:resource"] for event in trace if "org:resource" in event.keys()])) for trace in filtered_pm4py_log])
	median_resources = statistics.median([len(set([event["org:resource"] for event in trace if "org:resource" in event.keys()])) for trace in filtered_pm4py_log])

	handovers = {resource:dict() for resource in resources}
	handover_traces = []
	for trace in filtered_pm4py_log:
		trace_handovers_counter = 0
		for i in range(len(trace)-1):
			# handover to oneself does not count
			handover_from = trace[i]['org:resource']
			handover_to = trace[i+1]['org:resource']
			if handover_from != handover_to:
				if handover_to not in handovers[handover_from].keys():
					handovers[handover_from][handover_to] = 0
				handovers[handover_from][handover_to] += 1
				trace_handovers_counter += 1
		handover_traces.append(trace_handovers_counter)

	total_handovers = sum(handover_traces)
	avg_handovers = statistics.mean(handover_traces)
	median_handovers = statistics.median(handover_traces)

	avg_from_handovers = statistics.mean([sum(h.values()) for h in handovers.values()])
	median_from_handovers = statistics.median([sum(h.values()) for h in handovers.values()])

	sna_metrics.loc[len(sna_metrics)] = [block_number, total_resources, avg_resources, median_resources, total_handovers, avg_handovers, median_handovers, avg_from_handovers, median_from_handovers]
	if(args.verbose):
		print("Number of participants in "+"block "+str(block_number)+": "+str(total_resources))
		print("Average number of participants per trace: "+"block "+str(block_number)+": "+str(avg_resources))
		print("Median number of participants per trace: "+"block "+str(block_number)+": "+str(median_resources))
		print("Number of work handovers in "+"block "+str(block_number)+": "+str(total_handovers))
		print("Average number of work handovers per trace: "+"block "+str(block_number)+": "+str(avg_handovers))
		print("Median number of work handovers per trace: "+"block "+str(block_number)+": "+str(median_handovers))
		print("Average number of work handovers from participant: "+"block "+str(block_number)+": "+str(avg_from_handovers))
		print("Median number of work handovers from participant: "+"block "+str(block_number)+": "+str(median_from_handovers))


print("\nGeneric metrics")
print(generic_metrics)
print("\nVariant entropy")
print(variant_entropy)
print("\nSequence entropy")
print(sequence_entropy)
print("\nSequence entropy with linear forgetting")
print(sequence_entropy_linear)
print("\nSequence entropy with exponential forgetting (k="+str(args.ex_k)+")")
print(sequence_entropy_exponential)
print("\nThroughput time") # new
print(time_metrics) # new
print("\nSNA metrics") # new
print(sna_metrics) # new
if(args.save_plots):
	save_figures(variant_entropy, "Variant_Entropy_Blocks_"+str(args.window_size))
	save_figures(sequence_entropy, "Sequence_Entropy_Blocks_"+str(args.window_size))
	save_figures(sequence_entropy_linear, "Sequence_Entropy_Linear_Blocks_"+str(args.window_size))
	save_figures(sequence_entropy_exponential, "Sequence_Entropy_Exponential_Blocks_"+str(args.window_size))
	#TODO: save_figures for new metrics
if(args.save_csv):
	df_merged = reduce(lambda left, right: Complexity.pd.merge(left, right, on=["Block"], how="outer"), 
		[generic_metrics, variant_entropy, sequence_entropy, sequence_entropy_linear, sequence_entropy_exponential, time_metrics, sna_metrics]).fillna('NA')
	df_merged.to_csv(base_filename+"_metrics_blocks_"+str(args.window_size)+".csv", index=False)
