#!/usr/bin/env python3.8
DEBUG = True
# Necessary imports
from dateutil import rrule
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
args = parser.parse_args()
# Read log
base_filename = Complexity.extract_base_filename(args.file)
pm4py_log = Complexity.generate_pm4py_log(args.file, verbose=args.verbose)
first_timestamp = min([event["time:timestamp"] for trace in pm4py_log for event in trace])
last_timestamp = max([event["time:timestamp"] for trace in pm4py_log for event in trace])
# Define start and end timestamps
start_timestamp = Complexity.datetime(first_timestamp.year, first_timestamp.month, 1, 0, 0, 0, tzinfo=None)
end_timestamp = last_timestamp.replace(tzinfo=None)

if args.verbose:
	print("Overall time range: from "+str(Complexity.calendar.month_name[first_timestamp.month])+" "+str(first_timestamp.year)+" to "+str(Complexity.calendar.month_name[last_timestamp.month])+" "+str(last_timestamp.year))

# Specialized sequence entropy with forgetting
# where only events that are in previous months are discounted
def sequence_entropy_month_forgetting(pa, start):
	normalize = sum([len(AT.sequence) for AT in Complexity.flatten(pa.activity_types.values())])
	normalize = normalize*Complexity.math.log(normalize)
	entropy = 0
	for AT in Complexity.flatten(pa.activity_types.values()):
		for event in AT.sequence:
			# TODO
			entropy += math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k)
	entropy = Complexity.math.log(entropy) * entropy
	for i in range(1,pa.c+1):
		e = 0
		for AT in pa.nodes:
			if AT.c == i:
				for event in AT.sequence:
					# TODO
					e += Complexity.math.exp((-(last_timestamp - event.timestamp).total_seconds()/timespan)*k)
		entropy -= math.log(e)*e
	return entropy,(entropy/normalize)

# Prepare
def save_figures(df, metric):
	Complexity.plt.figure(figsize=(1920,1080))
	df.plot(df.columns[0], df.columns[1])
	Complexity.plt.savefig(base_filename+"_"+metric+".png")
	Complexity.plt.figure(figsize=(1920,1080))
	df.plot(df.columns[0], df.columns[2])
	Complexity.plt.savefig(base_filename+"_"+metric+"_Normalized.png")
	Complexity.plt.close("all")

generic_metrics = Complexity.pd.DataFrame(columns=["Date", "Cases", "Distinct activities", "Number of distinct activities", "Number of simple cycles in period", "Average number of simple cycles per trace", "Median number of simple cycles per trace", "Number of activity repetitions in period", "Average number of activity repetitions per trace", "Median number of activity repetitions per trace"])
variant_entropy = Complexity.pd.DataFrame(columns=["Date", "Variant entropy", "Normalized variant entropy"])
sequence_entropy = Complexity.pd.DataFrame(columns=["Date", "Sequence entropy", "Normalized sequence entropy"])
sequence_entropy_linear = Complexity.pd.DataFrame(columns=["Date", "Sequence entropy (linear forgetting)", "Normalized sequence entropy (linear forgetting)"])
sequence_entropy_exponential = Complexity.pd.DataFrame(columns=["Date", "Sequence entropy (exponential forgetting)", "Normalized sequence entropy (exponential forgetting)"])

time_metrics = Complexity.pd.DataFrame(columns=["Date", "Median throughput time", "Average throughput time", "PM4Py median throughput time", "Median cycle time", "Average cycle time"]) # new
sna_metrics = Complexity.pd.DataFrame(columns=["Date", "Number of participants in period", "Average number of participants per trace", "Median number of participants per trace", "Handover of work in period", "Average handover of work per trace", "Median handover of work per trace", "Average handover of work from participant", "Median handover of work from participant"]) # new
#TODO: add sojourn time?
#TODO: add rework (activities)?
#TODO: add handover, subcontracting, working together, similar activities, roles, clustering

# Iterate over every month in log
for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_timestamp, until=end_timestamp):
	# Filter PM4Py log
	filtered_pm4py_log = timestamp_filter.filter_traces_intersecting(pm4py_log, dt, dt+Complexity.relativedelta(months=1))
	# Note that PM4Py uses tzinfo=None, look at it in case os timezone issues:
	# https://github.com/pm4py/pm4py-core/blob/97da56f91a3b8cee0215be9790dad2f6da32ad87/pm4py/algo/filtering/log/timestamp/timestamp_filter.py#L177
	if len(filtered_pm4py_log)==0:
		continue
	# Print number of cases in this month
	if(args.verbose or DEBUG):
		print(str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(len(filtered_pm4py_log))+" cases")

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
	generic_metrics.loc[len(generic_metrics)]=[dt, len(filtered_pm4py_log), distinct_activities, len(distinct_activities), total_simple_cycles, avg_simple_cycles, median_simple_cycles, total_activity_repetitions, avg_activity_repetitions, median_activity_repetitions]
	if(args.verbose):
		print("Number of simple cycles in "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(total_simple_cycles))
		print("Average number of simple cycles per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(avg_simple_cycles))
		print("Median number of simple cycles per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(median_simple_cycles))
		print("Number of activity repetitions in "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(total_activity_repetitions))
		print("Average number of activity repetitions per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(avg_activity_repetitions))
		print("Median number of activity repetitions per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(median_activity_repetitions))

	# Build event log
	filtered_log = Complexity.generate_log(filtered_pm4py_log, verbose=False) # args.verbose)
	# Build EPA on the filtered log
	pa = Complexity.build_graph(filtered_log)
	# Calculate variant entropy of the EPA
	var_ent = Complexity.graph_complexity(pa)
	variant_entropy.loc[len(variant_entropy)]=[dt, *var_ent]
	if(args.verbose):
		print("Variant entropy: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(var_ent))
	# Calculate sequence entropy of the EPA
	seq_ent = Complexity.log_complexity(pa)
	sequence_entropy.loc[len(sequence_entropy)]=[dt, *seq_ent]
	if(args.verbose):
		print("Sequence entropy: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(seq_ent))
	# Calculate sequence entropy of the EPA with linear forgetting
	seq_ent_lin = Complexity.log_complexity(pa, "linear")
	sequence_entropy_linear.loc[len(sequence_entropy_linear)]=[dt, *seq_ent_lin]
	if(args.verbose):
		print("Sequence entropy with linear forgetting: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(seq_ent_lin))
	# Calculate sequence entropy of the EPA with exponential forgetting
	seq_ent_exp = Complexity.log_complexity(pa, "exp", float(args.ex_k))
	sequence_entropy_exponential.loc[len(sequence_entropy_exponential)]=[dt, *seq_ent_exp]
	if(args.verbose):
		print("Sequence entropy with exponential forgetting (k="+str(args.ex_k)+")"+": "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(seq_ent_exp))

	# TODO: Calculate sequence entropy where only events in previous months are discounted
	#print(sequence_entropy_month_forgetting(pa, dt))

	# Calculate median throughput time
	pm_median_time = case_statistics.get_median_case_duration(filtered_pm4py_log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
	# Calculate manually instead of using pm4py methods
	#median_time = statistics.median(case_statistics.get_all_case_durations(filtered_pm4py_log,parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"}))
	#avg_time = statistics.mean(case_statistics.get_all_case_durations(filtered_pm4py_log,parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"}))
	median_time = statistics.median([(max([event["time:timestamp"] for event in trace])-min([event["time:timestamp"] for event in trace])).total_seconds() for trace in filtered_pm4py_log])
	avg_time = statistics.mean([(max([event["time:timestamp"] for event in trace])-min([event["time:timestamp"] for event in trace])).total_seconds() for trace in filtered_pm4py_log])
	enriched_filtered_log = Complexity.pm4py.objects.log.util.interval_lifecycle.assign_lead_cycle_time(filtered_pm4py_log)
	cycle_times = [sum([event['@@duration'] for event in trace]) for trace in enriched_filtered_log]
	median_cycle_time = statistics.median(cycle_times)
	avg_cycle_time = statistics.mean(cycle_times)
	time_metrics.loc[len(time_metrics)] = [dt, median_time, avg_time, pm_median_time, median_cycle_time, avg_cycle_time]
	if(args.verbose):
		print("Median case duration: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(median_time)+" seconds")
		print("Average case duration: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(avg_time)+" seconds")
		print("Median cycle time: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(median_cycle_time)+" seconds")
		print("Average cycle time: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(avg_cycle_time)+" seconds")

	# SNA
	# if org:resource attribute exists for event
	#if "org:resource" in filtered_pm4py_log[0][0].keys():
	if False in set(['org:resource' in event.keys() for trace in filtered_pm4py_log for event in trace]):
		sna_metrics.loc[len(sna_metrics)] = [dt]+ [Complexity.pd.NA]*8
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

	sna_metrics.loc[len(sna_metrics)] = [dt, total_resources, avg_resources, median_resources, total_handovers, avg_handovers, median_handovers, avg_from_handovers, median_from_handovers]
	if(args.verbose):
		print("HERE")
		print("Number of participants in "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(total_resources))
		print("Average number of participants per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(avg_resources))
		print("Median number of participants per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(median_resources))
		print("Number of work handovers in "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(total_handovers))
		print("Average number of work handovers per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(avg_handovers))
		print("Median number of work handovers per trace: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(median_handovers))
		print("Average number of work handovers from participant: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(avg_from_handovers))
		print("Median number of work handovers from participant: "+str(Complexity.calendar.month_name[dt.month])+" "+str(dt.year)+": "+str(median_from_handovers))

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
	save_figures(variant_entropy, "Monthly_Variant_Entropy")
	save_figures(sequence_entropy, "Monthly_Sequence_Entropy")
	save_figures(sequence_entropy_linear, "Monthly_Sequence_Entropy_Linear")
	save_figures(sequence_entropy_exponential, "Monthly_Sequence_Entropy_Exponential")
	#TODO: save_figures for new metrics
if(args.save_csv):
	df_merged = reduce(lambda left, right: Complexity.pd.merge(left, right, on=["Date"], how="outer"), 
		[generic_metrics, variant_entropy, sequence_entropy, sequence_entropy_linear, sequence_entropy_exponential, time_metrics, sna_metrics]).fillna('NA')
	df_merged.to_csv(base_filename+"_metrics.csv", index=False)
