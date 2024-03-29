# Process Complexity
## Everything you need to work with Extended Prefix Automata

This package offers tools to generate Extended Prefix Automata (EPA) and calculate corresponging process complexity metrics. You can read about EPA and process complexity in the journal article [The connection between process complexity of event sequences and models discovered by process mining](https://www.sciencedirect.com/science/article/pii/S0020025522002997) by Adriano Augusto, Jan Mendling, Maxim Vidgof and Bastian Wurm.

This tool can be used as stand-alone tool or as Python 3 module. In stand-alone mode the tool can calculate complexity metrics (see below) and optionally export the EPA as picture. The tool accepts XES and CSV files as input. When used as module, the tool offers all required code to build an EPA and use it in your code.

By default, *Variant entropy, Normalized variant entropy, Sequence entropy* and *Normalized sequence entropy* are calculated. It is also possible to calculate additional complexity measures (see **Usage**)

## Installation
1. Clone this repository
```sh
git clone https://github.com/MaxVidgof/process-complexity
```
2. Install requirements
```sh
pip install -r requirements.txt
```
3. Done!

## Usage

### Stand-alone
```sh
python3 Complexity.py [options]
```

#### Options:
```-h``` displays help
```-f [FILE]``` to specify the input event log
```-m [MEASURE]``` additional complexity measures. Choose from: magnitude, support, variety, level_of_detail, time_granularity, structure, affinity, trace_length, distinct_traces, deviation_from_random, lempel-ziv, pentland, *all*
```-d``` creates [GraphViz](https://graphviz.org/) specification and prints it to ```STDOUT```
```-g``` draws the Extended Prefix Automaton an an SVG image. Requires [GraphViz](https://graphviz.org/)
```-a``` explicitly mark accepting states
```-p``` print prefix for each state
```--png``` additional option for ```-g```, will try to creade a PNG image but might fail if EPA is too big
```--hide-event``` additional option for ```-g```, will only show states in the image but not the corresponding events
```-v``` verbose output (warning: a LOT of output)

#### Usage example
Calculate entropy metrics, magnitude and variety:
```sh
python3 Complexity.py -f some_log.xes -m magnitude -m variety
```
Calculate entropy metrics and create a PNG image of the EPA:
```sh
python3 Complexity.py -f some_log.xes -g --png
```

### Module
Create a new Python 3 script or start Python3 shell in the **same** directory where ```Complexity.py``` is stored.
Import the module:
```python
import Complexity
```
Build EPA from an event log:
```python
filename = "some_log.xes"
# Import log in PM4Py format
pm4py_log = Complexity.generate_pm4py_log(filename)
# Transform PM4Py log into plain log
log = Complexity.generate_log(pm4py_log) 
# Build EPA
epa = Complexity.build_graph(log)
```
You can now use the EPA in your code!

## Enriched Extended Prefix Automata
This package can be also used to work with Enriched Extended Prefix Automata (EEPA). In contrast to EPA, EEPA additionally use event data to calculate process complexity. More information can be found in the paper [Leveraging Event Data for Measuring Process Complexity](https://link.springer.com/chapter/10.1007/978-3-031-27815-0_7) by Maxim Vidgof and Jan Mendling presented at EdbA'22. 

### Usage
This functionality is provided by the script ```enriched.py``` and can be only used as a module. Building the EEPA is similar to building an EPA, however, the plain log generation is also different in order to keep the event data.
```python
import Complexity
import enriched
filename = "some_log.xes"
pm4py_log = Complexity.generate_pm4py_log(filename)
log = enriched.generate_enriched_log(pm4py_log)
eepa = enriched.build_enriched_graph(log)
```
## Complexity over time
This package also includes the sctipt ```over_time.py``` that can be used to evaluate how process complexity and other log measures evolve over time. The default time interval is a month, however, it can be adapred in the script. For more usage information, run the script with the ```-h``` flag.

## Scripts (old)
This package also includes scripts ```monthly.py```, ```weekly.py```, ```blocks.py``` and ```sliding.py```. These were used to evaluate how process complexity and other log measures evolve over time before ```over_time.py``` was developed. While these scripts will not be maintained, the user is still free to use them.
The user can split the input log into:
* time intervals (```monthly.py```, ```weekly.py```)
* blocks of fixed size (```blocks.py```)
* sliding window (```sliding.py```)

For more information, run any of the scripts with the ```-h``` flag.
