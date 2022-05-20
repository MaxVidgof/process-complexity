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
