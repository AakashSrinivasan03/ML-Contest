
import os
import matplotlib.pyplot as plt
from pandas import DataFrame


# The initial parameter set
#initial_parameters = set(["cycles","instructions","cache-references","cache-misses","bus-cycles"])
initial_parameters = set(["cycles","instructions","cache-references","cache-misses","bus-cycles","cpu-migrations","page-faults", "L1-dcache-load-misses", "L1-dcache-stores", "L1-icache-load-misses", "branch-load-misses","dTLB-load-misses","dTLB-store-misses","iTLB-load-misses","context-switches"])
#initial_parameters = set(["cycles","instructions","cache-references","cache-misses","bus-cycles","cpu-migrations","page-faults","LLC-load-misses","LLC-store-misses", "L1-dcache-load-misses", "L1-dcache-stores", "L1-icache-load-misses", "branch-load-misses","dTLB-load-misses","dTLB-store-misses","iTLB-load-misses", "node-load-misses","node-store-misses","context-switches","emulation-faults","task-clock","major-faults","alignment-faults"])
# Get the string representing the parameters
initial_parameter_string = ""
for param in initial_parameters:
	initial_parameter_string += param
	initial_parameter_string += ','
initial_parameter_string = initial_parameter_string[:-1] 	# Remove the last comma

# Create a dictionary with key as parameters
parameter_values = dict.fromkeys(list(initial_parameters))

# Initialize to empty lists
for key in parameter_values.keys():
	parameter_values[key] = []

# Profile the CPU for 5 seconds for basic parameters
how_many_iterations = range(5)
for i in how_many_iterations:
	os.system("sudo perf stat -e "+initial_parameter_string+" -a sleep 2 1>garbage.txt 2>runtime_profiled.txt")

	f = open("runtime_profiled.txt", "r")
	lines = f.readlines()
	for line in lines:
		temp_line = line.strip().split()
		try:
			if temp_line[1] in initial_parameters: 	# If the parameter is required
				number = temp_line[0]
				number = number.split(',')
				string_number = ""
				for num in number:
					string_number = string_number + num
				number = int(string_number)
				parameter_values[temp_line[1]].append(number)
		except:
			pass

# Store the average values for parameters
for parameter in parameter_values.keys():
	parameter_values[parameter] = float(sum(parameter_values[parameter]))/len(parameter_values[parameter])
ff = open("background_values_after_log_reg.csv","w")
for parameter in parameter_values.keys():
	#print parameter
	ff.write(parameter+","+str(parameter_values[parameter])+"\n")
# Load the values for background values
background_values = {}
f = open("background_values.csv","r")
lines = f.readlines()
for line in lines:
	temp_line = line.strip().split(',')
	background_values[temp_line[0]] = float(temp_line[1])

# Calculate percentage changes for all the parameters
fff = open("background_values_ratio.csv","w")
for parameter in parameter_values.keys():
	print parameter
	parameter_values[parameter] = float((parameter_values[parameter] - background_values[parameter]))/background_values[parameter]*100
	fff.write(parameter+","+str(parameter_values[parameter])+"\n")

print(parameter_values)
