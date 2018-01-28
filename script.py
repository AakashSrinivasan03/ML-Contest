
import os
import matplotlib.pyplot as plt


# The initial parameter set
#initial_parameters = set(["cycles","instructions","cache-references","cache-misses","bus-cycles"])
initial_parameters = set(["cycles","instructions","cache-references","cache-misses","bus-cycles","cpu-migrations","page-faults", "L1-dcache-load-misses", "L1-dcache-stores", "L1-icache-load-misses", "branch-load-misses","dTLB-load-misses","dTLB-store-misses","iTLB-load-misses","context-switches","emulation-faults","major-faults","alignment-faults"])
# Get the string representing the parameters
#initial_parameters = set(["cycles","instructions","cache-references","cache-misses","bus-cycles","cpu-migrations","page-faults","LLC-load-misses","LLC-store-misses", "L1-dcache-load-misses", "L1-dcache-stores", "L1-icache-load-misses", "branch-load-misses","dTLB-load-misses","dTLB-store-misses","iTLB-load-misses", "node-load-misses","node-store-misses","context-switches","emulation-faults","task-clock","major-faults","alignment-faults"])
initial_parameter_string = ""
for param in initial_parameters:
	initial_parameter_string += param
	initial_parameter_string += ','
initial_parameter_string = initial_parameter_string[:-1] 	# Remove the last comma

# Create a dictionary with key as parameters
parameter_values = dict.fromkeys(list(initial_parameters))
#print parameter_values

# Initialize to empty lists
for key in parameter_values.keys():
	parameter_values[key] = []

# Profile the CPU for 5 seconds for basic parameters
how_many_iterations = range(200)
for i in how_many_iterations:
	os.system("sudo perf stat -e "+initial_parameter_string+" -a sleep 2 1>garbage.txt 2>profiled.txt")

	f = open("profiled.txt", "r")
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


# # Plot graphs based on the parameter values
#for parameter in parameter_values.keys():
 	#plt.plot(how_many_iterations, parameter_values[parameter])
 	#plt.ylabel(parameter)
 	#plt.xlabel("Iterations")
 	#plt.show()

# Get the mean values of all parameter usages
# Write parameter values in file
#->nodestoremisses,taskclock,llc-store-misses,node-load-misses
f = open("background_values.csv","w")
for parameter in parameter_values.keys():
	#print parameter
	f.write(parameter+","+str(float(sum(parameter_values[parameter]))/len(parameter_values[parameter]))+"\n")
f.close()
