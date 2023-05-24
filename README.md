# bankruns
an ABM model of a Bank run

Modelling a bank run as a crisis of confidence with insights from epidemic modelling.

The code provides a simulation of a banking crisis using agent-based modeling. 

Here is a breakdown of what the code does:
The code defines a function crisis_stackplot that creates a stack plot visualization of a bank's condition over time.
It defines a class Bank that represents a bank in the simulation. Each bank has attributes such as condition, safety status, and initial deposits.
It defines a class Customer that represents a customer in the simulation. Each customer has attributes such as total deposits and allocations to different banks.
It defines a class CrisisModel that represents the simulation model. It initializes the agents (banks and customers) and the network of the model.
The step method of the CrisisModel class defines the events that occur in each simulation step, such as updating bank statuses, customer withdrawals, testing bank solvency, spreading the crisis to other banks, and managing the crisis.
The end method of the CrisisModel class records evaluation measures at the end of the simulation.
The code sets up the parameters for the simulation and creates an instance of the CrisisModel.
It runs the simulation and prints the results.
It visualizes the results using a stack plot and a network graph.
The code also includes a section for multiple-run experiments, where different parameter values are tested and sensitivity analysis is performed using the Sobol method.

Overall, the code simulates the spread and management of a banking crisis, allowing for the analysis of different scenarios and parameter settings.
