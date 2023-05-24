"""
Created on Sun Mar 26 11:48:30 2023

@author: alexandrekateb
"""

#%% Insolvency Spread
import os
import numpy as np 
# Model design
import agentpy as ap
import networkx as nx
import random

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


def crisis_stackplot(data, fig, ax):
    """ Stackplot of bank's condition over time. """
    x = data.index.get_level_values('t')
    y = [data[var] for var in ['I', 'S', 'R','L']]

    sns.set()
    ax.stackplot(x, y, labels=['Facing Run', 'Unaffected', 'Recovered','Liquidated'],
                 colors = ['r', 'b', 'g','black'])

    ax.legend(bbox_to_anchor=(1.5,0.7))
    ax.set_xlim(0, max(1, len(x)-1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Percentage of all banks")
    fig.tight_layout()


class Bank(ap.Agent):

    def setup(self):
        """ Initialize a new variable at agent creation. """
        self.condition = 0  # Susceptible to runs = 0, Facing Bank Run = 1, Recovered = 2, Liquidated = 3
        self.safe = 1 # safe = 1, unsafe = 0, irrelevant = 2
#        self.contagion_prob = self.p.contagion_prob 
#        self.recovery_chance = self.p.recovery_chance
#        self.liquidation_chance = self.p.liquidation_chance
#        self.solvency_threshold = self.p.solvency_threshold
        self.initial_deposits = 0
    
    def update_status(self):
        rng = self.model.random
        if self.condition ==1:
            self.safe = 0
        elif self.condition == 2:
            self.safe = 1
        elif self.condition == 3:
            self.safe = 2
        elif self.condition ==0 and rng.random() < self.p.sunspot_prob:
            self.safe = 0
        else:
            pass     


    def calculate_initial_deposits(self):
        for customer in self.model.customers:
            self.initial_deposits += customer.deposits[self]
            self.deposits = self.initial_deposits


    def test_solvency(self):
        self.deposits = 0
        for customer in self.model.customers:
            self.deposits += customer.deposits[self]
        if self.deposits < self.p.solvency_threshold * self.initial_deposits:
            if self.condition == 0:
                self.condition = 1

                                                                           
    def spread_crisis(self):
        """ Spread bank crisis to peers in the network. """
        rng = self.model.random
        for n in self.network.neighbors(self):
            if n.condition == 0 and self.p.contagion_prob > rng.random():
#            if self.contagion_prob > rng.random():
#                n.condition = 1  # Infect susceptible peer
                n.safe = 0  # Infect susceptible peer

    def manage_crisis(self):
        rng = self.model.random
        if self.p.recovery_chance > rng.random():
            self.condition = 2  # Recover from infection
        elif self.p.liquidation_chance > rng.random():
            self.condition = 3
        else:
            pass


class Customer(ap.Agent):
    
    def setup(self):
        """ Initialize a new variable at agent creation. """
#        self.withdrawal_prob = self.p.withdrawal_prob
##        self.banks = self.model.banks
        self.total_deposits = self.p.total_banking_deposits / self.p.customers_population
        self.deposits = dict()
        for i, bank in enumerate(self.model.banks):
            self.deposits[bank]= 0

    def allocate_deposits(self, amount=None):
        if amount == None:
            amount = self.total_deposits
        rng = self.model.nprandom
#        self.deposits = dict()
        safe_banks = self.model.banks.select(self.model.banks.safe == 1)
        self.allocation = self.p.deposits_dispersion+10*rng.choice(np.eye(len(safe_banks)),1).flatten()
#        self.allocation = list(rng.choice(np.eye(self.p.bank_population),1).flatten())
        self.allocation = np.random.dirichlet(self.allocation) 
        for i, bank in enumerate(safe_banks):
            self.deposits[bank] += self.allocation[i]*amount  

           
    def withdraw_deposits(self):
        rng = self.model.nprandom
        for bank in self.model.banks:
#            if bank.safe ==0 and bank.condition !=1:
            if bank.safe ==0 and bank.condition !=3:
#                self.deposits[bank]= self.deposits[bank] * (1-self.withdrawal_rate)
                withdrawal = self.deposits[bank]*rng.binomial(1,self.p.withdrawal_prob)         
                self.deposits[bank] = self.deposits[bank] - withdrawal
                if withdrawal > 0:
                    self.allocate_deposits(withdrawal)                        


class CrisisModel(ap.Model):

    def setup(self):
        """ Initialize the agents and network of the model. """
        # Prepare a small-world network
        bank_graph = nx.watts_strogatz_graph(
            self.p.bank_population,
            self.p.number_of_neighbors,
            self.p.network_randomness)
        # Create agents and network
        self.banks = ap.AgentDList(model=self, objs=self.p.bank_population, cls=Bank)
        self.bank_network = self.banks.network = ap.Network(self, bank_graph)
        self.bank_network.add_agents(self.banks, self.bank_network.nodes)

        self.customers = ap.AgentDList(model=self, objs=self.p.customers_population, cls=Customer)
        self.customers.allocate_deposits()
        
        self.banks.calculate_initial_deposits()

        # Infect a random share of the population
        I0 = int(self.p.initial_unsafe_share * self.p.bank_population)
        self.banks.random(I0).safe = 0
        


    def update(self):
        """ Record variables after setup and each step. """

        # Record share of banks with each condition
        for i, c in enumerate(('S', 'I', 'R','L')):
            n_banks = len(self.banks.select(self.banks.condition == i))
            self[c] = n_banks / self.p.bank_population
            self.record(c)
            
        # Record safe banks
        safe_banks = len(self.banks.select(self.banks.safe == 1))
        self['safe']= safe_banks / self.p.bank_population
        self.record('safe')
        

        # Stop simulation if disease is gone
        if self.safe == 1:
            self.stop()

    def step(self):
        """ Define the model's events per simulation step. """

        self.banks.update_status()
        
        self.customers.withdraw_deposits()        
        
        self.banks.test_solvency()
                
        # Call 'spread run' for "infected" banks
        self.banks.select(self.banks.safe == 0).spread_crisis()
        
        self.banks.select(self.banks.condition == 1).manage_crisis()
        

    def end(self):
        """ Record evaluation measures at the end of the simulation. """

        # Record final evaluation measures
        self.report('Total share facing runs', self.I + self.R+self.L)
        self.report('Peak share facing runs', max(self.log['I']))
        self.report('Total share liquidated', self.L)
        self.report('Total share safe banks', self.safe)


#### Running a simulation

parameters = {
    'steps':10,
    'bank_population': 100,
    'customers_population':100000,
    'total_banking_deposits':1000,
    'contagion_prob': 0.1,
    'recovery_chance': 0.3,
    'liquidation_chance':0.05,
    'initial_unsafe_share': 0.01,
    'number_of_neighbors': 10,
    'network_randomness': 0.5,
    'withdrawal_prob':0.8,
    'solvency_threshold':0.8,
    'sunspot_prob':0.001,
    'deposits_dispersion':0.1
}

model = CrisisModel(parameters)
results = model.run()

print(results)

results.variables.CrisisModel

    
fig, ax = plt.subplots()    
crisis_stackplot(model.output.variables.CrisisModel, fig, ax)

color_dict = {0:'b', 1:'r', 2:'g', 3:'black'}
colors = [color_dict[c] for c in model.banks.condition]
nx.draw_circular(model.bank_network.graph, node_color=colors,node_size=10)

#%% Multiple Run Experiment

parameters = {
    'steps':10,
    'bank_population': 100,
    'customers_population':100000,
    'total_banking_deposits':1000,
    'contagion_prob': ap.Range(0.1, 0.7),
    'recovery_chance': ap.Range(0.1, 0.9),
    'liquidation_chance':ap.Range(0.05, 0.1),
    'initial_unsafe_share': 0.01,
    'number_of_neighbors': ap.IntRange(5, 20),
    'network_randomness': 0.5,
    'withdrawal_prob':0.8,
    'solvency_threshold':0.8,
    'sunspot_prob':0.001,
    'deposits_dispersion':0.1
}


sample = ap.Sample(
    parameters,
    n=64,
    method='saltelli',
    calc_second_order=False
)

exp = ap.Experiment(CrisisModel, sample, iterations=10)
results = exp.run()


results.save()

results.reporters.hist()

results.calc_sobol()

def plot_sobol(results):
    """ Bar plot of Sobol sensitivity indices. """

    sns.set()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    si_list = results.sensitivity.sobol.groupby(by='reporter')
    si_conf_list = results.sensitivity.sobol_conf.groupby(by='reporter')

    for (key, si), (_, err), ax in zip(si_list, si_conf_list, axs):
        si = si.droplevel('reporter')
        err = err.droplevel('reporter')
        si.plot.barh(xerr=err, title=key, ax=ax, capsize = 3)
        ax.set_xlim(0)

    axs[0].get_legend().remove()
    axs[1].set(ylabel=None, yticklabels=[])
    axs[1].tick_params(left=False)
    plt.tight_layout()

plot_sobol(results)

def plot_sensitivity(results):
    """ Show average simulation results for different parameter values. """

    sns.set()
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs = [i for j in axs for i in j] # Flatten list

    data = results.arrange_reporters().astype('float')
    params = results.parameters.sample.keys()

    for x, ax in zip(params, axs):
        for y in results.reporters.columns[1:]:
            sns.regplot(x=x, y=y, data=data, ax=ax, ci=99,
                        x_bins=15, fit_reg=False, label=y)
        ax.set_ylim(0,1)
        ax.set_ylabel('')
        ax.legend()

    plt.tight_layout()

plot_sensitivity(results)

results.parameters.sample.keys()
results.reporters.columns
