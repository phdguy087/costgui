# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:50:22 2023

@author: apurb
"""

import streamlit as st
import functools
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
def dominates(fitnesses_1,fitnesses_2):
    # fitnesses_1 is a array of objectives of solution 1 [objective1, objective2 ...]
    larger_or_equal = fitnesses_1 >= fitnesses_2
    larger = fitnesses_1 > fitnesses_2
    if np.all(larger_or_equal) and np.any(larger):
        return True
    return False

pop_size = 100
all_fitnesses = np.random.rand(pop_size,2)

fronts = []

remaining_indicies = list(range(pop_size))
while True:
    
    non_dominated_indicies = []
    
    for i in remaining_indicies:
        dominated = False
        for j in remaining_indicies:
            if dominates(all_fitnesses[j],all_fitnesses[i]):
                dominated = True
                break
                
        if dominated is False:
            non_dominated_indicies.append(i)
            
    #print("Front: ",non_dominated_indicies)
    
    # remove current front from remaining indicies
    fronts.append(non_dominated_indicies)
    remaining_indicies = [x for x in remaining_indicies if x not in non_dominated_indicies] 
    
    if len(remaining_indicies) == 0:
        #print("Done")
        break

def calculate_pareto_fronts(fitnesses):
    
    # Calculate dominated set for each individual
    domination_sets = []
    domination_counts = []
    for fitnesses_1 in fitnesses:
        current_dimination_set = set()
        domination_counts.append(0)
        for i,fitnesses_2 in enumerate(fitnesses):
            if dominates(fitnesses_1,fitnesses_2):
                current_dimination_set.add(i)
            elif dominates(fitnesses_2,fitnesses_1):
                domination_counts[-1] += 1

        domination_sets.append(current_dimination_set)

    domination_counts = np.array(domination_counts)
    fronts = []
    while True:
        current_front = np.where(domination_counts==0)[0]
        if len(current_front) == 0:
            #print("Done")
            break
        #print("Front: ",current_front)
        fronts.append(current_front)

        for individual in current_front:
            domination_counts[individual] = -1 # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
            dominated_by_current_set = domination_sets[individual]
            for dominated_by_current in dominated_by_current_set:
                domination_counts[dominated_by_current] -= 1
            
    return fronts

# We use all_fitnesses defined in the previous example 
fronts = calculate_pareto_fronts(all_fitnesses)

def calculate_crowding_metrics(fitnesses,fronts):
    
    num_objectives = fitnesses.shape[1]
    num_individuals = fitnesses.shape[0]
    
    # Normalise each objectives, so they are in the range [0,1]
    # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
    normalized_fitnesses = np.zeros_like(fitnesses)
    for objective_i in range(num_objectives):
        min_val = np.min(fitnesses[:,objective_i])
        max_val = np.max(fitnesses[:,objective_i])
        val_range = max_val - min_val
        normalized_fitnesses[:,objective_i] = (fitnesses[:,objective_i] - min_val) / val_range
    
    fitnesses = normalized_fitnesses
    crowding_metrics = np.zeros(num_individuals)

    for front in fronts:
        for objective_i in range(num_objectives):
            
            sorted_front = sorted(front,key = lambda x : fitnesses[x,objective_i])
            
            crowding_metrics[sorted_front[0]] = np.inf
            crowding_metrics[sorted_front[-1]] = np.inf
            if len(sorted_front) > 2:
                for i in range(1,len(sorted_front)-1):
                    crowding_metrics[sorted_front[i]] += fitnesses[sorted_front[i+1],objective_i] - fitnesses[sorted_front[i-1],objective_i]

    return  crowding_metrics

# Let us plot the crowding metric for the previous example
crowding_metrics = calculate_crowding_metrics(all_fitnesses,fronts)
crowding_metrics[crowding_metrics == np.inf] = np.max(crowding_metrics[crowding_metrics != np.inf])  # replace inf with max


def fronts_to_nondomination_rank(fronts):
    nondomination_rank_dict = {}
    for i,front in enumerate(fronts):
        for x in front:   
            nondomination_rank_dict[x] = i
    return nondomination_rank_dict
        

def nondominated_sort(nondomination_rank_dict,crowding):
    
    num_individuals = len(crowding)
    indicies = list(range(num_individuals))

    def nondominated_compare(a,b):
        # returns 1 if a dominates b, or if they equal, but a is less crowded
        # return -1 if b dominates a, or if they equal, but b is less crowded
        # returns 0 if they are equal in every sense
        
        
        if nondomination_rank_dict[a] > nondomination_rank_dict[b]:  # domination rank, smaller better
            return -1
        elif nondomination_rank_dict[a] < nondomination_rank_dict[b]:
            return 1
        else:
            if crowding[a] < crowding[b]:   # crowding metrics, larger better
                return -1
            elif crowding[a] > crowding[b]:
                return 1
            else:
                return 0

    non_domiated_sorted_indicies = sorted(indicies,key = functools.cmp_to_key(nondominated_compare),reverse=True) # decreasing order, the best is the first
    return non_domiated_sorted_indicies

# Test
nondomination_rank_dict = fronts_to_nondomination_rank(fronts)
sorted_indicies = nondominated_sort(nondomination_rank_dict,crowding_metrics)

def touranment_selection(num_parents,num_offspring):
    offspring_parents = []
    for _ in range(num_offspring):
        contestants = np.random.randint(0,num_parents,2) # generate 2 random numbers, take the smaller (parent list is already sorted, smaller index, better)
        winner = np.min(contestants)
        offspring_parents.append(winner)
    
    return offspring_parents

# simple mutation
def get_mutated_copy(parent,min_val,max_val,mutation_power_ratio):
    mutation_power = (max_val - min_val) * mutation_power_ratio
    offspring = parent.copy()
    offspring += np.random.normal(0,mutation_power,size = offspring.shape)
    offspring = np.clip(offspring,min_val,max_val)
    return offspring



def NSGA2_create_next_generation(pop,fitnesses,config):
    
    # algorithm and task parameters
    half_pop_size = config["half_pop_size"]
    problem_dim = config["problem_dim"]
    gene_min_val = config["gene_min_val"]
    gene_max_val = config["gene_max_val"]
    mutation_power_ratio = config["mutation_power_ratio"]

    # calculate the pareto fronts and crowding metrics
    fronts = calculate_pareto_fronts(fitnesses)
    nondomination_rank_dict = fronts_to_nondomination_rank(fronts)
    
    crowding = calculate_crowding_metrics(fitnesses,fronts)
    
    # Sort the population
    non_domiated_sorted_indicies = nondominated_sort(nondomination_rank_dict,crowding)
    
    # The better half of the population survives to the next generation and have a chance to reproduce
    # The rest of the population is discarded
    surviving_individuals = pop[non_domiated_sorted_indicies[:half_pop_size]]
    #print(len(surviving_individuals))
    reproducing_individual_indicies = touranment_selection(num_parents=half_pop_size,num_offspring=half_pop_size)
    offsprings = np.array([get_mutated_copy(surviving_individuals[i],gene_min_val,gene_max_val,mutation_power_ratio) for i in reproducing_individual_indicies])
    
    new_pop = np.concatenate([surviving_individuals,offsprings])  # concatenate the 2 lists
    return new_pop

st.title("SCM PLANNING TOOL")
# Using object notation
add_selectbox = st.sidebar.title("OPTIONS")

# Using "with" notation
with st.sidebar:
    page_names = ['Bioretention','Dry Pond','Green Roof','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filterbed','Wet Pond']
    page= st.radio('Type of SCM',page_names)
    

if page == 'Green Roof':
    
    removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
    con_level = st.slider('Confidence interval', 0.0, 25.0)
    st.write(removal, '% Nutrient Reduction is needed')
    q=st.button('Run')
    if q:
        st.subheader("Optimal Outcomes for Green Roof")
        def simple_1d_fitness_func(p1,p2):
            objective_1 = 9881*(p1*p2)**0.267 
            objective_2 = (98-(117*(2.718)**(-5.21*(p2))))
            return np.stack([objective_1,objective_2],axis=1)
        config = {
            "half_pop_size" : 100,
            "problem_dim" : 2,
            "gene_min_val" : -1,
            "gene_max_val" : 1,
            "mutation_power_ratio" : 0.05,
        }

        pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])

        mean_fitnesses = []
        for generation in range(1):
            
            # evaluate pop
            fitnesses = simple_1d_fitness_func(pop,pop)
            mean_fitnesses.append(np.mean(fitnesses,axis=0))
            
            # transition to next generation
            pop = NSGA2_create_next_generation(pop,fitnesses,config)
            
            p1 = np.linspace(100,1400,100)
            p2= np.linspace(0.1,0.5,100)
            pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2)
            
            fig2 = make_subplots()
            fig2.add_trace(
                go.Scatter(x=pp_solutions_fitnesses[:, 0], y=pp_solutions_fitnesses[:, 1],name='Pareto-Optimal Line' ,line=dict(color='royalblue', width=2),mode='lines+markers')
                )
            fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                          dash='dash'))
            fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
            fig2.update_layout(height=600, width=800,  
              
             showlegend=True,
             xaxis_title='Cost (USD)',yaxis_title='Nutrient Reduction (%)',
             font=dict(
                 family="Arial",
                 size=18,
                 color="Black"),
             xaxis=dict(
                    showline=True,
                    showgrid=False,
                    showticklabels=True,
                    linecolor='black',
                    linewidth=2,
                    ticks='outside',
                    tickfont=dict(
                        family='Arial',
                        size=12,
                        color='black',
                    )),yaxis=dict(
                           showline=True,
                           showgrid=False,
                           showticklabels=True,
                           linecolor='black',
                           linewidth=2,
                           ticks='outside',
                           tickfont=dict(
                               family='Arial',
                               size=12,
                               color='black',
                           )))
            st.plotly_chart(fig2, use_container_width=True)
            
            with st.expander("See explanation"):
                st.write("The figure above shows total life cycle cost of the implementation of the SCM with respect to the reduction of nutrients")



    