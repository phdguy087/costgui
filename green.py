# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:20:00 2023

@author: sh21b
"""

import streamlit as st
import functools
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.optimize import curve_fit
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

#Development of the UI
st.set_page_config(layout="wide")
st.title("iPlantGreenS2: Integrated Planning toolkit for Green Infrastructure Siting and Selection")

add_selectbox1 = st.sidebar.title("OPTIONS")
with st.sidebar:
    page_names = ['Single','Series']
    page= st.radio('Type of Implementation',page_names)
    
if page == 'Single':
    col1, col2 = st.columns([1, 3])
    with col1:
        options1=['Green Roof','Rain Barrel','Cistern']
        SCM_type= st.selectbox('Type of SCM:',options1)
        if SCM_type=='Green Roof':
            number1 = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Green Roof")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2):
                        objective_1 = 9881*(p1*p2)**0.267 
                        objective_2 = (98-(117*(2.718)**(-5.21*(p2))))
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }

                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])

                    mean_fitnesses = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses = simple_1d_fitness_func(pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2)
                    xdata= pp_solutions_fitnesses[:,3]
                    ydata= pp_solutions_fitnesses[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x +B*x**2 + C
                        return y
                    parameters, covariance = curve_fit(Gauss, xdata, ydata)
                    fit_A = parameters[0]
                    fit_B = parameters[1]
                    fit_C= parameters[2]
                    cost = round(fit_A*removal+fit_B*removal**2+fit_C,2)
                    atn= round((tn*removal)/100,2)
                    atp= round((tp*removal)/100,2)
                    ecost=327.3
                    ccost=round((cost-ecost)*0.46,2)
                    opcost=round((cost-ecost)*0.54,2)
                    mcost=round(cost*0.37,2)
                    lcost=round(cost*0.4,2)
                    eqcost=round(cost*0.16,2)
                    encost=round(cost*0.03,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df["Nitrogen Reduction (lb/yr)"]=df["Reduction (%)"]*(tn/100)
                    df["Phosphorus Reduction (lb/yr)"]=df["Reduction (%)"]*(tp/100)
                    tn_removal=(tn*removal)/100
                    tp_removal=(tp*removal)/100
                    tn_con=(tn*con_level)/100
                    tp_con=(tp*con_level)/100
                    with tab1:
                        fig=make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.1,subplot_titles=("Nitrogen Reduction","Phosphorus Reduction"))
                        fig.add_trace(go.Scatter(x=df["cost (USD)"],y=df["Nitrogen Reduction (lb/yr)"],mode='markers',name='Nitrogen Reduction',marker=dict(color='rgba(8,88,158, 0.6)')),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df["cost (USD)"],y=df["Phosphorus Reduction (lb/yr)"],mode='markers',name='Phosphorus Reduction',marker=dict(color='rgba(241,105,19, 0.6)')),row = 2, col = 1)
                       
                        fig.add_hline(y=tn_removal,name= 'Nitrogen Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 1, col = 1)
                        fig.add_hline(y=tp_removal,name= 'Phosphorus Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 2, col = 1)
                        fig.add_hrect(y0=tn_removal-tn_con, y1=tn_removal + tn_con, line_width=0, fillcolor="red", opacity=0.2,row = 1, col = 1)
                        fig.add_hrect(y0=tp_removal-tp_con, y1=tp_removal + tp_con, line_width=0, fillcolor="red", opacity=0.2,row = 2, col = 1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost) + '<br>Total Nitrogen Concentration = '+str(atn)+ ' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(103,169,207)',
                                    y=tn,
                                    x=df.iloc[0]["cost (USD)"],
                                    xanchor='left',row=1, col=1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost) + '<br>Total Phosphorus  Concentration = '+str(atp)+' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(253,208,162)',
                                    y=tp,
                                    x=df.iloc[0]["cost (USD)"],
                                    xanchor='left',row=2, col=1)
                        fig.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=1, col=1)
                        fig.update_xaxes(title_text="Cost (USD)",showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         titlefont=dict(
                                             family='Arial',
                                             size = 25,
                                            color= 'black'),
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=2, col=1)
                        fig.update_yaxes(title_text="Nitrogene Reduction (lb/year)",showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         titlefont=dict(
                                             family='Arial',
                                             size = 25,
                                            color= 'black'),
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=1, col=1)
                        fig.update_yaxes(title_text="Phosphorus Reduction (lb/year)",showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         titlefont=dict(
                                             family='Arial',
                                             size = 25,
                                            color= 'black'),
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=2, col=1)
                                         
                        
                        fig.update_layout(height=1200, width=1000,
                                          font=dict(
                                              family="Arial",
                                              size=30,
                                              color="Black"),
                                          showlegend=True,
                                          legend=dict(
                                             orientation="h",
                                             yanchor="bottom",
                                             y=1.02,
                                             xanchor="right",
                                             x=1,
                                             title_font_family="Times New Roman",
                                             font=dict(family="Arial",
                                             size=25,
                                             color="black")))   
                        fig.update_layout(legend= {'itemsizing': 'constant'})
                        st.plotly_chart(fig, use_container_width=False)
                        
                    with tab2:
                        st.dataframe(df)
                        
                    with tab3:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(65,182,196, 0.6)',
                            line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(34,94,168, 0.6)',
                        line=dict(color='rgba(34,94,168, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(116,196,118, 0.6)',
                        line=dict(color='rgba(116,196,118, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(0,109,44, 0.6)',
                        line=dict(color='rgba(0,109,44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(251,106,74, 0.6)',
                        line=dict(color='rgba(251,106,74, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(165,15,21, 0.6)',
                        line=dict(color='rgba(165,15,21, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(254,153,41, 0.6)',
                        line=dict(color='rgba(254,153,41, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(247,104,161, 0.6)',
                        line=dict(color='rgba(247,104,161, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(129,15,124, 0.6)',
                        line=dict(color='rgba(129,15,124, 1.0)', width=3)
                           )
                            ))

                        fig.update_layout(barmode='stack',
                                          
                                          font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                                xaxis=dict(
                             showline=False,
                             showgrid=False,
                             showticklabels=True,
                             linecolor='black',
                             title='Cost (USD)',
                             titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                             linewidth=2,
                             ticks='outside',
                             tickfont=dict(
                                 family='Arial',
                                 size=20,
                                 color='black',
                                 )),yaxis=dict(
                                     
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=False,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )),
                                     showlegend= True,
                                     legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1,
                                        xanchor="right",
                                        x=0.8,
                                        title_font_family="Arial",
                                        font=dict(family="Arial",
                                        size=25,
                                        color="black")))
                        st.plotly_chart(fig, use_container_width=True)
                        fig1= go.Figure()
                        fig1.add_trace(go.Scatter(x=[1,1,3,3,4,3,3,1], y=[6,8,8,9,7,5,6,6], fill="toself"))
                        fig1.add_trace(go.Scatter(x=[5,5,8,8,5], y=[5,9,9,5,5], fill="toself"))
                        fig1.add_trace(go.Scatter(x=[9,9,11,11,12,11,11,9], y=[6,8,8,9,7,5,6,6], fill="toself"))
                        fig1.add_trace(go.Scatter(x=[6.5,5,6,6,7,7,8,6.5], y=[0,1,1,4,4,1,1,0], fill="toself"))
                        fig1.add_annotation(
                                    text='Total Nitrogenr = '+str(tn) + 'lb/year <br>Total Phosphorus = '+str(tp)+' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor=None,
                                    y=0.85,
                                    x=0.07,
                                    xanchor='left')
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        
                        
                        
        if SCM_type=='Rain Barrel':
           number = st.number_input('Available Area(sft)')
           tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
           tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
           removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
           con_level = st.slider('Confidence interval', 0.0, 25.0)
           st.write(removal, '% Nutrient Reduction is needed')
           q=st.button('Run')
           if q:
               with col2:
                   st.subheader("Optimal Outcomes for Rain Barrel")
                   tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                   def simple_1d_fitness_func(p1,p2):
                       objective_1 = 9881*(p1*p2)**0.267 
                       objective_2 = (98-(117*(2.718)**(-5.21*(p2))))
                       return np.stack([p1,p2,objective_1,objective_2],axis=1)
                   config = {
                               "half_pop_size" : 50,
                               "problem_dim" : 2,
                               "gene_min_val" : -1,
                               "gene_max_val" : 1,
                               "mutation_power_ratio" : 0.05,
                               }

                   pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])

                   mean_fitnesses = []
                   for generation in range(30):
                       # evaluate pop 
                       fitnesses = simple_1d_fitness_func(pop,pop)
                       mean_fitnesses.append(np.mean(fitnesses,axis=0))
                       
                       # transition to next generation
                       pop = NSGA2_create_next_generation(pop,fitnesses,config)
                  
                   p1 = np.linspace(100,number,100)
                   p2= np.linspace(0.1,0.5,100)
                   pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2)
                   xdata= pp_solutions_fitnesses[:,3]
                   ydata= pp_solutions_fitnesses[:,2]
                   def Gauss(x, A,B,C):
                       y = A*x +B*x**2 + C
                       return y
                   parameters, covariance = curve_fit(Gauss, xdata, ydata)
                   fit_A = parameters[0]
                   fit_B = parameters[1]
                   fit_C= parameters[2]
                   cost = round(fit_A*removal+fit_B*removal**2+fit_C,2)
                   atn= round((tn*removal)/100,2)
                   atp= round((tp*removal)/100,2)
                   ecost=327.3
                   ccost=round((cost-ecost)*0.46,2)
                   opcost=round((cost-ecost)*0.54,2)
                   mcost=round(cost*0.37,2)
                   lcost=round(cost*0.4,2)
                   eqcost=round(cost*0.16,2)
                   encost=round(cost*0.03,2)
                   ocost=round(cost*0.04,2)
                   df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                   df["Nitrogen Reduction (lb/yr)"]=df["Reduction (%)"]*(tn/100)
                   df["Phosphorus Reduction (lb/yr)"]=df["Reduction (%)"]*(tp/100)
                   tn_removal=(tn*removal)/100
                   tp_removal=(tp*removal)/100
                   tn_con=(tn*con_level)/100
                   tp_con=(tp*con_level)/100
                   with tab1:
                       fig=make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.1,subplot_titles=("Nitrogen Reduction","Phosphorus Reduction"))
                       fig.add_trace(go.Scatter(x=df["cost (USD)"],y=df["Nitrogen Reduction (lb/yr)"],mode='markers',name='Nitrogen Reduction',marker=dict(color='rgba(8,88,158, 0.6)')),row = 1, col = 1)
                       fig.add_trace(go.Scatter(x=df["cost (USD)"],y=df["Phosphorus Reduction (lb/yr)"],mode='markers',name='Phosphorus Reduction',marker=dict(color='rgba(241,105,19, 0.6)')),row = 2, col = 1)
                      
                       fig.add_hline(y=tn_removal,name= 'Nitrogen Reduction',line=dict(color='firebrick', width=2,
                                                    dash='dash'),row = 1, col = 1)
                       fig.add_hline(y=tp_removal,name= 'Phosphorus Reduction',line=dict(color='firebrick', width=2,
                                                    dash='dash'),row = 2, col = 1)
                       fig.add_hrect(y0=tn_removal-tn_con, y1=tn_removal + tn_con, line_width=0, fillcolor="red", opacity=0.2,row = 1, col = 1)
                       fig.add_hrect(y0=tp_removal-tp_con, y1=tp_removal + tp_con, line_width=0, fillcolor="red", opacity=0.2,row = 2, col = 1)
                       fig.add_annotation(
                                   text='Total Cost = $'+str(cost) + '<br>Total Nitrogen Concentration = '+str(atn)+ ' lb/year',
                                   align = 'left',
                                   showarrow= False,
                                   xref='paper',
                                   yref='paper',
                                   font=dict(family='Arial',
                                             size=20,
                                             color='black'),
                                   bgcolor='rgb(103,169,207)',
                                   y=tn,
                                   x=df.iloc[0]["cost (USD)"],
                                   xanchor='left',row=1, col=1)
                       fig.add_annotation(
                                   text='Total Cost = $'+str(cost) + '<br>Total Phosphorus  Concentration = '+str(atp)+' lb/year',
                                   align = 'left',
                                   showarrow= False,
                                   xref='paper',
                                   yref='paper',
                                   font=dict(family='Arial',
                                             size=20,
                                             color='black'),
                                   bgcolor='rgb(253,208,162)',
                                   y=tp,
                                   x=df.iloc[0]["cost (USD)"],
                                   xanchor='left',row=2, col=1)
                       fig.update_xaxes(showline=True,
                                        showgrid=False,
                                        linecolor='black',
                                        linewidth=2,
                                        ticks='outside',
                                        tickfont=dict(
                                            family='Arial',
                                            size=20,
                                            color='black'),
                                        row=1, col=1)
                       fig.update_xaxes(title_text="Cost (USD)",showline=True,
                                        showgrid=False,
                                        linecolor='black',
                                        titlefont=dict(
                                            family='Arial',
                                            size = 25,
                                           color= 'black'),
                                        linewidth=2,
                                        ticks='outside',
                                        tickfont=dict(
                                            family='Arial',
                                            size=20,
                                            color='black'),
                                        row=2, col=1)
                       fig.update_yaxes(title_text="Nitrogene Reduction (lb/year)",showline=True,
                                        showgrid=False,
                                        linecolor='black',
                                        titlefont=dict(
                                            family='Arial',
                                            size = 25,
                                           color= 'black'),
                                        linewidth=2,
                                        ticks='outside',
                                        tickfont=dict(
                                            family='Arial',
                                            size=20,
                                            color='black'),
                                        row=1, col=1)
                       fig.update_yaxes(title_text="Phosphorus Reduction (lb/year)",showline=True,
                                        showgrid=False,
                                        linecolor='black',
                                        titlefont=dict(
                                            family='Arial',
                                            size = 25,
                                           color= 'black'),
                                        linewidth=2,
                                        ticks='outside',
                                        tickfont=dict(
                                            family='Arial',
                                            size=20,
                                            color='black'),
                                        row=2, col=1)
                                        
                       
                       fig.update_layout(height=1200, width=1000,
                                         font=dict(
                                             family="Arial",
                                             size=30,
                                             color="Black"),
                                         showlegend=True,
                                         legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1,
                                            title_font_family="Times New Roman",
                                            font=dict(family="Arial",
                                            size=25,
                                            color="black")))   
                       fig.update_layout(legend= {'itemsizing': 'constant'})
                       st.plotly_chart(fig, use_container_width=False)
                       
                   with tab2:
                       st.dataframe(df)
                       
                   with tab3:
                       fig = go.Figure()
                       fig.add_trace(go.Bar(
                          y=['Total Cost (Life Cycle Stages)'],
                          x=[0],
                          name='Planning Cost',
                          orientation='h',
                          marker=dict(
                           color='rgba(65,182,196, 0.6)',
                           line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                 )
                                 ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Stages)'],
                       x=[ccost],
                       name='Construction Cost',
                       orientation='h',
                       marker=dict(
                       color='rgba(34,94,168, 0.6)',
                       line=dict(color='rgba(34,94,168, 1.0)', width=3)
                          )
                           ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Stages)'],
                       x=[opcost],
                       name='Operations and Maintenance Cost',
                       orientation='h',
                       marker=dict(
                       color='rgba(116,196,118, 0.6)',
                       line=dict(color='rgba(116,196,118, 1.0)', width=3)
                          )
                           ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Stages)'],
                       x=[ecost],
                       name='End of Life Cost',
                       orientation='h',
                       marker=dict(
                       color='rgba(0,109,44, 0.6)',
                       line=dict(color='rgba(0,109,44, 1.0)', width=3)
                          )
                           ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Cost Type)'],
                       x=[mcost],
                       name='Materials',
                       orientation='h',
                       marker=dict(
                       color='rgba(251,106,74, 0.6)',
                       line=dict(color='rgba(251,106,74, 1.0)', width=3)
                          )
                           ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Cost Type)'],
                       x=[lcost],
                       name='Labors',
                       orientation='h',
                       marker=dict(
                       color='rgba(165,15,21, 0.6)',
                       line=dict(color='rgba(165,15,21, 1.0)', width=3)
                          )
                           ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Cost Type)'],
                       x=[eqcost],
                       name='Equipments',
                       orientation='h',
                       marker=dict(
                       color='rgba(254,153,41, 0.6)',
                       line=dict(color='rgba(254,153,41, 1.0)', width=3)
                          )
                           ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Cost Type)'],
                       x=[encost],
                       name='Energy',
                       orientation='h',
                       marker=dict(
                       color='rgba(247,104,161, 0.6)',
                       line=dict(color='rgba(247,104,161, 1.0)', width=3)
                          )
                           ))
                       fig.add_trace(go.Bar(
                       y=['Total Cost (Life Cycle Cost Type)'],
                       x=[ocost],
                       name='Others',
                       orientation='h',
                       marker=dict(
                       color='rgba(129,15,124, 0.6)',
                       line=dict(color='rgba(129,15,124, 1.0)', width=3)
                          )
                           ))

                       fig.update_layout(barmode='stack',
                                         
                                         font=dict(
                            family="Arial",
                            size=25,
                            color="Black"),
                               xaxis=dict(
                            showline=False,
                            showgrid=False,
                            showticklabels=True,
                            linecolor='black',
                            title='Cost (USD)',
                            titlefont=dict(
                                family='Arial',
                                size = 25,
                                color= 'black'),
                            linewidth=2,
                            ticks='outside',
                            tickfont=dict(
                                family='Arial',
                                size=20,
                                color='black',
                                )),yaxis=dict(
                                    
                                    titlefont=dict(
                                family='Arial',
                                size = 25,
                                color= 'black'),
                                    showline=False,
                                    showgrid=False,
                                    showticklabels=True,
                                    linecolor='black',
                                    linewidth=2,
                                    ticks='outside',
                                    tickfont=dict(
                                        family='Arial',
                                        size=20,
                                        color='black',
                                        )),
                                    showlegend= True,
                                    legend=dict(
                                       orientation="h",
                                       yanchor="bottom",
                                       y=1,
                                       xanchor="right",
                                       x=0.8,
                                       title_font_family="Arial",
                                       font=dict(family="Arial",
                                       size=25,
                                       color="black")))
                       st.plotly_chart(fig, use_container_width=True)
                       fig1= go.Figure()
                       fig1.add_trace(go.Scatter(x=[1,1,3,3,4,3,3,1], y=[6,8,8,9,7,5,6,6], fill="toself"))
                       fig1.add_trace(go.Scatter(x=[5,5,8,8,5], y=[5,9,9,5,5], fill="toself"))
                       fig1.add_trace(go.Scatter(x=[9,9,11,11,12,11,11,9], y=[6,8,8,9,7,5,6,6], fill="toself"))
                       fig1.add_trace(go.Scatter(x=[6.5,5,6,6,7,7,8,6.5], y=[0,1,1,4,4,1,1,0], fill="toself"))
                       fig1.add_annotation(
                                   text='Total Nitrogenr = '+str(tn) + 'lb/year <br>Total Phosphorus = '+str(tp)+' lb/year',
                                   align = 'left',
                                   showarrow= False,
                                   xref='paper',
                                   yref='paper',
                                   font=dict(family='Arial',
                                             size=20,
                                             color='black'),
                                   bgcolor=None,
                                   y=0.85,
                                   x=0.07,
                                   xanchor='left')
                       st.plotly_chart(fig1, use_container_width=True)
                       
                       
        else:
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for cistern")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2):
                        objective_1 = 9881*(p1*p2)**0.267 
                        objective_2 = (98-(117*(2.718)**(-5.21*(p2))))
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }

                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])

                    mean_fitnesses = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses = simple_1d_fitness_func(pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2)
                    xdata= pp_solutions_fitnesses[:,3]
                    ydata= pp_solutions_fitnesses[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x +B*x**2 + C
                        return y
                    parameters, covariance = curve_fit(Gauss, xdata, ydata)
                    fit_A = parameters[0]
                    fit_B = parameters[1]
                    fit_C= parameters[2]
                    cost = round(fit_A*removal+fit_B*removal**2+fit_C,2)
                    atn= round((tn*removal)/100,2)
                    atp= round((tp*removal)/100,2)
                    ecost=327.3
                    ccost=round((cost-ecost)*0.46,2)
                    opcost=round((cost-ecost)*0.54,2)
                    mcost=round(cost*0.37,2)
                    lcost=round(cost*0.4,2)
                    eqcost=round(cost*0.16,2)
                    encost=round(cost*0.03,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df["Nitrogen Reduction (lb/yr)"]=df["Reduction (%)"]*(tn/100)
                    df["Phosphorus Reduction (lb/yr)"]=df["Reduction (%)"]*(tp/100)
                    tn_removal=(tn*removal)/100
                    tp_removal=(tp*removal)/100
                    tn_con=(tn*con_level)/100
                    tp_con=(tp*con_level)/100
                    with tab1:
                        fig=make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.1,subplot_titles=("Nitrogen Reduction","Phosphorus Reduction"))
                        fig.add_trace(go.Scatter(x=df["cost (USD)"],y=df["Nitrogen Reduction (lb/yr)"],mode='markers',name='Nitrogen Reduction',marker=dict(color='rgba(8,88,158, 0.6)')),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df["cost (USD)"],y=df["Phosphorus Reduction (lb/yr)"],mode='markers',name='Phosphorus Reduction',marker=dict(color='rgba(241,105,19, 0.6)')),row = 2, col = 1)
                       
                        fig.add_hline(y=tn_removal,name= 'Nitrogen Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 1, col = 1)
                        fig.add_hline(y=tp_removal,name= 'Phosphorus Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 2, col = 1)
                        fig.add_hrect(y0=tn_removal-tn_con, y1=tn_removal + tn_con, line_width=0, fillcolor="red", opacity=0.2,row = 1, col = 1)
                        fig.add_hrect(y0=tp_removal-tp_con, y1=tp_removal + tp_con, line_width=0, fillcolor="red", opacity=0.2,row = 2, col = 1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost) + '<br>Total Nitrogen Concentration = '+str(atn)+ ' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(103,169,207)',
                                    y=tn,
                                    x=df.iloc[0]["cost (USD)"],
                                    xanchor='left',row=1, col=1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost) + '<br>Total Phosphorus  Concentration = '+str(atp)+' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(253,208,162)',
                                    y=tp,
                                    x=df.iloc[0]["cost (USD)"],
                                    xanchor='left',row=2, col=1)
                        fig.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=1, col=1)
                        fig.update_xaxes(title_text="Cost (USD)",showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         titlefont=dict(
                                             family='Arial',
                                             size = 25,
                                            color= 'black'),
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=2, col=1)
                        fig.update_yaxes(title_text="Nitrogene Reduction (lb/year)",showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         titlefont=dict(
                                             family='Arial',
                                             size = 25,
                                            color= 'black'),
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=1, col=1)
                        fig.update_yaxes(title_text="Phosphorus Reduction (lb/year)",showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         titlefont=dict(
                                             family='Arial',
                                             size = 25,
                                            color= 'black'),
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=2, col=1)
                                         
                        
                        fig.update_layout(height=1200, width=1000,
                                          font=dict(
                                              family="Arial",
                                              size=30,
                                              color="Black"),
                                          showlegend=True,
                                          legend=dict(
                                             orientation="h",
                                             yanchor="bottom",
                                             y=1.02,
                                             xanchor="right",
                                             x=1,
                                             title_font_family="Times New Roman",
                                             font=dict(family="Arial",
                                             size=25,
                                             color="black")))   
                        fig.update_layout(legend= {'itemsizing': 'constant'})
                        st.plotly_chart(fig, use_container_width=False)
                        
                    with tab2:
                        st.dataframe(df)
                        
                    with tab3:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(65,182,196, 0.6)',
                            line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(34,94,168, 0.6)',
                        line=dict(color='rgba(34,94,168, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(116,196,118, 0.6)',
                        line=dict(color='rgba(116,196,118, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(0,109,44, 0.6)',
                        line=dict(color='rgba(0,109,44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(251,106,74, 0.6)',
                        line=dict(color='rgba(251,106,74, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(165,15,21, 0.6)',
                        line=dict(color='rgba(165,15,21, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(254,153,41, 0.6)',
                        line=dict(color='rgba(254,153,41, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(247,104,161, 0.6)',
                        line=dict(color='rgba(247,104,161, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(129,15,124, 0.6)',
                        line=dict(color='rgba(129,15,124, 1.0)', width=3)
                           )
                            ))

                        fig.update_layout(barmode='stack',
                                          
                                          font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                                xaxis=dict(
                             showline=False,
                             showgrid=False,
                             showticklabels=True,
                             linecolor='black',
                             title='Cost (USD)',
                             titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                             linewidth=2,
                             ticks='outside',
                             tickfont=dict(
                                 family='Arial',
                                 size=20,
                                 color='black',
                                 )),yaxis=dict(
                                     
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=False,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )),
                                     showlegend= True,
                                     legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1,
                                        xanchor="right",
                                        x=0.8,
                                        title_font_family="Arial",
                                        font=dict(family="Arial",
                                        size=25,
                                        color="black")))
                        st.plotly_chart(fig, use_container_width=True)
                        fig1= go.Figure()
                        fig1.add_trace(go.Scatter(x=[1,1,3,3,4,3,3,1], y=[6,8,8,9,7,5,6,6], fill="toself"))
                        fig1.add_trace(go.Scatter(x=[5,5,8,8,5], y=[5,9,9,5,5], fill="toself"))
                        fig1.add_trace(go.Scatter(x=[9,9,11,11,12,11,11,9], y=[6,8,8,9,7,5,6,6], fill="toself"))
                        fig1.add_trace(go.Scatter(x=[6.5,5,6,6,7,7,8,6.5], y=[0,1,1,4,4,1,1,0], fill="toself"))
                        fig1.add_annotation(
                                    text='Total Nitrogenr = '+str(tn) + 'lb/year <br>Total Phosphorus = '+str(tp)+' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor=None,
                                    y=0.85,
                                    x=0.07,
                                    xanchor='left')
                        st.plotly_chart(fig1, use_container_width=True)
