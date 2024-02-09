# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:31:42 2023

@author: apurb
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
st.title("iPlantGreenS\u00b2 : Integrated Planning toolkit for Green Infrastructure Siting and Selection")

add_selectbox1 = st.sidebar.title("OPTIONS")
with st.sidebar:
    page_names = ['Single','Series']
    page= st.radio('Type of Implementation',page_names)
    
if page == 'Single':
    col1, col2 = st.columns([1, 3])
    with col1:
        options1=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond','Overview']
        SCM_type= st.selectbox('Type of SCM:',options1)
        if SCM_type=='Bioretention':
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2):
                        objective_1 = 29631*(p1*p2)**0.026 
                        objective_2 = (98-(117.1*(2.718)**(-5.21*(p2))))
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
                        #st.plotly_chart(fig1, use_container_width=True)
                        
        elif SCM_type=='Dry Pond':
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Dry Pond")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2):
                        objective_1 = 10525*(p1*p2)**0.29 
                        objective_2 = (98.26-(109.04*(2.718)**(-5.75*(p2))))
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
                    ecost=1614.87
                    ccost=round((cost-ecost)*0.43,2)
                    opcost=round((cost-ecost)*0.57,2)
                    mcost=round(cost*0.31,2)
                    lcost=round(cost*0.26,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.09,2)
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
                            color='rgba(166,206,227, 0.6)',
                            
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(31,120,180, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(178,223,138, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(51,160,44, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(251,154,153, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(227,26,28, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(255,127,0, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(202,178,214, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(106,61,154, 0.6)'
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        elif SCM_type=='Constructed Wetland':
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal_n = st.slider('Required Total Nitrogene Reduction', 0.0, 100.0, 0.5)
            removal_p = st.slider('Required Total Phosphorus Reduction', 0.0, 100.0, 0.5)
            
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            
            q=st.button('Run')
            
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Constructed Wetland")
                    tab1,tab2,tab3= st.tabs(["graph","Table","Cost"])
                    def simple_1d_fitness_func_tn(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503 
                        objective_2 = (4389.78*(p2)**0.012)-4266.26
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503
                        objective_2 = (260.665*(p2)**0.0285)-223.36
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    p1 = np.linspace(300,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = round(fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1,2) 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = round(fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2,2)
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=966.5
                    ccost1=round((cost1-ecost)*0.46,2)
                    opcost1=round((cost1-ecost)*0.54,2)
                    mcost1=round(cost1*0.32,2)
                    lcost1=round(cost1*0.25,2)
                    eqcost1=round(cost1*0.30,2)
                    encost1=round(cost1*0.09,2)
                    ocost1=round(cost1*0.04,2)
                    ccost2=round((cost2-ecost)*0.46,2)
                    opcost2=round((cost2-ecost)*0.54,2)
                    mcost2=round(cost2*0.32,2)
                    lcost2=round(cost2*0.25,2)
                    eqcost2=round(cost2*0.30,2)
                    encost2=round(cost2*0.09,2)
                    ocost2=round(cost2*0.04,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Nitrogen Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Phosphorus Reduction (%)"])
                    df1["Nitrogen Reduction (lb/yr)"]=df1["Nitrogen Reduction (%)"]*(tn/100)
                    df2["Phosphorus Reduction (lb/yr)"]=df2["Phosphorus Reduction (%)"]*(tp/100)
                    df1["Phosphorus Reduction (%)"]=df2["Phosphorus Reduction (%)"]
                    df1["Phosphorus Reduction (lb/yr)"]=df2["Phosphorus Reduction (lb/yr)"]
                    tn_removal=(tn*removal_n)/100
                    tp_removal=(tp*removal_p)/100
                    tn_con=(tn*con_level)/100
                    tp_con=(tp*con_level)/100
                    with tab1:
                        fig=make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.1,subplot_titles=("Nitrogen Reduction","Phosphorus Reduction"))
                        fig.add_trace(go.Scatter(x=df1["cost (USD)"],y=df1["Nitrogen Reduction (lb/yr)"],mode='markers',name='Nitrogen Reduction',marker=dict(color='rgba(8,88,158, 0.6)')),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df2["cost (USD)"],y=df2["Phosphorus Reduction (lb/yr)"],mode='markers',name='Phosphorus Reduction',marker=dict(color='rgba(241,105,19, 0.6)')),row = 2, col = 1)
                       
                        fig.add_hline(y=tn_removal,name= 'Nitrogen Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 1, col = 1)
                        fig.add_hline(y=tp_removal,name= 'Phosphorus Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 2, col = 1)
                        fig.add_hrect(y0=tn_removal-tn_con, y1=tn_removal + tn_con, line_width=0, fillcolor="red", opacity=0.2,row = 1, col = 1)
                        fig.add_hrect(y0=tp_removal-tp_con, y1=tp_removal + tp_con, line_width=0, fillcolor="red", opacity=0.2,row = 2, col = 1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost1) + '<br>Total Nitrogen Concentration = '+str(atn)+ ' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(103,169,207)',
                                    y=tn,
                                    x=df1.iloc[0]["cost (USD)"],
                                    xanchor='left',row=1, col=1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost2) + '<br>Total Phosphorus  Concentration = '+str(atp)+' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(253,208,162)',
                                    y=tp,
                                    x=df2.iloc[0]["cost (USD)"],
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
                        st.dataframe(df1)
                    
                    with tab3:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                        fig1 = go.Figure()
                        fig1.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
                           )
                            ))

                        fig1.update_layout(barmode='stack',
                                          
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
                                         )))
                        st.plotly_chart(fig1, use_container_width=True)
        elif SCM_type=='Grassed Swale':
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Grassed Swale")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2):
                        objective_1 = 42504*(p1*p2)**0.0344 
                        objective_2 = (97.7936-(107.28*(2.718)**(-5.85*(p2))))
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
                            color='rgba(166,206,227, 0.6)',
                            
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(31,120,180, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(178,223,138, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(51,160,44, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(251,154,153, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(227,26,28, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(255,127,0, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(202,178,214, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(106,61,154, 0.6)'
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        elif SCM_type=='Infiltration Trench':
        
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Infiltration Trench")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2):
                        objective_1 = 27632*(p1*p2)**0.0431 
                        objective_2 = (63767.5*(p2)**0.000285)-63679.2
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
                            color='rgba(166,206,227, 0.6)',
                            
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(31,120,180, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(178,223,138, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(51,160,44, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(251,154,153, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(227,26,28, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(255,127,0, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(202,178,214, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(106,61,154, 0.6)'
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        elif SCM_type=='Porous Pavement':
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Porous Pavement")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2):
                        objective_1 = 40540*(p1*p2)**0.0327 
                        objective_2 = (97.9016-(105.3*(2.718)**(-5.51*(p2))))
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
                    for generation in range(10):
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
                            color='rgba(166,206,227, 0.6)',
                            
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(31,120,180, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(178,223,138, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(51,160,44, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(251,154,153, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(227,26,28, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(255,127,0, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(202,178,214, 0.6)',
                        
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(106,61,154, 0.6)'
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        elif SCM_type=='Vegetative Filter Bed':
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal_n = st.slider('Required Total Nitrogene Reduction', 0.0, 100.0, 0.5)
            removal_p = st.slider('Required Total Phosphorus Reduction', 0.0, 100.0, 0.5)
            
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            
            q=st.button('Run')
            
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2):
                        objective_1 = 687.5*(p1*p2)**0.59 
                        objective_2 = (584.706*(p2)**0.012)-560.448
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2):
                        objective_1 = 687.5*(p1*p2)**0.59 
                        objective_2 = (29.031*(p2)**0.17)+ 8.47
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=327.3
                    ccost1=round((cost1-ecost)*0.46,2)
                    opcost1=round((cost1-ecost)*0.54,2)
                    mcost1=round(cost1*0.37,2)
                    lcost1=round(cost1*0.4,2)
                    eqcost1=round(cost1*0.16,2)
                    encost1=round(cost1*0.03,2)
                    ocost1=round(cost1*0.04,2)
                    ccost2=round((cost2-ecost)*0.46,2)
                    opcost2=round((cost2-ecost)*0.54,2)
                    mcost2=round(cost2*0.37,2)
                    lcost2=round(cost2*0.4,2)
                    eqcost2=round(cost2*0.16,2)
                    encost2=round(cost2*0.03,2)
                    ocost2=round(cost2*0.04,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Nitrogen Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Phosphorus Reduction (%)"])
                    df1["Nitrogen Reduction (lb/yr)"]=df1["Nitrogen Reduction (%)"]*(tn/100)
                    df2["Phosphorus Reduction (lb/yr)"]=df2["Phosphorus Reduction (%)"]*(tp/100)
                    df1["Phosphorus Reduction (%)"]=df2["Phosphorus Reduction (%)"]
                    df1["Phosphorus Reduction (lb/yr)"]=df2["Phosphorus Reduction (lb/yr)"]
                    tn_removal=(tn*removal_n)/100
                    tp_removal=(tp*removal_p)/100
                    tn_con=(tn*con_level)/100
                    tp_con=(tp*con_level)/100
                    with tab1:
                        fig=make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.1,subplot_titles=("Nitrogen Reduction","Phosphorus Reduction"))
                        fig.add_trace(go.Scatter(x=df1["cost (USD)"],y=df1["Nitrogen Reduction (lb/yr)"],mode='markers',name='Nitrogen Reduction',marker=dict(color='rgba(8,88,158, 0.6)')),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df2["cost (USD)"],y=df2["Phosphorus Reduction (lb/yr)"],mode='markers',name='Phosphorus Reduction',marker=dict(color='rgba(241,105,19, 0.6)')),row = 2, col = 1)
                       
                        fig.add_hline(y=tn_removal,name= 'Nitrogen Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 1, col = 1)
                        fig.add_hline(y=tp_removal,name= 'Phosphorus Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 2, col = 1)
                        fig.add_hrect(y0=tn_removal-tn_con, y1=tn_removal + tn_con, line_width=0, fillcolor="red", opacity=0.2,row = 1, col = 1)
                        fig.add_hrect(y0=tp_removal-tp_con, y1=tp_removal + tp_con, line_width=0, fillcolor="red", opacity=0.2,row = 2, col = 1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost1) + '<br>Total Nitrogen Concentration = '+str(atn)+ ' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(103,169,207)',
                                    y=tn,
                                    x=df1.iloc[0]["cost (USD)"],
                                    xanchor='left',row=1, col=1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost2) + '<br>Total Phosphorus  Concentration = '+str(atp)+' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(253,208,162)',
                                    y=tp,
                                    x=df2.iloc[0]["cost (USD)"],
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
                        st.dataframe(df1)
                    
                    with tab3:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                        fig1 = go.Figure()
                        fig1.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
                           )
                            ))

                        fig1.update_layout(barmode='stack',
                                          
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
                                         )))
                        st.plotly_chart(fig1, use_container_width=True)
            
        elif SCM_type== 'Wet Pond':
            
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal_n = st.slider('Required Total Nitrogene Reduction', 0.0, 100.0, 0.5)
            removal_p = st.slider('Required Total Phosphorus Reduction', 0.0, 100.0, 0.5)
            
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            
            q=st.button('Run')
            
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Wet Pond")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph","Tabl","Cost"])
                    def simple_1d_fitness_func_tn(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503 
                        objective_2 = (4389.78*(p2)**0.012)-4186.26
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503
                        objective_2 = (260.665*(p2)**0.0285)-223.36
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=327.3
                    ccost1=round((cost1-ecost)*0.46,2)
                    opcost1=round((cost1-ecost)*0.54,2)
                    mcost1=round(cost1*0.37,2)
                    lcost1=round(cost1*0.4,2)
                    eqcost1=round(cost1*0.16,2)
                    encost1=round(cost1*0.03,2)
                    ocost1=round(cost1*0.04,2)
                    ccost2=round((cost2-ecost)*0.46,2)
                    opcost2=round((cost2-ecost)*0.54,2)
                    mcost2=round(cost2*0.37,2)
                    lcost2=round(cost2*0.4,2)
                    eqcost2=round(cost2*0.16,2)
                    encost2=round(cost2*0.03,2)
                    ocost2=round(cost2*0.04,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Nitrogen Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Phosphorus Reduction (%)"])
                    df1["Nitrogen Reduction (lb/yr)"]=df1["Nitrogen Reduction (%)"]*(tn/100)
                    df2["Phosphorus Reduction (lb/yr)"]=df2["Phosphorus Reduction (%)"]*(tp/100)
                    df1["Phosphorus Reduction (%)"]=df2["Phosphorus Reduction (%)"]
                    df1["Phosphorus Reduction (lb/yr)"]=df2["Phosphorus Reduction (lb/yr)"]
                    tn_removal=(tn*removal_n)/100
                    tp_removal=(tp*removal_p)/100
                    tn_con=(tn*con_level)/100
                    tp_con=(tp*con_level)/100
                    with tab1:
                        fig=make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0.1,subplot_titles=("Nitrogen Reduction","Phosphorus Reduction"))
                        fig.add_trace(go.Scatter(x=df1["cost (USD)"],y=df1["Nitrogen Reduction (lb/yr)"],mode='markers',name='Nitrogen Reduction',marker=dict(color='rgba(8,88,158, 0.6)')),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df2["cost (USD)"],y=df2["Phosphorus Reduction (lb/yr)"],mode='markers',name='Phosphorus Reduction',marker=dict(color='rgba(241,105,19, 0.6)')),row = 2, col = 1)
                       
                        fig.add_hline(y=tn_removal,name= 'Nitrogen Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 1, col = 1)
                        fig.add_hline(y=tp_removal,name= 'Phosphorus Reduction',line=dict(color='firebrick', width=2,
                                                     dash='dash'),row = 2, col = 1)
                        fig.add_hrect(y0=tn_removal-tn_con, y1=tn_removal + tn_con, line_width=0, fillcolor="red", opacity=0.2,row = 1, col = 1)
                        fig.add_hrect(y0=tp_removal-tp_con, y1=tp_removal + tp_con, line_width=0, fillcolor="red", opacity=0.2,row = 2, col = 1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost1) + '<br>Total Nitrogen Concentration = '+str(atn)+ ' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(103,169,207)',
                                    y=tn,
                                    x=df1.iloc[0]["cost (USD)"],
                                    xanchor='left',row=1, col=1)
                        fig.add_annotation(
                                    text='Total Cost = $'+str(cost2) + '<br>Total Phosphorus  Concentration = '+str(atp)+' lb/year',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(253,208,162)',
                                    y=tp,
                                    x=df2.iloc[0]["cost (USD)"],
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
                        st.dataframe(df1)
                    
                    with tab3:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                        fig1 = go.Figure()
                        fig1.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig1.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
                           )
                            ))

                        fig1.update_layout(barmode='stack',
                                          
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
                                         )))
                        st.plotly_chart(fig1, use_container_width=True)
                
        elif SCM_type== 'Overview' :
            number = st.number_input('Available Area(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
           
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Comparisons among the single use implementations")
                    tab1,tab2,tab3= st.tabs(["graph","Table_N","Table_P"])
                    def bioretention(p1,p2):
                        objective_1 = 29631*(p1*p2)**0.026 
                        objective_2 = (98-(117.1*(2.718)**(-5.21*(p2))))
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def drypond(p1,p2):
                        objective_1 = 10525*(p1*p2)**0.29 
                        objective_2 = (98.26-(109.04*(2.718)**(-5.75*(p2))))
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def Constructedwetland_tn(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503 
                        objective_2 = (4389.78*(p2)**0.012)-4266.26
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def Constructedwetland_tp(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503
                        objective_2 = (260.665*(p2)**0.0285)-223.36
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def grassedswale(p1,p2):
                        objective_1 = 42504*(p1*p2)**0.0344 
                        objective_2 = (97.7936-(107.28*(2.718)**(-5.85*(p2))))
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def infiltrationtrench(p1,p2):
                        objective_1 = 27632*(p1*p2)**0.0431 
                        objective_2 = (63767.5*(p2)**0.000285)-63679.2
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def porouspavement(p1,p2):
                        objective_1 = 40540*(p1*p2)**0.0327 
                        objective_2 = (97.9016-(105.3*(2.718)**(-5.51*(p2))))
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def filterbed_tp(p1,p2):
                        objective_1 = 687.5*(p1*p2)**0.59 
                        objective_2 = (584.706*(p2)**0.012)-560.448
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def filterbed_tn(p1,p2):
                        objective_1 = 687.5*(p1*p2)**0.59 
                        objective_2 = (29.031*(p2)**0.17)+ 8.47
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def wetpond_tn(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503 
                        objective_2 = (4389.78*(p2)**0.012)-4266.26
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    def wetpond_tp(p1,p2):
                        objective_1 = 1875*(p1*p2)**0.503
                        objective_2 = (260.665*(p2)**0.0285)-223.36
                        return np.stack([p1,p2,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }

                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    #bioretention
                    mean_fitnesses_bio = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_bio = bioretention(pop,pop)
                        mean_fitnesses_bio.append(np.mean(fitnesses_bio,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_bio,config)
                    #drypond
                    mean_fitnesses_dp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_dp = drypond(pop,pop)
                        mean_fitnesses_dp.append(np.mean(fitnesses_dp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_dp,config)
                    #cw
                    mean_fitnesses_cwtn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_cwtn = Constructedwetland_tn(pop,pop)
                        mean_fitnesses_cwtn.append(np.mean(fitnesses_cwtn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_cwtn,config)
                     
                    mean_fitnesses_cwtp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_cwtp = Constructedwetland_tp(pop,pop)
                        mean_fitnesses_cwtp.append(np.mean(fitnesses_cwtp,axis=0))
                         
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_cwtp,config) 
                        
                    #gs
                    mean_fitnesses_gs = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_gs = grassedswale(pop,pop)
                        mean_fitnesses_gs.append(np.mean(fitnesses_gs,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_gs,config)
                    
                    #it
                    mean_fitnesses_it = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_it = infiltrationtrench(pop,pop)
                        mean_fitnesses_it.append(np.mean(fitnesses_it,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_it,config)
                        
                    #pp
                    mean_fitnesses_pp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_pp = porouspavement(pop,pop)
                        mean_fitnesses_pp.append(np.mean(fitnesses_pp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_pp,config)
                    #vf
                    mean_fitnesses_vftp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_vftp = filterbed_tp(pop,pop)
                        mean_fitnesses_vftp.append(np.mean(fitnesses_vftp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_vftp,config)
                     
                    mean_fitnesses_vftn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_vftn = filterbed_tn(pop,pop)
                        mean_fitnesses_vftn.append(np.mean(fitnesses_vftn,axis=0))
                         
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_vftn,config)
                        
                    #wp
                    mean_fitnesses_wptn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_wptn = wetpond_tn(pop,pop)
                        mean_fitnesses_wptn.append(np.mean(fitnesses_wptn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_wptn,config)
                     
                    mean_fitnesses_wptp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_wptp = wetpond_tp(pop,pop)
                        mean_fitnesses_cwtp.append(np.mean(fitnesses_wptp,axis=0))
                         
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_wptp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses1 = bioretention(p1,p2)
                    
                    pp_solutions_fitnesses2 = drypond(p1,p2)
                    
                    pp_solutions_fitnesses3 = Constructedwetland_tn(p1,p2)
                    
                    pp_solutions_fitnesses4 = Constructedwetland_tp(p1,p2)
                    
                    pp_solutions_fitnesses5 = grassedswale(p1,p2)
                    
                    pp_solutions_fitnesses6 = infiltrationtrench(p1,p2)
                    
                    pp_solutions_fitnesses7 = porouspavement(p1,p2)
                    
                    pp_solutions_fitnesses8 = filterbed_tp(p1,p2)
                    
                    pp_solutions_fitnesses9 = filterbed_tn(p1,p2)
                    
                    pp_solutions_fitnesses10 = wetpond_tn(p1,p2)
                    
                    pp_solutions_fitnesses11 = wetpond_tp(p1,p2)
                    
                    df1= pd.DataFrame(pp_solutions_fitnesses1,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df2= pd.DataFrame(pp_solutions_fitnesses2,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df3= pd.DataFrame(pp_solutions_fitnesses3,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df4= pd.DataFrame(pp_solutions_fitnesses4,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df5= pd.DataFrame(pp_solutions_fitnesses5,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df6= pd.DataFrame(pp_solutions_fitnesses6,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df7= pd.DataFrame(pp_solutions_fitnesses7,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df8= pd.DataFrame(pp_solutions_fitnesses8,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df9= pd.DataFrame(pp_solutions_fitnesses9,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df10= pd.DataFrame(pp_solutions_fitnesses10,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    df11= pd.DataFrame(pp_solutions_fitnesses11,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    
                    
                    df1["Nitrogen Reduction (lb/yr)"]=df1["Reduction (%)"]*(tn/100)
                    df1["Phosphorus Reduction (lb/yr)"]=df1["Reduction (%)"]*(tp/100)
                    df2["Nitrogen Reduction (lb/yr)"]=df2["Reduction (%)"]*(tn/100)
                    df2["Phosphorus Reduction (lb/yr)"]=df2["Reduction (%)"]*(tp/100)
                    df3["Nitrogen Reduction (lb/yr)"]=df3["Reduction (%)"]*(tn/100)
                    df4["Phosphorus Reduction (lb/yr)"]=df4["Reduction (%)"]*(tp/100)
                    df5["Nitrogen Reduction (lb/yr)"]=df5["Reduction (%)"]*(tn/100)
                    df5["Phosphorus Reduction (lb/yr)"]=df5["Reduction (%)"]*(tp/100)
                    df6["Nitrogen Reduction (lb/yr)"]=df6["Reduction (%)"]*(tn/100)
                    df6["Phosphorus Reduction (lb/yr)"]=df6["Reduction (%)"]*(tp/100)
                    df7["Nitrogen Reduction (lb/yr)"]=df7["Reduction (%)"]*(tn/100)
                    df7["Phosphorus Reduction (lb/yr)"]=df7["Reduction (%)"]*(tp/100)
                    df8["Nitrogen Reduction (lb/yr)"]=df8["Reduction (%)"]*(tn/100)
                    df9["Phosphorus Reduction (lb/yr)"]=df9["Reduction (%)"]*(tp/100)
                    df10["Nitrogen Reduction (lb/yr)"]=df10["Reduction (%)"]*(tn/100)
                    df11["Phosphorus Reduction (lb/yr)"]=df11["Reduction (%)"]*(tp/100)
                    
                    dc=pd.DataFrame()
                    dtn=pd.DataFrame()
                    dtp=pd.DataFrame()
                    dcc=pd.DataFrame()
                    dco=pd.DataFrame()
                    dcm=pd.DataFrame()
                    dcl=pd.DataFrame()
                    dceq=pd.DataFrame()
                    dcen=pd.DataFrame()
                    dcot=pd.DataFrame()
                    
                    dc['Bioretention']=df1["cost (USD)"]
                    dc['Dry Pond']=df2["cost (USD)"]
                    dc['Constructed Wetland']=df3["cost (USD)"]
                    dc['Grassed Swale']=df5["cost (USD)"]
                    dc['Infiltration Trench']=df6["cost (USD)"]
                    dc['Porous Pavement']=df7["cost (USD)"]
                    dc['Vegetative Filter Bed']=df8["cost (USD)"]
                    dc['Wet Pond']=df10["cost (USD)"]
                    dtn['Bioretention']=df1["Nitrogen Reduction (lb/yr)"]
                    dtn['Dry Pond']=df2["Nitrogen Reduction (lb/yr)"]
                    dtn['Constructed Wetland']=df3["Nitrogen Reduction (lb/yr)"]
                    dtn['Grassed Swale']=df5["Nitrogen Reduction (lb/yr)"]
                    dtn['Infiltration Trench']=df6["Nitrogen Reduction (lb/yr)"]
                    dtn['Porous Pavement']=df7["Nitrogen Reduction (lb/yr)"]
                    dtn['Vegetative Filter Bed']=df8["Nitrogen Reduction (lb/yr)"]
                    dtn['Wet Pond']=df10["Nitrogen Reduction (lb/yr)"]
                    dtp['Bioretention']=df1["Phosphorus Reduction (lb/yr)"]
                    dtp['Dry Pond']=df2["Phosphorus Reduction (lb/yr)"]
                    dtp['Constructed Wetland']=df4["Phosphorus Reduction (lb/yr)"]
                    dtp['Grassed Swale']=df5["Phosphorus Reduction (lb/yr)"]
                    dtp['Infiltration Trench']=df6["Phosphorus Reduction (lb/yr)"]
                    dtp['Porous Pavement']=df7["Phosphorus Reduction (lb/yr)"]
                    dtp['Vegetative Filter Bed']=df9["Phosphorus Reduction (lb/yr)"]
                    dtp['Wet Pond']=df11["Phosphorus Reduction (lb/yr)"]
                    
                    dcc['Bioretention']=(df1["cost (USD)"]-327.3)*0.46
                    dcc['Dry Pond']=(df2["cost (USD)"]-1614.89)*0.43
                    dcc['Constructed Wetland']=(df3["cost (USD)"]-966.5)*0.46
                    dcc['Grassed Swale']=(df5["cost (USD)"]-252.13)*0.33
                    dcc['Infiltration Trench']=(df6["cost (USD)"]-294.15)*0.54
                    dcc['Porous Pavement']=(df7["cost (USD)"]-294.15)*0.35
                    dcc['Vegetative Filter Bed']=(df8["cost (USD)"]-7380.08)*0.32
                    dcc['Wet Pond']=(df10["cost (USD)"]-4537.86)*0.36
                    
                    dco['Bioretention']=(df1["cost (USD)"]-327.3)*0.54
                    dco['Dry Pond']=(df2["cost (USD)"]-1614.89)*0.57
                    dco['Constructed Wetland']=(df3["cost (USD)"]-966.5)*0.54
                    dco['Grassed Swale']=(df5["cost (USD)"]-252.13)*0.67
                    dco['Infiltration Trench']=(df6["cost (USD)"]-294.15)*0.46
                    dco['Porous Pavement']=(df7["cost (USD)"]-294.15)*0.65
                    dco['Vegetative Filter Bed']=(df8["cost (USD)"]-7380.08)*0.68
                    dco['Wet Pond']=(df10["cost (USD)"]-4537.86)*0.64
                    
                    dcm['Bioretention']=df1["cost (USD)"]*0.37
                    dcm['Dry Pond']=df2["cost (USD)"]*0.31
                    dcm['Constructed Wetland']=df3["cost (USD)"]*0.12
                    dcm['Grassed Swale']=df5["cost (USD)"]*0.25
                    dcm['Infiltration Trench']=df6["cost (USD)"]*0.44
                    dcm['Porous Pavement']=df7["cost (USD)"]*0.29
                    dcm['Vegetative Filter Bed']=df8["cost (USD)"]*0.19
                    dcm['Wet Pond']=df10["cost (USD)"]*0.19
                    
                    dcl['Bioretention']=df1["cost (USD)"]*0.4
                    dcl['Dry Pond']=df2["cost (USD)"]*0.26
                    dcl['Constructed Wetland']=df3["cost (USD)"]*0.09
                    dcl['Grassed Swale']=df5["cost (USD)"]*0.27
                    dcl['Infiltration Trench']=df6["cost (USD)"]*0.34
                    dcl['Porous Pavement']=df7["cost (USD)"]*0.22
                    dcl['Vegetative Filter Bed']=df8["cost (USD)"]*0.62
                    dcl['Wet Pond']=df10["cost (USD)"]*0.33
                    
                    dceq['Bioretention']=df1["cost (USD)"]*0.16
                    dceq['Dry Pond']=df2["cost (USD)"]*0.29
                    dceq['Constructed Wetland']=df3["cost (USD)"]*0.16
                    dceq['Grassed Swale']=df5["cost (USD)"]*0.38
                    dceq['Infiltration Trench']=df6["cost (USD)"]*0.15
                    dceq['Porous Pavement']=df7["cost (USD)"]*0.38
                    dceq['Vegetative Filter Bed']=df8["cost (USD)"]*0.13
                    dceq['Wet Pond']=df10["cost (USD)"]*0.36
                    
                    dcen['Bioretention']=df1["cost (USD)"]*0.03
                    dcen['Dry Pond']=df2["cost (USD)"]*0.09
                    dcen['Constructed Wetland']=df3["cost (USD)"]*0.03
                    dcen['Grassed Swale']=df5["cost (USD)"]*0.06
                    dcen['Infiltration Trench']=df6["cost (USD)"]*0.03
                    dcen['Porous Pavement']=df7["cost (USD)"]*0.08
                    dcen['Vegetative Filter Bed']=df8["cost (USD)"]*0.04
                    dcen['Wet Pond']=df10["cost (USD)"]*0.09
                    
                    dcot['Bioretention']=df1["cost (USD)"]*0.04
                    dcot['Dry Pond']=df2["cost (USD)"]*0.04
                    dcot['Constructed Wetland']=df3["cost (USD)"]*0.01
                    dcot['Grassed Swale']=df5["cost (USD)"]*0.03
                    dcot['Infiltration Trench']=df6["cost (USD)"]*0.05
                    dcot['Porous Pavement']=df7["cost (USD)"]*0.03
                    dcot['Vegetative Filter Bed']=df8["cost (USD)"]*0.03
                    dcot['Wet Pond']=df10["cost (USD)"]*0.03
                    
                    x1= df1["Reduction (%)"]
                    y1= df1["cost (USD)"]
                    x2= df2["Reduction (%)"]
                    y2= df2["cost (USD)"]
                    x3= df3["Reduction (%)"]
                    y3= df3["cost (USD)"]
                    x4= df4["Reduction (%)"]
                    y4= df4["cost (USD)"]
                    x5= df5["Reduction (%)"]
                    y5= df5["cost (USD)"]
                    x6= df6["Reduction (%)"]
                    y6= df6["cost (USD)"]
                    x7= df7["Reduction (%)"]
                    y7= df7["cost (USD)"]
                    x8= df8["Reduction (%)"]
                    y8= df8["cost (USD)"]
                    x9= df9["Reduction (%)"]
                    y9= df9["cost (USD)"]
                    x10= df10["Reduction (%)"]
                    y10= df10["cost (USD)"]
                    x11= df11["Reduction (%)"]
                    y11= df11["cost (USD)"]
                    def Gauss(x, A,B,C):
                        y = A*x +B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1, y1)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = round(fit_A1*removal+fit_B1*removal**2+fit_C1,2)
                    
                    parameters2, covariance2 = curve_fit(Gauss, x2, y2)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = round(fit_A2*removal+fit_B2*removal**2+fit_C2,2)
                    
                    parameters3, covariance3 = curve_fit(Gauss, x3, y3)
                    fit_A3 = parameters3[0]
                    fit_B3 = parameters3[1]
                    fit_C3= parameters3[2]
                    cost3 = round(fit_A3*removal+fit_B3*removal**2+fit_C3,2)
                    
                    parameters4, covariance4 = curve_fit(Gauss, x4, y4)
                    fit_A4 = parameters4[0]
                    fit_B4 = parameters4[1]
                    fit_C4= parameters4[2]
                    cost4 = round(fit_A4*removal+fit_B4*removal**2+fit_C4,2)
                    
                    parameters5, covariance5 = curve_fit(Gauss, x5, y5)
                    fit_A5 = parameters5[0]
                    fit_B5 = parameters5[1]
                    fit_C5= parameters5[2]
                    cost5 = round(fit_A5*removal+fit_B5*removal**2+fit_C5,2)
                    
                    parameters6, covariance6 = curve_fit(Gauss, x6, y6)
                    fit_A6 = parameters6[0]
                    fit_B6 = parameters6[1]
                    fit_C6= parameters6[2]
                    cost6 = round(fit_A6*removal+fit_B6*removal**2+fit_C6,2)
                    
                    parameters7, covariance7 = curve_fit(Gauss, x7, y7)
                    fit_A7 = parameters7[0]
                    fit_B7 = parameters7[1]
                    fit_C7= parameters7[2]
                    cost7 = round(fit_A7*removal+fit_B7*removal**2+fit_C7,2)
                    
                    parameters8, covariance8 = curve_fit(Gauss, x8, y8)
                    fit_A8 = parameters8[0]
                    fit_B8 = parameters8[1]
                    fit_C8= parameters8[2]
                    cost8 = round(fit_A8*removal+fit_B8*removal**2+fit_C8,2)
                    
                    parameters9, covariance9 = curve_fit(Gauss, x9, y9)
                    fit_A9 = parameters9[0]
                    fit_B9 = parameters9[1]
                    fit_C9= parameters9[2]
                    cost9 = round(fit_A9*removal+fit_B9*removal**2+fit_C9,2)
                    
                    parameters10, covariance10 = curve_fit(Gauss, x10, y10)
                    fit_A10 = parameters10[0]
                    fit_B10 = parameters10[1]
                    fit_C10= parameters10[2]
                    cost10= round(fit_A10*removal+fit_B10*removal**2+fit_C10,2)
                    
                    parameters11, covariance11 = curve_fit(Gauss, x11, y11)
                    fit_A11 = parameters11[0]
                    fit_B11 = parameters11[1]
                    fit_C11= parameters11[2]
                    cost11 = round(fit_A11*removal+fit_B11*removal**2+fit_C11,2)
                    
                    ecost1=327.3
                    ccost1=round((cost1-ecost1)*0.46,2)
                    opcost1=round((cost1-ecost1)*0.54,2)
                    mcost1=round(cost1*0.37,2)
                    lcost1=round(cost1*0.4,2)
                    eqcost1=round(cost1*0.16,2)
                    encost1=round(cost1*0.03,2)
                    ocost1=round(cost1*0.04,2)
                    
                    ecost2=1614.89
                    ccost2=round((cost2-ecost1)*0.43,2)
                    opcost2=round((cost2-ecost1)*0.57,2)
                    mcost2=round(cost2*0.31,2)
                    lcost2=round(cost2*0.26,2)
                    eqcost2=round(cost2*0.29,2)
                    encost2=round(cost2*0.09,2)
                    ocost2=round(cost2*0.04,2)
                    
                    ecost3=966.5
                    ccost3=round((cost3-ecost3)*0.46,2)
                    opcost3=round((cost3-ecost3)*0.54,2)
                    mcost3=round(cost3*0.32,2)
                    lcost3=round(cost3*0.25,2)
                    eqcost3=round(cost3*0.30,2)
                    encost3=round(cost3*0.09,2)
                    ocost3=round(cost3*0.04,2)
                    
                    ecost4=966.5
                    ccost4=round((cost4-ecost4)*0.46,2)
                    opcos4=round((cost4-ecost4)*0.54,2)
                    mcost4=round(cost4*0.12,2)
                    lcost4=round(cost4*0.09,2)
                    eqcost4=round(cost4*0.16,2)
                    encost4=round(cost4*0.03,2)
                    ocost4=round(cost4*0.01,2)
                    
                    ecost5=252.13
                    ccost5=round((cost5-ecost5)*0.33,2)
                    opcost5=round((cost5-ecost5)*0.67,2)
                    mcost5=round(cost5*0.25,2)
                    lcost5=round(cost5*0.27,2)
                    eqcost5=round(cost5*0.38,2)
                    encost5=round(cost5*0.06,2)
                    ocost5=round(cost5*0.03,2)
                    
                    ecost6=294.15
                    ccost6=round((cost6-ecost6)*0.54,2)
                    opcost6=round((cost6-ecost6)*0.46,2)
                    mcost6=round(cost6*0.44,2)
                    lcost6=round(cost6*0.34,2)
                    eqcost6=round(cost6*0.15,2)
                    encost6=round(cost6*0.03,2)
                    ocost6=round(cost6*0.05,2)
                    
                    ecost7= 294.15
                    ccost7=round((cost7-ecost7)*0.35,2)
                    opcost7=round((cost7-ecost7)*0.65,2)
                    mcost7=round(cost7*0.29,2)
                    lcost7=round(cost7*0.22,2)
                    eqcost7=round(cost7*0.38,2)
                    encost7=round(cost7*0.08,2)
                    ocost7=round(cost7*0.03,2)
                    
                    ecost8=7380.08
                    ccost8=round((cost8-ecost8)*0.32,2)
                    opcost8=round((cost8-ecost8)*0.68,2)
                    mcost8=round(cost8*0.19,2)
                    lcost8=round(cost8*0.62,2)
                    eqcost8=round(cost8*0.13,2)
                    encost8=round(cost8*0.04,2)
                    ocost8=round(cost8*0.03,2)
                    
                    ecost9= 7380.08
                    ccost9=round((cost9-ecost9)*0.32,2)
                    opcost9=round((cost9-ecost9)*0.68,2)
                    mcost9=round(cost9*0.19,2)
                    lcost9=round(cost9*0.62,2)
                    eqcost9=round(cost9*0.13,2)
                    encost9=round(cost9*0.04,2)
                    ocost9=round(cost9*0.03,2)
                    
                    ecost10=4537.86
                    ccost10=round((cost10-ecost10)*0.36,2)
                    opcost10=round((cost10-ecost10)*0.64,2)
                    mcost10=round(cost10*0.19,2)
                    lcost10=round(cost10*0.33,2)
                    eqcost10=round(cost10*0.36,2)
                    encost10=round(cost10*0.09,2)
                    ocost10=round(cost10*0.03,2)
                    
                    ecost11=4537.86
                    ccost11=round((cost11-ecost11)*0.46,2)
                    opcost11=round((cost11-ecost11)*0.54,2)
                    mcost11=round(cost11*0.19,2)
                    lcost11=round(cost11*0.33,2)
                    eqcost11=round(cost11*0.36,2)
                    encost11=round(cost11*0.09,2)
                    ocost11=round(cost11*0.03,2)
                    
                    
                    
                    with tab1:
                        fig=make_subplots(rows=5, cols=1,vertical_spacing=0.05,subplot_titles=("Pareto Optimal Curve for Nitrogen Reduction","Pareto Optimal Curve for Phosphorus Reduction","Total Cost","Nitrogen Reduction","Phosphorus Reduction"))
                        fig.add_trace(go.Scatter(x=df1["cost (USD)"],y=df1["Nitrogen Reduction (lb/yr)"],mode='markers',name='Bioretention',legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df2["cost (USD)"],y=df2["Nitrogen Reduction (lb/yr)"],mode='markers',name='Dry Pond',legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df3["cost (USD)"],y=df3["Nitrogen Reduction (lb/yr)"],mode='markers',name='Constructed Wetland',legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df5["cost (USD)"],y=df5["Nitrogen Reduction (lb/yr)"],mode='markers',name='Grassed Swale',legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df6["cost (USD)"],y=df6["Nitrogen Reduction (lb/yr)"],mode='markers',name='Infiltration Trench',legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df7["cost (USD)"],y=df7["Nitrogen Reduction (lb/yr)"],mode='markers',name='Porous Pavement',legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df8["cost (USD)"],y=df8["Nitrogen Reduction (lb/yr)"],mode='markers',name='Vegetative Filter Bed',legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df10["cost (USD)"],y=df10["Nitrogen Reduction (lb/yr)"],mode='markers',name='Wet Pond',legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)'),showlegend=False),row = 1, col = 1)
                        fig.add_trace(go.Scatter(x=df1["cost (USD)"],y=df1["Phosphorus Reduction (lb/yr)"],mode='markers',name='Bioretention',legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Scatter(x=df2["cost (USD)"],y=df2["Phosphorus Reduction (lb/yr)"],mode='markers',name='Dry Pond',legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Scatter(x=df4["cost (USD)"],y=df4["Phosphorus Reduction (lb/yr)"],mode='markers',name='Constructed Wetland',legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Scatter(x=df5["cost (USD)"],y=df5["Phosphorus Reduction (lb/yr)"],mode='markers',name='Grassed Swale',legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Scatter(x=df6["cost (USD)"],y=df6["Phosphorus Reduction (lb/yr)"],mode='markers',name='Infiltration Trench',legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Scatter(x=df7["cost (USD)"],y=df7["Phosphorus Reduction (lb/yr)"],mode='markers',name='Porous Pavement',legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Scatter(x=df9["cost (USD)"],y=df9["Phosphorus Reduction (lb/yr)"],mode='markers',name='Vegetative Filter Bed',legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Scatter(x=df11["cost (USD)"],y=df11["Phosphorus Reduction (lb/yr)"],mode='markers',name='Wet Pond',legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)'),showlegend=False),row = 2, col = 1)
                        fig.add_trace(go.Box(y=dc['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dc['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dc['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dc['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dc['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dc['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dc['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dc['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=True),row = 3, col = 1)
                        fig.add_trace(go.Box(y=dtn['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtn['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtn['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtn['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtn['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtn['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtn['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtn['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig.add_trace(go.Box(y=dtp['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.add_trace(go.Box(y=dtp['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.add_trace(go.Box(y=dtp['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.add_trace(go.Box(y=dtp['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.add_trace(go.Box(y=dtp['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.add_trace(go.Box(y=dtp['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.add_trace(go.Box(y=dtp['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.add_trace(go.Box(y=dtp['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig.update_xaxes(title_text="Total Cost (USD)",showline=True,
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
                        fig.update_xaxes(title_text="Total Cost (USD)",showline=True,
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
                        fig.update_xaxes(showline=True,
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
                                         row=3, col=1)
                        fig.update_xaxes(showline=True,
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
                                         row=4, col=1)
                        fig.update_xaxes(showline=True,
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
                                         row=5, col=1)
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
                        fig.update_yaxes(title_text="Total Cost (USD)",showline=True,
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
                                         row=3, col=1)
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
                                         row=4, col=1)
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
                                         row=5, col=1)
                                         
                        
                        fig.update_layout(height=3500, width=1500,
                                          font=dict(
                                              family="Arial",
                                              size=30,
                                              color="Black"),
                                          
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
                        fig1=make_subplots(rows=7, cols=1,vertical_spacing=0.04,subplot_titles=("Construction Cost","Operations and Maintenance Cost","Materials Cost","Labor Cost","Equipment Cost","Energy Cost","Others Cost"))
                        fig1.add_trace(go.Box(y=dcc['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        fig1.add_trace(go.Box(y=dcc['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        fig1.add_trace(go.Box(y=dcc['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        fig1.add_trace(go.Box(y=dcc['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        fig1.add_trace(go.Box(y=dcc['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        fig1.add_trace(go.Box(y=dcc['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        fig1.add_trace(go.Box(y=dcc['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        fig1.add_trace(go.Box(y=dcc['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=True),row = 1, col = 1)
                        
                        fig1.add_trace(go.Box(y=dco['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        fig1.add_trace(go.Box(y=dco['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        fig1.add_trace(go.Box(y=dco['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        fig1.add_trace(go.Box(y=dco['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        fig1.add_trace(go.Box(y=dco['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        fig1.add_trace(go.Box(y=dco['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        fig1.add_trace(go.Box(y=dco['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        fig1.add_trace(go.Box(y=dco['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 2, col = 1)
                        
                        fig1.add_trace(go.Box(y=dcm['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        fig1.add_trace(go.Box(y=dcm['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        fig1.add_trace(go.Box(y=dcm['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        fig1.add_trace(go.Box(y=dcm['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        fig1.add_trace(go.Box(y=dcm['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        fig1.add_trace(go.Box(y=dcm['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        fig1.add_trace(go.Box(y=dcm['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        fig1.add_trace(go.Box(y=dcm['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 3, col = 1)
                        
                        fig1.add_trace(go.Box(y=dcl['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig1.add_trace(go.Box(y=dcl['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig1.add_trace(go.Box(y=dcl['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig1.add_trace(go.Box(y=dcl['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig1.add_trace(go.Box(y=dcl['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig1.add_trace(go.Box(y=dcl['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig1.add_trace(go.Box(y=dcl['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        fig1.add_trace(go.Box(y=dcl['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 4, col = 1)
                        
                        fig1.add_trace(go.Box(y=dceq['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig1.add_trace(go.Box(y=dceq['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig1.add_trace(go.Box(y=dceq['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig1.add_trace(go.Box(y=dceq['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig1.add_trace(go.Box(y=dceq['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig1.add_trace(go.Box(y=dceq['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig1.add_trace(go.Box(y=dceq['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        fig1.add_trace(go.Box(y=dceq['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 5, col = 1)
                        
                        fig1.add_trace(go.Box(y=dcen['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        fig1.add_trace(go.Box(y=dcen['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        fig1.add_trace(go.Box(y=dcen['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        fig1.add_trace(go.Box(y=dcen['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        fig1.add_trace(go.Box(y=dcen['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        fig1.add_trace(go.Box(y=dcen['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        fig1.add_trace(go.Box(y=dcen['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        fig1.add_trace(go.Box(y=dcen['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 6, col = 1)
                        
                        fig1.add_trace(go.Box(y=dcot['Bioretention'],name='Bioretention',notched=True,legendgroup='group1',marker=dict(color='rgba(158,1,66, 0.6)',line=dict(color='rgba(158,1,66, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.add_trace(go.Box(y=dcot['Dry Pond'],name='Dry Pond',notched=True,legendgroup='group2',marker=dict(color='rgba(244,109,67, 0.6)',line=dict(color='rgba(244,109,67, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.add_trace(go.Box(y=dcot['Constructed Wetland'],name='Constructed Wetland',notched=True,legendgroup='group3',marker=dict(color='rgba(224,130,20, 0.6)',line=dict(color='rgba(224,130,20, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.add_trace(go.Box(y=dcot['Grassed Swale'],name='Grassed Swale',notched=True,legendgroup='group4',marker=dict(color='rgba(166,217,106, 0.6)',line=dict(color='rgba(166,217,106, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.add_trace(go.Box(y=dcot['Infiltration Trench'],name='Infiltration Trench',notched=True,legendgroup='group5',marker=dict(color='rgba(0,104,55, 0.6)',line=dict(color='rgba(0,104,55, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.add_trace(go.Box(y=dcot['Porous Pavement'],name='Porous Pavement',notched=True,legendgroup='group6',marker=dict(color='rgba(102,194,165, 0.6)',line=dict(color='rgba(102,194,165, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.add_trace(go.Box(y=dcot['Vegetative Filter Bed'],name='Vegetative Filter Bed',notched=True,legendgroup='group7',marker=dict(color='rgba(50,136,189, 0.6)',line=dict(color='rgba(50,136,189, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.add_trace(go.Box(y=dcot['Wet Pond'],name='Wet Pond',notched=True,legendgroup='group8',marker=dict(color='rgba(84,39,136, 0.6)',line=dict(color='rgba(84,39,136, 1.0)', width=3)),showlegend=False),row = 7, col = 1)
                        fig1.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=1, col=1)
                        fig1.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=2, col=1)
                        fig1.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=3, col=1)
                        fig1.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=4, col=1)
                        fig1.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=5, col=1)
                        fig1.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=6, col=1)
                        fig1.update_xaxes(showline=True,
                                         showgrid=False,
                                         linecolor='black',
                                         linewidth=2,
                                         ticks='outside',
                                         tickfont=dict(
                                             family='Arial',
                                             size=20,
                                             color='black'),
                                         row=7, col=1)
                        fig1.update_yaxes(title_text="Construction Cost (USD)",showline=True,
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
                        fig1.update_yaxes(title_text="Operation and Maintenance Cost (USD)",showline=True,
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
                        fig1.update_yaxes(title_text="Material Cost (USD)",showline=True,
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
                                         row=3, col=1)
                        fig1.update_yaxes(title_text="Labor Cost (USD)",showline=True,
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
                                         row=4, col=1)
                        fig1.update_yaxes(title_text="Equipment Cost (USD)",showline=True,
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
                                         row=5, col=1)
                        fig1.update_yaxes(title_text="Energy Cost (USD)",showline=True,
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
                                         row=6, col=1)
                        fig1.update_yaxes(title_text="Others Cost (USD)",showline=True,
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
                                         row=7, col=1)
                                         
                        
                        fig1.update_layout(height=5600, width=1800,
                                          font=dict(
                                              family="Arial",
                                              size=30,
                                              color="Black"),
                                          
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
                        fig1.update_layout(legend= {'itemsizing': 'constant'})
                        st.plotly_chart(fig1, use_container_width=False)
                     
                    with tab3:
                        if removal <= 20.00:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[0,0,0,0,0,0,0,0],
                               name='Planning Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(65,182,196, 0.6)',
                                line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[ccost1,ccost2,ccost3,ccost5,ccost6,ccost7,ccost9,ccost10],
                               name='Construction Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(34,94,168, 0.6)',
                                line=dict(color='rgba(34,94,168, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[opcost1,opcost2,opcost3,opcost5,opcost6,opcost7,opcost9,opcost10],
                               name="Operations and Maintenance Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(116,196,118, 0.6)',
                                line=dict(color='rgba(116,196,118, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[ecost1,ecost2,ecost3,ecost5,ecost6,ecost7,ecost9,ecost10],
                               name="End of Life Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(0,109,44, 0.6)',
                                line=dict(color='rgba(0,109,44, 1.0)', width=3)
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
                            fig1 = go.Figure()
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[mcost1,mcost2,mcost3,mcost5,mcost6,mcost7,mcost9,mcost10],
                               name='Material Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(227, 26, 28, 0.6)',
                                line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[lcost1,lcost2,lcost3,lcost5,lcost6,lcost7,lcost9,lcost10],
                               name='Labor Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(165,15,21, 0.6)',
                                line=dict(color='rgba(165,15,21, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[eqcost1,eqcost2,eqcost3,eqcost5,eqcost6,eqcost7,eqcost9,eqcost10],
                               name="Equipment Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(254,153,41, 0.6)',
                                line=dict(color='rgba(254,153,41, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[encost1,encost2,encost3,encost5,encost6,encost7,encost9,encost10],
                               name="Energy Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(247,104,161, 0.6)',
                                line=dict(color='rgba(247,104,161, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[ocost1,ocost2,ocost3,ocost5,ocost6,ocost7,ocost9,ocost10],
                               name="Others Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(129,15,124, 0.6)',
                                line=dict(color='rgba(129,15,124, 1.0)', width=3)
                                      )
                                      ))
                            fig1.update_layout(barmode='stack',
                                              
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
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        if removal <= 35.00 and removal >20.00:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[0,0,0,0,0,0,0,0],
                               name='Planning Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(65,182,196, 0.6)',
                                line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[ccost1,ccost2,ccost3,ccost5,ccost6,ccost7,ccost8,ccost11],
                               name='Construction Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(34,94,168, 0.6)',
                                line=dict(color='rgba(34,94,168, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[opcost1,opcost2,opcost3,opcost5,opcost6,opcost7,opcost8,opcost11],
                               name="Operations and Maintenance Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(116,196,118, 0.6)',
                                line=dict(color='rgba(116,196,118, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[ecost1,ecost2,ecost3,ecost5,ecost6,ecost7,ecost8,ecost11],
                               name="End of Life Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(0,109,44, 0.6)',
                                line=dict(color='rgba(0,109,44, 1.0)', width=3)
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
                            fig1 = go.Figure()
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[mcost1,mcost2,mcost3,mcost5,mcost6,mcost7,mcost8,mcost11],
                               name='Material Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(227, 26, 28, 0.6)',
                                line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[lcost1,lcost2,lcost3,lcost5,lcost6,lcost7,lcost8,lcost11],
                               name='Labor Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(165,15,21, 0.6)',
                                line=dict(color='rgba(165,15,21, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[eqcost1,eqcost2,eqcost3,eqcost5,eqcost6,eqcost7,eqcost8,eqcost11],
                               name="Equipment Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(254,153,41, 0.6)',
                                line=dict(color='rgba(254,153,41, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[encost1,encost2,encost3,encost5,encost6,encost7,encost8,encost11],
                               name="Energy Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(247,104,161, 0.6)',
                                line=dict(color='rgba(247,104,161, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Vegetative Filter Bed','Wet Pond'],
                               y=[ocost1,ocost2,ocost3,ocost5,ocost6,ocost7,ocost8,ocost11],
                               name="Others Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(129,15,124, 0.6)',
                                line=dict(color='rgba(129,15,124, 1.0)', width=3)
                                      )
                                      ))
                            fig1.update_layout(barmode='stack',
                                              
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
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        if removal <= 67.00 and removal >35.00:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[0,0,0,0,0,0,0,0],
                               name='Planning Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(65,182,196, 0.6)',
                                line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[ccost1,ccost2,ccost3,ccost5,ccost6,ccost7,ccost10],
                               name='Construction Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(34,94,168, 0.6)',
                                line=dict(color='rgba(34,94,168, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[opcost1,opcost2,opcost3,opcost5,opcost6,opcost7,opcost10],
                               name="Operations and Maintenance Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(116,196,118, 0.6)',
                                line=dict(color='rgba(116,196,118, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[ecost1,ecost2,ecost3,ecost5,ecost6,ecost7,ecost10],
                               name="End of Life Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(0,109,44, 0.6)',
                                line=dict(color='rgba(0,109,44, 1.0)', width=3)
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
                            fig1 = go.Figure()
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[mcost1,mcost2,mcost3,mcost5,mcost6,mcost7,mcost10],
                               name='Material Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(227, 26, 28, 0.6)',
                                line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[lcost1,lcost2,lcost3,lcost5,lcost6,lcost7,lcost10],
                               name='Labor Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(165,15,21, 0.6)',
                                line=dict(color='rgba(165,15,21, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[eqcost1,eqcost2,eqcost3,eqcost5,eqcost6,eqcost7,eqcost10],
                               name="Equipment Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(254,153,41, 0.6)',
                                line=dict(color='rgba(254,153,41, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[encost1,encost2,encost3,encost5,encost6,encost7,encost10],
                               name="Energy Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(247,104,161, 0.6)',
                                line=dict(color='rgba(247,104,161, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement','Wet Pond'],
                               y=[ocost1,ocost2,ocost3,ocost5,ocost6,ocost7,ocost10],
                               name="Others Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(129,15,124, 0.6)',
                                line=dict(color='rgba(129,15,124, 1.0)', width=3)
                                      )
                                      ))
                            fig1.update_layout(barmode='stack',
                                              
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
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        if removal <= 75.00 and removal >67.00:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[0,0,0,0,0,0,0,0],
                               name='Planning Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(65,182,196, 0.6)',
                                line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[ccost1,ccost2,ccost3,ccost5,ccost6,ccost7],
                               name='Construction Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(34,94,168, 0.6)',
                                line=dict(color='rgba(34,94,168, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[opcost1,opcost2,opcost3,opcost5,opcost6,opcost7],
                               name="Operations and Maintenance Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(116,196,118, 0.6)',
                                line=dict(color='rgba(116,196,118, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[ecost1,ecost2,ecost3,ecost5,ecost6,ecost7],
                               name="End of Life Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(0,109,44, 0.6)',
                                line=dict(color='rgba(0,109,44, 1.0)', width=3)
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
                            fig1 = go.Figure()
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[mcost1,mcost2,mcost3,mcost5,mcost6,mcost7],
                               name='Material Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(227, 26, 28, 0.6)',
                                line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[lcost1,lcost2,lcost3,lcost5,lcost6,lcost7],
                               name='Labor Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(165,15,21, 0.6)',
                                line=dict(color='rgba(165,15,21, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[eqcost1,eqcost2,eqcost3,eqcost5,eqcost6,eqcost7],
                               name="Equipment Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(254,153,41, 0.6)',
                                line=dict(color='rgba(254,153,41, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[encost1,encost2,encost3,encost5,encost6,encost7],
                               name="Energy Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(247,104,161, 0.6)',
                                line=dict(color='rgba(247,104,161, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Infiltration Trench','Porous Pavement'],
                               y=[ocost1,ocost2,ocost3,ocost5,ocost6,ocost7],
                               name="Others Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(129,15,124, 0.6)',
                                line=dict(color='rgba(129,15,124, 1.0)', width=3)
                                      )
                                      ))
                            fig1.update_layout(barmode='stack',
                                              
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
                            st.plotly_chart(fig1, use_container_width=True)
                            
                        if removal >75.00:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[0,0,0,0,0],
                               name='Planning Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(65,182,196, 0.6)',
                                line=dict(color='rgba(65,182,196, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[ccost1,ccost2,ccost3,ccost5,ccost7],
                               name='Construction Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(34,94,168, 0.6)',
                                line=dict(color='rgba(34,94,168, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[opcost1,opcost2,opcost3,opcost5,opcost7],
                               name="Operations and Maintenance Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(116,196,118, 0.6)',
                                line=dict(color='rgba(116,196,118, 1.0)', width=3)
                                      )
                                      ))
                            fig.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[ecost1,ecost2,ecost3,ecost5,ecost7],
                               name="End of Life Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(0,109,44, 0.6)',
                                line=dict(color='rgba(0,109,44, 1.0)', width=3)
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
                            fig1 = go.Figure()
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[mcost1,mcost2,mcost3,mcost5,mcost7],
                               name='Material Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(227, 26, 28, 0.6)',
                                line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[lcost1,lcost2,lcost3,lcost5,lcost7],
                               name='Labor Cost',
                               orientation='v',
                               marker=dict(
                                color='rgba(165,15,21, 0.6)',
                                line=dict(color='rgba(165,15,21, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[eqcost1,eqcost2,eqcost3,eqcost5,eqcost7],
                               name="Equipment Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(254,153,41, 0.6)',
                                line=dict(color='rgba(254,153,41, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[encost1,encost2,encost3,encost5,encost7],
                               name="Energy Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(247,104,161, 0.6)',
                                line=dict(color='rgba(247,104,161, 1.0)', width=3)
                                      )
                                      ))
                            fig1.add_trace(go.Bar(
                               x=['Bioretention','Dry Pond','Constructed Wetland','Grassed Swale','Porous Pavement'],
                               y=[ocost1,ocost2,ocost3,ocost5,ocost7],
                               name="Others Cost",
                               orientation='v',
                               marker=dict(
                                color='rgba(129,15,124, 0.6)',
                                line=dict(color='rgba(129,15,124, 1.0)', width=3)
                                      )
                                      ))
                            fig1.update_layout(barmode='stack',
                                              
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
                            st.plotly_chart(fig1, use_container_width=True)
else:
    col1, col2 = st.columns([1, 3])
    with col1:
        options=['Bioretention & Wet Pond','Bioretention & Dry Pond','Bioretention & Porous Pavement','Bioretention & Grassed Swale','Bioretention & Vegetative Filterbed','Bioretention & Infiltration Trench','Bioretention & Constructed Wetland', 'Porous Pavement & Wet Pond','Porus Pavement & Grassed Swale','Porous Pavement & Dry Pond','Porus Pavement & Vegetative Filterbed','Porous Pavement & Infiltration Trench','Porus Pavement & Constructed Wetpond','Infiltration Trench & Grassed Swale','Infiltration Trench & Dry Pond','Infiltration Trench & Vegetative Filterbed','Infiltration Trench & Wet Pond','Infiltration Trench & Constructed Wetpond','Grassed Swale & Vegetative Filterbed','Grassed Swale & Wet Pond','Grassed Swale & Constructed Wetpond','Dry Pond & Grassed Swale','Wet Pond & Vegetative Filterbed','Vegetative Filterbed & Constructed Wetpond','Dry Pond & Vegetative Filterbed','Bioretention, Porous Pavement & Wet Pond','Bioretention, Grassed Swale & Wet Pond','Bioretention, Vegetative Filterbed & Wet Pond','Bioretention, Porous Pavement, Vegetative Filterbed & Wet Pond','Bioretention, Porous Pavement, Grassed Swale & Wet Pond']
        SCM_type= st.selectbox('Type of SCM:',options)
        if SCM_type=='Bioretention & Dry Pond':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Dry pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention & Dry Pond")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 10525*(p3*p4)**0.29   
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(98.26-(109.04*(2.718)**(-5.75*(p4)))))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.42,2)
                    opcost=round((cost-ecost)*0.54,2)
                    mcost=round(cost*0.33,2)
                    lcost=round(cost*0.3,2)
                    eqcost=round(cost*0.26,2)
                    encost=round(cost*0.08,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Dry Pond (sft)","depth of Dry Pond (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention & Porous Pavement':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Porous Pavement(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention & Porous Pavement")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 40540*(p3*p4)**0.0327   
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.9016-(105.3*(2.718)**(-5.51*(p4)))))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        
        if SCM_type=='Bioretention & Grassed Swale':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Grassed Swale(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 42504*(p3*p4)**0.0344    
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.7936-(107.28*(2.718)**(-5.85*(p4)))))/100
                       
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        
        if SCM_type=='Bioretention & Infiltration Trench':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Infiltration Trench(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 27632*(p3*p4)**0.0431    
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*((63767.5*(p4)**0.000285)-63679.2))/100
                    
                    
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention & Constructed Wetland':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Constructed Wetland (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 1875*(p3*p4)**0.503 
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p4))))*((4389.78*(p2)**0.012)-4286.26))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 1875*(p3*p4)**0.503 
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p4))))*((260.665*(p2)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention & Vegetative Filterbed':
            number1 = st.number_input('Available Area for Bioretention (sft)')
            number2 = st.number_input('Available  Area for Vegetative Filterbed (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 687.5*(p3*p4)**0.59 
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*((584.706*(p4)**0.012)-560.448))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 687.5*(p3*p4)**0.59 
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*((29.031*(p4)**0.17)+ 8.47))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention & Wet Pond':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 1875*(p3*p4)**0.503
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*((4389.78*(p4)**0.012)-4286.26))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 29631*(p1*p2)**0.026 + 1875*(p3*p4)**0.503  
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        
        if SCM_type=='Porous Pavement & Dry Pond':
            number1 = st.number_input('Available Area for Porous Pavement (sft)')
            number2 = st.number_input('Available  Area for Dry pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 10525*(p3*p4)**0.29 
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*(98.26-(109.04*(2.718)**(-5.75*(p4)))))/100
                        
                       
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Porous Pavement & Grassed Swale':
            number1 = st.number_input('Available Area for Porous Pavement (sft)')
            number2 = st.number_input('Available  Area for Grassed Swale (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 42504*(p3*p4)**0.0344    
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*(97.7936-(107.28*(2.718)**(-5.85*(p4)))))/100
                        
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Porous Pavement & Infiltration Trench':
            number1 = st.number_input('Available Area for Porous Pavement (sft)')
            number2 = st.number_input('Available  Area for Infiltration Trench(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 27632*(p3*p4)**0.0431    
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*((63767.5*(p4)**0.000285)-63679.2))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Porous Pavement & Constructed Wetland':
            number1 = st.number_input('Available Area for Porous Pavement (sft)')
            number2 = st.number_input('Available  Area for Constructed Wetland (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 1875*(p3*p4)**0.503 
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*((4389.78*(p4)**0.012)-4286.26))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 1875*(p3*p4)**0.503
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    
                        
                 
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
            
        if SCM_type=='Porous Pavement & Vegetative Filterbed':
            number1 = st.number_input('Available Area for Porous Pavement (sft)')
            number2 = st.number_input('Available  Area for Vegetative Filterbed (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 687.5*(p3*p4)**0.59 
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*((584.706*(p4)**0.012)-560.448))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 687.5*(p3*p4)**0.59 
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*((29.031*(p4)**0.17)+ 8.47))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        
        if SCM_type=='Porous Pavement & Wet Pond':
            number1 = st.number_input('Available Area for Porous Pavement (sft)')
            number2 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 1875*(p3*p4)**0.503  
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*((4389.78*(p4)**0.012)-4286.26))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 40540*(p1*p2)**0.0327  + 1875*(p3*p4)**0.503  
                        objective_2 = ((97.9016-(105.3*(2.718)**(-5.51*(p2))))*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Infiltration Trench & Dry Pond':
            number1 = st.number_input('Available Area for Infiltration Trench(sft)')
            number2 = st.number_input('Available  Area for Dry pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 10525*(p3*p4)**0.29  
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*(98.26-(109.04*(2.718)**(-5.75*(p4)))))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Infiltration Trench & Grassed Swale':
            number1 = st.number_input('Available Area for Infiltration Trench(sft)')
            number2 = st.number_input('Available  Area for Grassed Swale (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 42504*(p3*p4)**0.0344   
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*(97.7936-(107.28*(2.718)**(-5.85*(p4)))))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        
        if SCM_type=='Infiltration Trench & Constructed Wetland':
            number1 = st.number_input('Available Area for Infiltration Trench(sft)')
            number2 = st.number_input('Available  Area for Constructed Wetland (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 1875*(p3*p4)**0.503  
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*((4389.78*(p4)**0.012)-4286.26))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 1875*(p3*p4)**0.503  
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Infiltration Trench & Vegetative Filterbed':
            number1 = st.number_input('Available Area for Infiltration Trench(sft)')
            number2 = st.number_input('Available  Area for Vegetative Filterbed (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 687.5*(p3*p4)**0.59 
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*((584.706*(p4)**0.012)-560.448))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 687.5*(p3*p4)**0.59 
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*((29.031*(p4)**0.17)+ 8.47))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        
        if SCM_type=='Infiltration Trench & Wet Pond':
            number1 = st.number_input('Available Area for Infiltration Trench(sft)')
            number2 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 1875*(p3*p4)**0.503  
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*((4389.78*(p4)**0.012)-4286.26))/100
                         
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 27632*(p1*p2)**0.0431  + 1875*(p3*p4)**0.503
                        objective_2 = (((63767.5*(p2)**0.000285)-63679.2)*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Grassed Swale & Dry Pond':
            number1 = st.number_input('Available Area for Grassed Swale (sft)')
            number2 = st.number_input('Available  Area for Dry pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Bioretention")
                    tab1,tab2,tab3 = st.tabs(["graph","table","Cost"])
                    def simple_1d_fitness_func(p1,p2,p3,p4):
                        objective_1 = 42504*(p1*p2)**0.0344   + 10525*(p3*p4)**0.29 
                        objective_2 = ((97.7936-(107.28*(2.718)**(-5.85*(p2))))*(98.26-(109.04*(2.718)**(-5.75*(p4)))))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
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
                        fitnesses = simple_1d_fitness_func(pop,pop,pop,pop)
                        mean_fitnesses.append(np.mean(fitnesses,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses,config)
                   
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses = simple_1d_fitness_func(p1,p2,p3,p4)
                    xdata= pp_solutions_fitnesses[:,5]
                    ydata= pp_solutions_fitnesses[:,4]
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
                    ecost=1942.2
                    ccost=round((cost-ecost)*0.38,2)
                    opcost=round((cost-ecost)*0.59,2)
                    mcost=round(cost*0.32,2)
                    lcost=round(cost*0.29,2)
                    eqcost=round(cost*0.29,2)
                    encost=round(cost*0.06,2)
                    ocost=round(cost*0.04,2)
                    df= pd.DataFrame(pp_solutions_fitnesses,columns=["Area of Bioretention (sft)","depth of Bioretention (ft)","Area of Porous Pavement (sft)","depth of Porous Pavement (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df, x="cost (USD)", y="Reduction (%)", color='Area of Bioretention (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal-con_level, y1=removal + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nutrient Reduction = "+str(removal)+' % <br> Total Cost = $'+str(cost) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb <br> Available Total Phosphorus  Concentration = '+str(atp)+'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                        
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
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Grassed Swale & Constructed Wetland':
            number1 = st.number_input('Available Area for Grassed Swale (sft)')
            number2 = st.number_input('Available  Area for Constructed Wetland (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 42504*(p1*p2)**0.0344   + 1875*(p3*p4)**0.503  
                        objective_2 = ((97.7936-(107.28*(2.718)**(-5.85*(p2))))*((4389.78*(p4)**0.012)-4286.26))/100
                         
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 42504*(p1*p2)**0.0344   + 1875*(p3*p4)**0.503   
                        objective_2 = ((97.7936-(107.28*(2.718)**(-5.85*(p2))))*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Grassed Swale & Vegetative Filterbed':
            number1 = st.number_input('Available Area for Grassed Swale (sft)')
            number2 = st.number_input('Available  Area for Vegetative Filterbed (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 42504*(p1*p2)**0.0344  + 687.5*(p3*p4)**0.59 
                        objective_2 = ((97.7936-(107.28*(2.718)**(-5.85*(p2))))*((584.706*(p4)**0.012)-560.448))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 42504*(p1*p2)**0.0344  + 687.5*(p3*p4)**0.59 
                        objective_2 = ((97.7936-(107.28*(2.718)**(-5.85*(p2))))*((29.031*(p4)**0.17)+ 8.47))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        
        if SCM_type=='Grassed Swale & Wet Pond':
            number1 = st.number_input('Available Area for Grassed Swale (sft)')
            number2 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 42504*(p1*p2)**0.0344   + 1875*(p3*p4)**0.503  
                        objective_2 = ((97.7936-(107.28*(2.718)**(-5.85*(p2))))*((4389.78*(p4)**0.012)-4286.26))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 42504*(p1*p2)**0.0344   + 1875*(p3*p4)**0.503  
                        objective_2 = ((97.7936-(107.28*(2.718)**(-5.85*(p2))))*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Wet Pond & Vegetative Filterbed':
            number1 = st.number_input('Available Area for Wet Pond (sft)')
            number2 = st.number_input('Available  Area for Vegetative Filterbed(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 687.5*(p1*p2)**0.59 + 1875*(p3*p4)**0.503  
                        objective_2 = (((29.031*(p2)**0.17)+ 8.47)*((4389.78*(p4)**0.012)-4286.26))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 687.5*(p1*p2)**0.59 + 1875*(p3*p4)**0.503  
                        objective_2 = (((584.706*(p2)**0.012)-560.448)*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Vegetative Filterbed & Constructed Wetland':
            number1 = st.number_input('Available Area for Vegetative Filterbed (sft)')
            number2 = st.number_input('Available  Area for Constructed Wetland (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 687.5*(p1*p2)**0.59 + 1875*(p3*p4)**0.503   
                        objective_2 = (((29.031*(p2)**0.17)+ 8.47)*((4389.78*(p4)**0.012)-4286.26))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 687.5*(p1*p2)**0.59 + 1875*(p3*p4)**0.503  
                        objective_2 = (((584.706*(p2)**0.012)-560.448)*((260.665*(p4)**0.0285)-223.36))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
        if SCM_type=='Dry Pond & Vegetative Filterbed':
            number1 = st.number_input('Available Area for Dry pond (sft)')
            number2 = st.number_input('Available  Area for Vegetative Filterbed (sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4):
                        objective_1 = 10525*(p1*p2)**0.29 + 687.5*(p3*p4)**0.59 
                        objective_2 = ((98.26-(109.04*(2.718)**(-5.75*(p2))))*((584.706*(p4)**0.012)-560.448))/100
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4):
                        objective_1 = 10525*(p1*p2)**0.29 + 687.5*(p3*p4)**0.59   
                        objective_2 = ((98.26-(109.04*(2.718)**(-5.75*(p2))))*((29.031*(p4)**0.17)+ 8.47))/100
                        
                        return np.stack([p1,p2,p3,p4,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number,100)
                    p2= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention, Porous Pavement & Wet Pond':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Porous Pavement(sft)')
            number3 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4,p5,p6):
                        objective_1 = 29631*(p1*p2)**0.026 +40540*(p3*p4)**0.0327+ 1875*(p5*p6)**0.503 
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.9016-(105.3*(2.718)**(-5.51*(p4))))*((4389.78*(p6)**0.012)-4286.26))/10000
                        return np.stack([p1,p2,p3,p4,p5,p6,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4,p5,p6):
                        objective_1 = 29631*(p1*p2)**0.026 +40540*(p3*p4)**0.0327+ 1875*(p5*p6)**0.503 
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.9016-(105.3*(2.718)**(-5.51*(p4))))*((260.665*(p6)**0.0285)-223.36))/10000
                        return np.stack([p1,p2,p3,p4,p5,p6,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    p5 = np.linspace(100,number3,100)
                    p6= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention, Grassed Swale & Wet Pond':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Grassed Swale(sft)')
            number3 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4,p5,p6):
                        objective_1 = 29631*(p1*p2)**0.026 + 42504*(p3*p4)**0.0344 + 1875*(p5*p6)**0.503  
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.7936-(107.28*(2.718)**(-5.85*(p4))))*((4389.78*(p6)**0.012)-4286.26))/10000
                        return np.stack([p1,p2,p3,p4,p5,p6,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4,p5,p6):
                        objective_1 = 29631*(p1*p2)**0.026 + 42504*(p3*p4)**0.0344 + 1875*(p5*p6)**0.503   
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.7936-(107.28*(2.718)**(-5.85*(p4))))*((260.665*(p6)**0.0285)-223.36))/10000
                        return np.stack([p1,p2,p3,p4,p5,p6,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    p5 = np.linspace(100,number3,100)
                    p6= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention, Vegetative Filterbed & Wet Pond':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Vegetative Filterbed(sft)')
            number2 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4,p5,p6):
                        objective_1 = 29631*(p1*p2)**0.026 + 29631*(p3*p4)**0.026 +687.5*(p5*p6)**0.59   
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*((29.031*(p4)**0.17)+ 8.47)*((4389.78*(p6)**0.012)-4286.26))/10000
                        return np.stack([p1,p2,p3,p4,p5,p6,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4,p5,p6):
                        objective_1 = 29631*(p1*p2)**0.026 + 29631*(p3*p4)**0.026 +687.5*(p5*p6)**0.59    
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*((584.706*(p2)**0.012)-560.448)*((260.665*(p2)**0.0285)-223.36))/10000

                        return np.stack([p1,p2,p3,p4,p5,p6,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    p5 = np.linspace(100,number3,100)
                    p6= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention, Porous Pavement, Vegetative Filterbed & Wet Pond':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Porous Pavement(sft)')
            number3 = st.number_input('Available Area for Vegetative Filterbed(sft)')
            number4 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4,p5,p6,p7,p8):
                        objective_1 = 29631*(p1*p2)**0.026 + 40540*(p3*p4)**0.0327 + 687.5*(p5*p6)**0.59 + 1875*(p7*p8)**0.503  
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.9016-(105.3*(2.718)**(-5.51*(p4))))*((29.031*(p6)**0.17)+ 8.47)*((4389.78*(p8)**0.012)-4286.26))/1000000
                        return np.stack([p1,p2,p3,p4,p5,p6,p7,p8,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4,p5,p6,p7,p8):
                        objective_1 = 29631*(p1*p2)**0.026 + 40540*(p3*p4)**0.0327 + 687.5*(p5*p6)**0.59 + 1875*(p7*p8)**0.503   
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.9016-(105.3*(2.718)**(-5.51*(p4))))*((584.706*(p6)**0.012)-560.448)*((260.665*(p8)**0.0285)-223.36))/1000000
                        return np.stack([p1,p2,p3,p4,p5,p6,p7,p8,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    p5 = np.linspace(100,number3,100)
                    p6= np.linspace(0.1,0.5,100)
                    p7 = np.linspace(100,number4,100)
                    p8= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
            
        if SCM_type=='Bioretention, Porous Pavement, Grassed Swale & Wet Pond':
            number1 = st.number_input('Available Area for Bioretention(sft)')
            number2 = st.number_input('Available  Area for Porous Pavement(sft)')
            number3 = st.number_input('Available Area for Grassed Swale(sft)')
            number4 = st.number_input('Available  Area for Wet pond(sft)')
            tn= st.number_input('Initial Total Nitrogene Concentration (lb)')
            tp= st.number_input('Initial Total Phosphorus Concentration (lb)')
            removal = st.slider('Required Nutrient Reduction', 0.0, 100.0, 0.5)
            con_level = st.slider('Confidence interval', 0.0, 25.0)
            st.write(removal, '% Nutrient Reduction is needed')
            q=st.button('Run')
            if q:
                with col2:
                    st.subheader("Optimal Outcomes for Vegetative Filter Bed")
                    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["graph_N","Graph_P","Table_N","Table_P","Cost_N","Cost_P"])
                    def simple_1d_fitness_func_tp(p1,p2,p3,p4,p5,p6,p7,p8):
                        objective_1 = 29631*(p1*p2)**0.026 + 40540*(p3*p4)**0.0327 + 42504*(p5*p6)**0.0344  + 1875*(p7*p8)**0.503  
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.9016-(105.3*(2.718)**(-5.51*(p4))))*(97.7936-(107.28*(2.718)**(-5.85*(p6))))*((4389.78*(p8)**0.012)-4286.26))/1000000
                        return np.stack([p1,p2,p3,p4,p5,p6,p7,p8,objective_1,objective_2],axis=1)
                    def simple_1d_fitness_func_tn(p1,p2,p3,p4,p5,p6,p7,p8):
                        objective_1 = 29631*(p1*p2)**0.026 + 40540*(p3*p4)**0.0327 + 42504*(p5*p6)**0.0344  + 1875*(p7*p8)**0.503  
                        objective_2 = ((98-(117.1*(2.718)**(-5.21*(p2))))*(97.9016-(105.3*(2.718)**(-5.51*(p4))))*(97.7936-(107.28*(2.718)**(-5.85*(p6))))*((260.665*(p8)**0.0285)-223.36))/1000000
                        return np.stack([p1,p2,p3,p4,p5,p6,p7,p8,objective_1,objective_2],axis=1)
                    config = {
                                "half_pop_size" : 50,
                                "problem_dim" : 2,
                                "gene_min_val" : -1,
                                "gene_max_val" : 1,
                                "mutation_power_ratio" : 0.05,
                                }
                    
                    pop = np.random.uniform(config["gene_min_val"],config["gene_max_val"],2*config["half_pop_size"])
                    mean_fitnesses_tn = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tn = simple_1d_fitness_func_tn(pop,pop,pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tn.append(np.mean(fitnesses_tn,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tn,config)
                                
                    mean_fitnesses_tp = []
                    for generation in range(30):
                        # evaluate pop 
                        fitnesses_tp = simple_1d_fitness_func_tp(pop,pop,pop,pop,pop,pop,pop,pop)
                        mean_fitnesses_tp.append(np.mean(fitnesses_tp,axis=0))
                        
                        # transition to next generation
                        pop = NSGA2_create_next_generation(pop,fitnesses_tp,config)
                    
                    p1 = np.linspace(100,number1,100)
                    p2= np.linspace(0.1,0.5,100)
                    p3 = np.linspace(100,number2,100)
                    p4= np.linspace(0.1,0.5,100)
                    p5 = np.linspace(100,number3,100)
                    p6= np.linspace(0.1,0.5,100)
                    p7 = np.linspace(100,number4,100)
                    p8= np.linspace(0.1,0.5,100)
                    pp_solutions_fitnesses_tn = simple_1d_fitness_func_tn(p1,p2)
                    pp_solutions_fitnesses_tp = simple_1d_fitness_func_tp(p1,p2)
                    x1data= pp_solutions_fitnesses_tn[:,3]
                    y1data= pp_solutions_fitnesses_tn[:,2]
                    x2data= pp_solutions_fitnesses_tp[:,3]
                    y2data= pp_solutions_fitnesses_tp[:,2]
                    def Gauss(x, A,B,C):
                        y = A*x + B*x**2 + C
                        return y
                    parameters1, covariance1 = curve_fit(Gauss, x1data, y1data)
                    fit_A1 = parameters1[0]
                    fit_B1 = parameters1[1]
                    fit_C1= parameters1[2]
                    cost1 = fit_A1*removal_n +fit_B1*removal_n**2 + fit_C1 
                    parameters2, covariance2 = curve_fit(Gauss, x2data, y2data)
                    fit_A2 = parameters2[0]
                    fit_B2 = parameters2[1]
                    fit_C2= parameters2[2]
                    cost2 = fit_A2*removal_p +fit_B2*removal_p**2 + fit_C2
                    atn= round((tn*removal_n)/100,2)
                    atp= round((tp*removal_p)/100,2)
                    ecost=7380.1
                    ccost1=round((cost1-ecost)*0.25,2)
                    opcost1=round((cost1-ecost)*0.59,2)
                    mcost1=round(cost1*0.19,2)
                    lcost1=round(cost1*0.62,2)
                    eqcost1=round(cost1*0.13,2)
                    encost1=round(cost1*0.04,2)
                    ocost1=round(cost1*0.03,2)
                    ccost2=round((cost2-ecost)*0.25,2)
                    opcost2=round((cost2-ecost)*0.59,2)
                    mcost2=round(cost2*0.19,2)
                    lcost2=round(cost2*0.62,2)
                    eqcost2=round(cost2*0.13,2)
                    encost2=round(cost2*0.04,2)
                    ocost2=round(cost2*0.03,2)
    
                    df1= pd.DataFrame(pp_solutions_fitnesses_tn,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])  
                    df2= pd.DataFrame(pp_solutions_fitnesses_tp,columns=["Area (sft)","depth (ft)","cost (USD)","Reduction (%)"])
                    with tab1:
                        fig1 = px.line(df1, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df1, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_n,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_n-con_level, y1=removal_n + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Nitrogene Reduction = "+str(removal_n)+' % <br> Total Cost = $'+str(cost1) + '<br> Available Total Nitrogene Concentration = '+str(atn)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab2:
                        fig1 = px.line(df2, x="cost (USD)", y="Reduction (%)")
                        fig2 = px.scatter(df2, x="cost (USD)", y="Reduction (%)", color='Area (sft)',color_continuous_scale=px.colors.sequential.Bluered)
                        fig3 = go.Figure(data=fig1.data + fig2.data)
                        fig2.add_hline(y=removal_p,name= 'Reduction level',line=dict(color='firebrick', width=2,
                                                      dash='dash'))
                        fig2.add_hrect(y0=removal_p-con_level, y1=removal_p + con_level, line_width=0, fillcolor="red", opacity=0.2)
                        fig2.add_annotation(
                                    text="Total Phosphorus Reduction = "+str(removal_p)+' % <br> Total Cost = $'+str(cost2) + '<br> Available Total Phosphorus Concentration = '+str(atp)+ 'lb',
                                    align = 'left',
                                    showarrow= False,
                                    xref='paper',
                                    yref='paper',
                                    font=dict(family='Arial',
                                              size=20,
                                              color='black'),
                                    bgcolor='rgb(127,205,187)',
                                    y=1,
                                    x=0,
                                    xanchor='left')
                        fig2.update_layout(height=600, width=800,  
                              
                         showlegend=True,
                         
                         font=dict(
                             family="Arial",
                             size=25,
                             color="Black"),
                         xaxis=dict(
                             showline=True,
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
                                     title='Nutrient Reduction (%)',
                                     titlefont=dict(
                                 family='Arial',
                                 size = 25,
                                 color= 'black'),
                                     showline=True,
                                     showgrid=False,
                                     showticklabels=True,
                                     linecolor='black',
                                     linewidth=2,
                                     ticks='outside',
                                     tickfont=dict(
                                         family='Arial',
                                         size=20,
                                         color='black',
                                         )))
                        st.plotly_chart(fig2, use_container_width=True)
                    with tab3:
                        st.dataframe(df1)
                    with tab4:
                        st.dataframe(df2)
                    with tab5:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost1],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost1],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost1],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost1],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost1],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost1],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost1],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
                    with tab6:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                           y=['Total Cost (Life Cycle Stages)'],
                           x=[0],
                           name='Planning Cost',
                           orientation='h',
                           marker=dict(
                            color='rgba(246, 78, 139, 0.6)',
                            line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                                  )
                                  ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ccost2],
                        name='Construction Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(202, 0, 32, 0.6)',
                        line=dict(color='rgba(202, 0, 32, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[opcost2],
                        name='Operations and Maintenance Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(230, 97, 1, 0.6)',
                        line=dict(color='rgba(230, 97, 1, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Stages)'],
                        x=[ecost],
                        name='End of Life Cost',
                        orientation='h',
                        marker=dict(
                        color='rgba(58, 71, 80, 0.6)',
                        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[mcost2],
                        name='Materials',
                        orientation='h',
                        marker=dict(
                        color='rgba(227, 26, 28, 0.6)',
                        line=dict(color='rgba(227, 26, 28, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[lcost2],
                        name='Labors',
                        orientation='h',
                        marker=dict(
                        color='rgba(251, 154, 153, 0.6)',
                        line=dict(color='rgba(251, 154, 153, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[eqcost2],
                        name='Equipments',
                        orientation='h',
                        marker=dict(
                        color='rgba(51, 160, 44, 0.6)',
                        line=dict(color='rgba(51, 160, 44, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[encost2],
                        name='Energy',
                        orientation='h',
                        marker=dict(
                        color='rgba(31, 120, 180, 0.6)',
                        line=dict(color='rgba(31, 120, 180, 1.0)', width=3)
                           )
                            ))
                        fig.add_trace(go.Bar(
                        y=['Total Cost (Life Cycle Cost Type)'],
                        x=[ocost2],
                        name='Others',
                        orientation='h',
                        marker=dict(
                        color='rgba(166, 206, 227, 0.6)',
                        line=dict(color='rgba(166, 206, 227, 1.0)', width=3)
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
                                         )))
                        st.plotly_chart(fig, use_container_width=True)
