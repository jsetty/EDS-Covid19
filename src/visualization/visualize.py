import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import dash_daq as daq

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
print(os.getcwd())
df_input_large=pd.read_csv('../data/processed/COVID_final_set.csv',sep=';')
df_input_recov=pd.read_csv('../data/processed/COVID_final_recov_set.csv',sep=';')
df_analyse = pd.read_csv('../data/processed/COVID_full_flat_table.csv',sep=';')
df_SIR_data = pd.read_csv('../data/processed/COVID_SIR_Model_Data.csv',sep=';')


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Enterprise Data Science - Course Project: Covid 19

    Automated data gathering, data transformations,
    filtering and machine learning to approximating the doubling time, and
    (static) deployment of responsive dashboard on Covid-19 Data

    '''),

    dcc.Markdown('''
    ## Select Country for Country level Stats
    '''),

    dcc.Dropdown(
        id='country_drop_down_stats',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value='India', # which is pre-selected
        multi=False
    ),
    html.Div(children=[
                dcc.Markdown('''
                **Scale Modes**
                '''),
                dcc.RadioItems(
                    options=[
                        {'label': 'Uniform', 'value': 'linear'},
                        {'label': 'Logarithmic', 'value': 'log'},
                        
                    ],
                    value='linear',
                    id='scale_type',
                    labelStyle={'display': 'inline-block'}
                ),
            
            ]),
    dcc.Graph( id='multi_graph'),
    dcc.Markdown('''
    ## Doubling Rate Visualisation
    '''),

    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['US', 'Germany'], # which are pre-selected
        multi=True
    ),

    dcc.Dropdown(
    id='doubling_time',
    options=[
        
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed_DR',
    multi=False
    ),
    dcc.Graph(figure=fig, id='main_window_slope'),
    dcc.Markdown('''
            ##  SIR model visualization
            '''),
            dcc.Dropdown(
                id = 'country_drop_down_sir',
                options = [{'label':name, 'value':name} for name in df_input_large['country'].unique()],
                value = [ 'US'], # default selected values
                multi = True # for allowing multi value selection
            ),
            dcc.Graph(figure=fig, id='sir_chart')
], style={'textAlign': 'center'})


@app.callback(
    Output('multi_graph', 'figure'),
    [Input('country_drop_down_stats', 'value'),
    Input('scale_type', 'value')]
)
def update_cummulative_stacked_plot(country,scale_type):
    traces = []
    df_plot=df_input_large[df_input_large['country'] == country]
    df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date','deaths']].groupby(['country','date']).agg(np.sum).reset_index()
    df_plot_recov = df_input_recov[df_input_recov['country'] == country]
    df_plot_recov=df_plot_recov[['state','country','recovered','date']].groupby(['country','date']).agg(np.sum).reset_index()

    fig=make_subplots(rows=4, cols=1,
                subplot_titles=("Total Confirmed", "Total Active ", "Total Recovered", 'Total Deaths'),
                shared_xaxes=False,
    )
    fig.add_trace(go.Scatter(
                        x=df_plot.date,
                        y=df_plot['confirmed'],
                        mode='markers+lines',
                        showlegend=False,
                        opacity=0.9
                 ), row=1,col=1
    )
    
    fig.add_trace(go.Scatter(
                        x=df_plot.date,
                        y=df_plot['confirmed']-df_plot['deaths'],
                        mode='markers+lines',
                        showlegend=False,
                        opacity=0.9
                 ), row=2,col=1
    )
    
    fig.add_trace(go.Scatter(
                        x=df_plot_recov.date,
                        y=df_plot_recov['recovered'],
                        mode='markers+lines',
                        showlegend=False,
                        opacity=0.9
                 ), row=3,col=1
    )
    
    fig.add_trace(go.Scatter(
                        x=df_plot.date,
                        y=df_plot['deaths'],
                        mode='markers+lines',
                        showlegend=False,
                        opacity=0.9
                 ), row=4,col=1
    )
    
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=1, col=1)
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=2, col=1)
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=3, col=1)
    fig.update_xaxes(type="date",
                    tickangle=-45,
                    nticks=20,
                    tickfont=dict(size=14,color="#7f7f7f"), 
                    row=4, col=1)
    fig.update_yaxes(type=scale_type, row=1, col=1, title='Confirmed')
    fig.update_yaxes(type=scale_type, row=2, col=1, title='Active')
    fig.update_yaxes(type=scale_type, row=3, col=1, title='Recovered')
    fig.update_yaxes(type=scale_type, row=4, col=1, title='Deaths')
    fig.update_layout(dict (

                width=1300,
                height=750,
                template="plotly"
                
        ))
    return fig

@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')]
    )
def update_figure(country_list, show_doubling):
    if 'doubling_rate' in show_doubling:
        my_yaxis={'type':"log",
               'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
              }
    else:
        my_yaxis={'type':"log",
                  'title':'Confirmed infected people (source johns hopkins csse, log-scale)'
              }
    traces = []
    for each in country_list:
        df_plot=df_input_large[df_input_large['country']==each]
        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],
                                mode='markers+lines',
                                opacity=0.9,
                                name=each,
                        )
                )
    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },
                yaxis=my_yaxis
        )
    }


@app.callback(
    Output('sir_chart', 'figure'),
    [Input('country_drop_down_sir', 'value')])
def update_figure(country_list):
    traces = []
    if(len(country_list) > 0):
        for each in country_list:
            country_data = df_analyse[each][35:]
            ydata = np.array(country_data)
            t = np.arange(len(ydata))
            fitted = np.array(df_SIR_data[each])
            traces.append(dict(
                x = t,
                y = ydata,
                mode = 'markers+lines',
                name = each+str(' - Truth'),
                opacity = 0.9
            ))
            traces.append(
            dict(
                x = t,
                y = fitted,
                mode = 'markers+lines',
                name = each+str(' - Simulation'),
                opacity = 0.9
            ))
    return {
        'data': traces,
        'layout': dict(
            width = 1280,
            height = 720,
            title = 'Fit of SIR model for: '+', '.join(country_list),
            xaxis = {
                'title': 'Days', #'Fit of SIR model for '+str(each)+' cases',
                'tickangle': -45,
                'nticks' : 20,
                'tickfont' : dict(size = 14, color = '#7F7F7F')
            },
            yaxis = {
                'title': 'Population Infected',
                'type': 'log'
            }
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False,port=5555)