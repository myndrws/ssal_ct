##################################################
# Purpose: Create dashboard app to visualise AL  #
# Author: Amy Andrews                            #
# Resources used:
# DASH documentation and tutorials available at
# https://dash.plotly.com/
##################################################

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from utils import convert_to_names
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename

import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import matplotlib.image as mpimg
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import config

#########################
# app prep
#########################

# prepare css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
tab_style = {
    'padding': '6px',
    'backgroundColor': '#111111',
    'width': '10%'
}
tab_selected_style = {
    'backgroundColor': '#7FDBFF',
    'color': '#111111',
    'fontWeight': 'bold',
    'padding': '6px',
    'width': '10%'
}

# prepare data
names = convert_to_names()
colours = px.colors.qualitative.Light24[:20] + ["black", "black"]
scatter_data = pd.read_csv("embs/scatterDataFrame_customQuerying80LabsAt132Epochs.csv")
scatter_data['Pseudo-label'].replace("Thomsons Gazelle","Thomson's Gazelle", inplace=True)
scatter_data['Species'].replace("Thomsons Gazelle","Thomson's Gazelle", inplace=True)
scatter_data['AnimalsLabels'].replace("Thomsons Gazelle","Thomson's Gazelle", inplace=True)
scatter_data['Pseudo-label'].replace("Grants Gazelle","Grant's Gazelle", inplace=True)
scatter_data['Species'].replace("Grants Gazelle","Grant's Gazelle", inplace=True)
scatter_data['AnimalsLabels'].replace("Grants Gazelle","Grant's Gazelle", inplace=True)
im_directory = ""

########################
# component creation
########################

# create the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#create authentication
with open('app_auth.json', 'rb') as handle:
    VALID_USERNAME_PASSWORD_PAIRS = json.load(handle)
handle.close()
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

# create the umap visualisation
fig = px.scatter(scatter_data,
                 x="x", y="y", color="AnimalsLabels",
                 hover_name='HoverName',
                 hover_data={'x': False,
                             'y': False,
                             'x3d': False,
                             'y3d': False,
                             'z3d': False,
                             'ID': True,
                             'PseudoAndTrueAmalg': False,
                             "AnimalsLabels": False,
                             'Label Type': False,
                             'QueriesOrNot': False,
                             'True Label': False,
                             'Pseudo-label': False,
                             'Species': False,
                             'Daytime': True,
                             'Image Path': False  # currently hidden as contains location ids
                             },
                 opacity=1,
                 symbol='Label Type',
                 symbol_sequence=['circle-open', 'star', 'diamond'],
                 size='QueriesOrNot',
                 size_max=14,
                 color_discrete_sequence=colours,
                 category_orders={"AnimalsLabels": names + ['True Label',
                                                            'Suggested Query']}
                 )

fig.update_yaxes(visible=False, showticklabels=False)
fig.update_xaxes(visible=False, showticklabels=False)
fig.update_layout(legend_title_text='Species and current label status', clickmode='event+select',
                  plot_bgcolor=colors['background'],
                  paper_bgcolor=colors['background'],
                  font_color=colors['text'],
                  modebar_add=['drawopenpath',
                               'eraseshape'
                               ]
                  )
# fig.update_traces(marker=dict(line=dict(width=1)),
#                   selector=dict(mode='markers'),
#                   )
# fig.update_traces(marker_size=8)
fig.update_layout({'dragmode': 'pan'})


# create image div
def create_image_div(img_path):
    imread = mpimg.imread(os.path.join(im_directory,
                                       img_path.replace('_', '/', 1)))
    im1 = px.imshow(imread)
    im1.update_yaxes(visible=False, showticklabels=False)
    im1.update_xaxes(visible=False, showticklabels=False)
    im1.update_layout(plot_bgcolor=colors['text'],
                      paper_bgcolor=colors['text'],
                      margin=dict(l=5, r=5, t=10, b=10),
                      font_color=colors['text'],
                      modebar_add=['drawopenpath',
                                   'eraseshape'
                                   ])
    return im1


# placeholder figure for when there's nothing to show
def blank_fig():
    fig = px.scatter()
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_layout(template=None,
                      plot_bgcolor=colors['background'],
                      paper_bgcolor=colors['background'])
    return fig


# label names list
def create_radio_buttons():
    names = convert_to_names()
    labelling_buttons = []
    for i in range(len(names)):
        labelling_buttons.append({'label': names[i], 'value': i})
    return labelling_buttons


#########################
# defining layout
#########################

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='Active Learning for camera trap data',
            style={
                'textAlign': 'left',
                'color': colors['text'],
                'padding-top': '10px',
                'margin-left': '50px'
            }
            ),

    dcc.Tabs(style= {'float': 'right', 'display':'inline', 'margin-left': '50px'} ,children=[
        dcc.Tab(label='Dashboard', style=tab_style, selected_style=tab_selected_style, children=[

            ### EVERYTHING FOR TAB ONE, THE MAIN CONTENT

            html.Div(children='''
                    Zoom in and select a point to get started!
                ''', style={
                'textAlign': 'left',
                'color': colors['text'],
                'padding-bottom': '10px',
                'margin-left': '50px',
                'margin-top': '10px'
            }),

            dcc.Graph(
                id='scatter-graph',
                figure=fig,
                config=dict({'scrollZoom': True, 'displaylogo': False}),
                hoverData={'points': [{
                    'customdata': ''}]},
                style={'width': '58%', 'height': '88vh', 'display': 'inline-block'}
            ),

            html.Div(id='viewing-container', children=[

                dcc.Graph(
                    id='image-vis',
                    style={'height': '40vh', 'width': '100%', 'display': 'inline-block',
                           'vertical-align': 'top', 'padding': '20px 20px 20px 20px'},
                    figure=blank_fig()  # no figure shown on start up
                ),

                html.Div(id='radio-buttons-container',
                         children=[
                             dcc.RadioItems(
                                 id='labelling-buttons',
                                 options=create_radio_buttons(),
                                 value=100,  # just setting the value to something it can't be for labelling
                                 labelStyle={'display': 'inline-block', 'color': colors['text'],
                                             'margin': '2px 10px 0 10px'}
                             ),
                             html.Button(id='submit-button-state',
                                         n_clicks=0,
                                         children='Add label',
                                         style={'display': 'none'}),
                             html.Div(children="",
                                      id="button-clicked-message",
                                      style={'display': 'inline-block', 'margin': '2px 10px 0 10px'}),
                             dcc.Store(id='intermediate-values'),
                             html.Button(id="download-button",
                                         children="Download Labels",
                                         style={'display': 'none'}),
                             dcc.Download(id="download-text"),
                             html.Div(children="You have labelled 0 images in this session",
                                      id="label-tally",
                                      style={'position': 'fixed', 'bottom': '0', 'margin': '20px'}),
                         ], style={'display': 'none', 'padding': '20px 20px 20px 20px'}
                         # on first loading display is none
                         ),

            ], style={'width': '35%', 'height': '88vh', 'display': 'inline-block',
                      'vertical-align': 'top', 'padding-top': '25px', 'padding-left': '10px', 'margin-left': '20px'}
                     # this is the style for the whole right panel div
                     )

        ]),  # end of tab one

        dcc.Tab(label='Info', style=tab_style, selected_style=tab_selected_style, children=[

            ### EVERYTHING FOR TAB TWO, INFO TAB

            html.Div(children='''
                
                The camera trap data used here was provided by the Biome Health Project, a research project executed by 
                UCL's Centre for Biodiversity and Environment Research and funded by WWF-UK.
                Find out more here https://www.biomehealthproject.com/ 
                
                The data used in this dashboard is a set of training data on which a machine learning model has been 
                constructed, which makes use of both the unlabelled and labelled data contained within. The model 
                constructs representations of the images shown, which are being visualised in the scatter plot. Ideally,
                representations of the same species should be 'closer' to one another in the visualisation than 
                representations of different species. 
                
                A true label is the ground-truth for the species shown in the cropped image. This was a label assigned
                at an earlier date when a human labelled this training data. The model used to construct the 
                representations shown here had access to only 80 true labels in order to learn the features of different
                species contained within the images. The rest of the learning has come from associating those available
                true labels with the abundant unlabelled data.
                
                A pseudo-label is a guess which the model has made as to the true label of the species in the cropped
                image shown. This is the model's best guess based on what it knows so far. 
                
                A query is an image that the model suggests could be useful to get true labels for so that it can 
                maximise accuracy on continued training knowing the correct ground-truth for this image.
                 
                As some species are much rarer than others, the model tends to perform a little worse for these species.
                This is why some of the pseudo-labels are correct less often for the minority species. 
    
                The app was developed by Amy Andrews as part of the MSc thesis project at UCL, 
                supervised by Gabriel Brostow.  
                
            ''', style={
                'textAlign': 'left',
                'color': colors['text'],
                'height': '88vh',
                'padding': '25px',
                'margin': '0px 50px 75px 50px',
                'whiteSpace': 'pre-wrap'
            }),

        ]),  # end of tab two

    ])  # end of tab layout

])  # end of app layout


###########################
# defining interactivity
###########################

# callback for hovering on plot
@app.callback(
    Output('image-vis', 'figure'),
    Input('scatter-graph', 'hoverData'),
    Input('scatter-graph', 'selectedData'), prevent_initial_call=True)
def show_hovered_image(hoverData, selectedData):
    if selectedData is not None:
        im_path = selectedData['points'][0]['customdata'][12]  # this is the impath
    else:
        im_path = hoverData['points'][0]['customdata'][12]
    # based on advice from https://community.plotly.com/t/writing-secure-dash-apps-community-thread/54619
    safe_file_path = secure_filename(im_path)
    return create_image_div(img_path=safe_file_path)


@app.callback(
    Output('radio-buttons-container', 'style'),
    Input('scatter-graph', 'selectedData'))
def show_labelling_panel(selectedData):
    if selectedData is not None:
        return {'display': 'inline-block', 'color': colors['text'], 'margin': '20px 10px 0 10px'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('submit-button-state', 'style'),
    Input('labelling-buttons', 'value'))
def show_submit_button(value):
    if value != 100 and value is not None:
        # print('value:', value)
        return {'display': 'inline-block', 'color': colors['text'], 'margin': '20px 10px 0 10px'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('labelling-buttons', 'value'),
    Output('submit-button-state', 'n_clicks'),
    Input('scatter-graph', 'selectedData'),
    State('intermediate-values', 'data'),
    prevent_initial_call=True)
def refresh_labelling_panel(selectedData, data):
    # when points are changed reset them
    if selectedData is not None:
        imID = selectedData['points'][0]['customdata'][3]
        if data is not None and str(imID) in data.keys():
            return data[str(imID)], 1
        else:
            return None, 0
    else:
        return None, 0


@app.callback(
    Output("download-button", "style"),
    Input("submit-button-state", "n_clicks"),
    prevent_initial_call=True)
def reveal_download_button(n_clicks):
    if n_clicks > 0:
        return {'display': 'block', 'vertical-align': 'bottom', 'margin': '20px 10px 0 10px'}
    else:
        return {'display': 'none'}


# check if the id already existed in dict
# update the existing entry with the new entry
# if it does then the output message reads 'label updated'
# if the id does not already exist
# add the tuple to the end of the list
# the output message reads 'label added'
@app.callback(
    Output('button-clicked-message', 'children'),
    Output('intermediate-values', 'data'),
    Input('submit-button-state', 'n_clicks'),
    State('scatter-graph', 'selectedData'),
    State('labelling-buttons', 'value'),
    State('intermediate-values', 'data'),
    prevent_initial_call=True)
def submit_button_clicked(n_clicks, selectedData, value, data):
    # this whole thing should only fire if the components exist
    if value != 100 and value is not None and selectedData is not None:
        imID = selectedData['points'][0]['customdata'][3]
        data = data or {}
        data[str(imID)] = value  # a dictionary ensures the values is always the latest one
        if n_clicks == 1:
            msg = "'{}' label added!".format(names[value])
        elif n_clicks > 1:
            msg = "Label updated to '{}'".format(names[value])
        else:
            msg = ""
        return msg, data
    elif value is None or value == 100:
        # this is basically in the instance no label assigned
        msg = ""
        return msg, data
    else:
        raise PreventUpdate


# define a download of the session store for data on button click
@app.callback(
    Output("download-text", "data"),
    Input("download-button", "n_clicks"),
    State("intermediate-values", "data"),
    prevent_initial_call=True)
def download_session_store(n_clicks, data):
    return dict(content=str(data), filename="labels.txt")

# count how many images have been labelled
@app.callback(
    Output("label-tally", "children"),
    Input("intermediate-values", "data"),
    prevent_initial_call=True)
def count_labelled(data):
    if data is not None:
        num_lab = len(data.keys())
    else:
        num_lab = 0
    return "You have labelled {} images in this session".format(num_lab)


if __name__ == '__main__':
    app.run_server(debug=True)
