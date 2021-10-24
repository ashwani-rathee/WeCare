import time
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import base64
import datetime
import io
import plotly.graph_objs as go
import cufflinks as cf
from dash import dash_table
import dash_bootstrap_components as dbc
import skimage.io as sio
import base64
from PIL import Image
import numpy as np
import requests
import pandas as pd
from skimage import data
import xml.etree.ElementTree as ET
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, find_boundaries
from skimage.transform import resize
img = data.chelsea()  # or any image represented as a numpy array
mask = np.zeros((512, 512))
with open("img.png", "rb") as image_file:
        imgcontents = "data:image/png;base64, " + \
            str(base64.b64encode(image_file.read()))[2:-1]
with open("mask.png", "rb") as image_file:
        maskcontents = "data:image/png;base64, " + \
            str(base64.b64encode(image_file.read()))[2:-1]
app = dash.Dash(__name__, external_stylesheets=[
                'https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP, "./test.css"])
server = app.server

table_header = [
    html.Thead(html.Tr([html.Th(html.H3("Steps in Self Examination"))]))
]

row1 = html.Tr([
    html.Td(html.H3('''
           Step 1: Begin by looking at your breasts in the mirror with your shoulders straight and your arms on your hips.

            Here's what you should look for:

            Breasts that are their usual size, shape, and color
            Breasts that are evenly shaped without visible distortion or swelling
            If you see any of the following changes, bring them to your doctor's attention:

            Dimpling, puckering, or bulging of the skin
            A nipple that has changed position or an inverted nipple (pushed inward instead of sticking out)
            Redness, soreness, rash, or swelling        
           ''')),
])
row2 = html.Tr([
    html.Td(html.H3('''
            Step 2: Now, raise your arms and look for the same changes. \n

            Step 3: While you're at the mirror, look for any signs of fluid coming out of one or both nipples (this could be a watery, milky, or yellow fluid or blood).
           ''')),
])
row3 = html.Tr([
    html.Td(html.H3('''
            Step 4: Next, feel your breasts while lying down, using your right hand to feel your left breast and then your left hand to feel your right breast.
            Use a firm, smooth touch with the first few finger pads of your hand, keeping the fingers flat and together. Use a circular motion, about the size of a quarter. \n

            Cover the entire breast from top to bottom, side to side — from your collarbone to the top of your abdomen, and from your armpit to your cleavage.

            Follow a pattern to be sure that you cover the whole breast. You can begin at the nipple, moving in larger and larger circles until you reach the outer edge of the breast.
            You can also move your fingers up and down vertically, in rows, as if you were mowing a lawn. This up-and-down approach seems to work best for most women.
            Be sure to feel all the tissue from the front to the back of your breasts: for the skin and tissue just beneath, 
            use light pressure; use medium pressure for tissue in the middle of your breasts; use firm pressure for the deep tissue in the back.
            When you've reached the deep tissue, you should be able to feel down to your ribcage.      
           ''')),
])
row4 = html.Tr([
    html.Td(html.H3('''
                Step 5: Finally, feel your breasts while you are standing or sitting. Many women find that the easiest way to 
                feel their breasts is when their skin is wet and slippery, so they like to do this step in the shower. 
                Cover your entire breast, using the same hand movements described in step 4.
           ''')),
])

table_body = [html.Tbody([row1, row2, row3, row4])]

table = dbc.Table(table_header + table_body, bordered=True, 
                  hover=True,
                  responsive=True,
                  striped=True,)


app.layout = html.Div([
    dbc.Row([dbc.Col(html.Img(src='./assets/1.png', className="logo")),
             dbc.Col(
            dbc.Nav(
                [
                    dbc.NavLink("About Breast Cancer",
                                href="#about", external_link=True,className="navlink"),
                    dbc.NavLink("Self Examination", href="#table",
                                external_link=True,className="navlink"),
                    dbc.NavLink("Annotation & Classification",
                                href="#annotate", external_link=True,className="navlink"),
                    dbc.NavLink("References", href="#footer",
                                external_link=True,className="navlink"),
                ],
                vertical=False,
                pills=True,card=True
            ),),

    ], className='header'),
    html.Div(
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H1('Breast Cancer Awareness Month',
                            className='big-awareness'),
                    html.H3('This is my submission for Cal hacks 8.0 which provides a tool to annotate breast tumor which is computer aided(the initial mask comes for radiologists).Automatic segmentation of breast tumors from medical images is important for clinical assessment and treatment planning of breast tumors. '),
                ],
                    className="bigbox")),
            # <iframe width="560" height="315" src="https://www.youtube.com/embed/8q2JZk6Cnmc" title="YouTube video player" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            dbc.Col([html.H1('Its time to RISE', className='big-awareness1'),
                     html.Div(html.Iframe(width="560", height="315", src="https://www.youtube.com/embed/8q2JZk6Cnmc", title="YouTube video player",
                                          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"), className="bigbozframe")], className="bigbox12"),
        ], className="headerrol1"), className='divHeader1'),
    dbc.Col([
        dbc.Row(html.H1('Annotation And Classification Tool',
                        className='big-awareness')),
        dbc.Row([
            dbc.Col(
                html.Div([
                    dbc.Col([
                        dbc.Row([html.Div([html.P("Image"),
                                           html.Hr(), html.Div(html.Img(src=imgcontents),id='output-image-upload')], className="graphout")]),
                        dbc.Row([html.Div([html.P("Mask"),
                                           html.Hr(), html.Div(html.Img(src=maskcontents),id='mask-image-upload')], className="graphout")])
                    ])
                ])
            ),
            dbc.Col(
                html.Div([
                    dbc.Col(html.Div([html.P("Graph"),
                                      html.Hr(), html.Div(id='graph')], className="graphout1")),
                ])
            ),
            dbc.Col(
                html.Div([
                    dbc.Col(html.Div([html.P("Classification Model Results"),
                                      html.Hr(),
                                      html.Div(id='button-clicks', className="classifys",style={'whiteSpace': 'pre-line'}),
                    ], className="graphout2")),
                ])
            )
        ]),
        dbc.Row([html.Hr()]),
        dbc.Row([dbc.Col(dbc.Row([
            # dbc.Col(html.Div(id='display-value')),
            dbc.Col(html.Button('Show Mask', id='button-mask-show')),
            dbc.Col(html.Button('Show Graph', id='button')),
            dbc.Col(html.Button('Classify this Image', id='classify')),
            
            dbc.Col([html.Button('Save Example Image for test', id='example-save'),
                     dcc.Download(id="example-save-index")]),
            dbc.Col([html.Button('Save Mask', id='mask-save'),
                     dcc.Download(id="mask-save-index")]),
            # dbc.Col([html.Button('Save Graph', id='graph-save'),
            #          dcc.Download(id="graph-save-index")]),
        ]),), dbc.Col(dcc.Upload(html.Button('Upload File'), id="upload-image",),)]), ], className="annotationtool", id="annotate"),
        dbc.Row([
        dbc.Col(
            html.Div([
                html.H1('About Breast Cancer',
                        className='big-awareness'),
                html.H2('''Overview'''),
                html.H3('''
Breast cancer is cancer that forms in the cells of the breasts.

After skin cancer, breast cancer is the most common cancer diagnosed in women in the United States. Breast cancer can occur in both men and women, but it's far more common in women.

Substantial support for breast cancer awareness and research funding has helped created advances in the diagnosis and treatment of breast cancer. Breast cancer survival rates have increased, and the number of deaths associated with this disease is steadily declining, largely due to factors such as earlier detection, a new personalized approach to treatment and a better understanding of the disease.'''),

            ], className="bigbox2",id="about")),
        dbc.Col(table,id="table",className="table"),
        ]), 
    dbc.Row([
        dbc.Col(html.Button(html.A('Github', href='https://github.com/ashwani-rathee/breast_cancer_awareness',
                                   target="_blank"), id='button5', style={'width': '100%'})),
        dbc.Col(html.Button(html.A('Devpost', href='https://devpost.com/software/breast-cancer-tool',
                                   target="_blank"), id='button6', style={'width': '100%'})),
        dbc.Col(html.Button(html.A('Linkedin', href='https:/github.com',
                                   target="_blank"), id='button7', style={'width': '100%'})),
        dbc.Col(html.Button(html.A('Youtube Vid', href='https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                                   target="_blank"), id='button8', style={'width': '100%'})),
    ], className='divfooter', id="footer"
    ),


])

@app.callback(
    Output('button-clicks', 'children'),
    [Input('classify', 'n_clicks')])
def clicks(n_clicks):
    if n_clicks:
        resp = requests.post("https://breast-cancer-api.as.r.appspot.com/classify",
                            files={"file": open('img.png', 'rb')})
        json_load= resp.json()
        predict = np.asarray(json_load["class"])
        if predict==0:
            return "According to classifier model, YOU ARE IN NORMAL CONDITION NO NEED TO WORRY ABOUT"
        elif predict==1:
            return "According to classifier model, The cells are not yet cancerous, but they have the potential to become malignant consult the doctor"
        else:
            return "Malignant tumors are cancerous. The cells can grow and spread to other parts of the body."
        
    return ''

@app.callback(Output("mask-save-index", "data"), Input("mask-save", "n_clicks"))
def func(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    else:
        return dcc.send_file(
            "./mask.png"
        )


@app.callback(Output("example-save-index", "data"), Input("example-save", "n_clicks"))
def func(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    else:
        return dcc.send_file(
            "./example.png"
        )


def parse_contents(contents, filename, date):
    global mask
    data = contents[contents.find(','):-1]+'=='
    imgdata = np.array(Image.open(io.BytesIO(base64.b64decode(data))))
    shape = 256
    imgdata = resize(imgdata, (shape, shape))
    sio.imsave('img.png', imgdata)
    with open("img.png", "rb") as image_file:
        contents = "data:image/png;base64, " + \
            str(base64.b64encode(image_file.read()))[2:-1]

    resp = requests.post("https://breast-cancer-api.as.r.appspot.com/predict",
                         files={"file": open('img.png', 'rb')})
    json_load = resp.json()
    mask = ~np.asarray(json_load["mask"])
    sio.imsave('mask.png', mask)
    print("mask updated")
    return html.Div([

        html.Img(src=contents),
    ])

counter = 0 
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    global counter
    if counter == 0:
        counter = counter + 1
        return html.Div([html.Img(src=imgcontents),])
    if list_of_contents is not None:
        children = [
            parse_contents(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('mask-image-upload', 'children'),
              [Input('button-mask-show', 'n_clicks')], State('upload-image', 'contents'))
def show_mask(n_clicks, contents):
    global counter
    if counter == 0:
        return html.Div([html.Img(src=maskcontents),])
    if n_clicks:
        filename = "mask.png"
        with open("mask.png", "rb") as image_file:
            contents = "data:image/png;base64, " + \
                str(base64.b64encode(image_file.read()))[2:-1]
        return html.Div([
            html.Img(src=contents),
            # html.Hr(),
            # html.P("Filename: " + filename),
            # html.Div('Raw Content'),
            # html.Pre(contents[0:50] + '...', style={
            #     'whiteSpace': 'pre-wrap',
            #     'wordBreak': 'break-all'
            # })
        ])
    return ''


@app.callback(
    dash.dependencies.Output('graph', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')], State('upload-image', 'contents'))
def update_output1(n_clicks, list_of_contents):
    if n_clicks:
        contents = list_of_contents
        data = contents[contents.find(','):-1]+'=='
        imgdata = np.array(Image.open(io.BytesIO(base64.b64decode(data))))
        shape = 256
        imgdata = resize(imgdata, (shape, shape))
        fig = px.imshow(imgdata)
        resp = requests.post("https://vectorization-server.herokuapp.com/tosvg",
                             files={"file": open('mask.png', 'rb')})
        json_load = resp.json()
        a_restored = np.asarray(json_load["output"])
        root = ET.fromstring("""<?xml version="1.0"?>"""+str(a_restored))
        fig.add_shape(editable=True, type="path",
                      path=root.attrib["d"]+" Z",
                      line_color="SkyBlue", line_width=5)

        fig.update_layout(
            dragmode='drawrect',  # define dragmode
            newshape=dict(line_color='cyan'))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20), width=900, height=500)
        return [dcc.Graph(figure=fig, config={'modeBarButtonsToAdd': ['drawline',
                                                                      'drawopenpath',
                                                                      'drawclosedpath',
                                                                      'drawcircle',
                                                                      'drawrect',
                                                                      'eraseshape'
                                                                      ]})]
    return ''


if __name__ == '__main__':
    app.run_server(debug=True)
