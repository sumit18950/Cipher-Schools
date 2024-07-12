import dash
#defining layout
from dash import dcc,html
from dash.dependencies import Input, Output

app=dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("My Dash App", style={'textAlign': 'center'}),
#     dcc.Input(id='input-box', type='text', value='Type something...'),
#     html.Button('Submit', id='submit-button', n_clicks=0),
#     html.Div(id='output-box')
# ])
# #defining callback
# from dash.dependencies import Input, Output

# @app.callback(
#    Output('output-box','children') ,
#    Input('submit-button','n_clicks'),
#    [dash.dependencies.State('input-box','value')]


# )
# def update_output(n_clicks, value):
#     if n_clicks is not None:
#         return 'You have entered: {}'.format(value)
#     return ''

# if __name__ == '__main__':
#     app.run_server(debug=True)


    #creation of text area

app.layout = html.Div([
    html.H1("ChatBot", style={'textAlign': 'center'}),
    dcc.Textarea(id='user-input', value='Ask something...', style={'width': '100%', 'height': 100}),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='chatbot-output', style={'padding': '10px'})
])

# creating a chatbot response

@app.callback(
    Output('chatbot-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('user-input', 'value')]
)

def update_output(n_clicks, user_input):
    if n_clicks > 0:
        return html.Div([
            html.P(f"You: {user_input}", style={'margin': '10px'}),
            html.P("Bot: I am training now, ask something else.", style={'margin': '10px', 'backgroundColor': 'beige', 'padding': '10px'})
        ])
    
    return "Ask me something!"



if __name__ == '__main__':
     app.run_server(debug=True)