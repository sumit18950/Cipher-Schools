import pandas as pd
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc,html
from dash.dependencies import Input, Output
import random


#Load dataset
data=pd.read_excel('training dataset.xlsx')

#data preprocessing
nltk.download('punkt')
data['Concept']=data['Concept'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

#split
x_train,x_test,y_train,y_test= train_test_split(data['Concept'], data['Description'], test_size=0.2, random_state=42)

#create model pipeline
model= make_pipeline(TfidfVectorizer(),MultinomialNB() )
model.fit(x_train,y_train)

def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    answer = model.predict([question])[0]
    return answer

#INITIALISE DASH APP
app=dash.Dash(__name__)

#define layout
app.layout = html.Div([
    html.H1("ChatBot", style={'textAlign': 'center'}),
    dcc.Textarea(id='user-input', value='Ask something...', style={'width': '100%', 'height': 100}),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='chatbot-output', style={'padding': '10px'})
])





#callback to update chatbot response

@app.callback(
    Output('chatbot-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('user-input', 'value')]
)

def update_output(n_clicks, user_input):
    if n_clicks > 0:
        response=get_response(user_input)
        return html.Div([
            html.P(f"You: {user_input}", style={'margin': '10px'}),
            html.P(f"ChatBot: {response}",style={'margin':'10px', 'backgroundColor': 'beige', 'padding': '10px'})
        ])
    
    return "Ask me something!"


#RUN THE APP
if __name__ == '__main__':
     app.run_server(debug=True)