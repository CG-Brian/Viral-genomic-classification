# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
import numpy as np
import pickle

from ML.inference import infer, load_for_inference

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "virific"
server = app.server

model_path = 'ML/assets/sgrnn_emb_ftrue.pth'
model, tokenizer, label_dict = load_for_inference(model_path,
                                                  "ML/assets/gene_tokenizer.json",
                                                  "ML/assets/label_dict.json",
                                                  embedding_dim=256,
                                                  hidden_dim=512,
                                                  num_layers=1)

with open('ML/assets/virus_pca.pkl', 'rb') as f:
    pca = pickle.load(f)

virus_embeddings = np.load('ML/assets/virus_embeddings_3d.npy')
with open('ML/assets/label.pkl', 'rb') as f:
    label = pickle.load(f)

label_index = {k: [] for k in set(label)}

for i, l in enumerate(label):
    label_index[l].append(i)

label_index = {k: np.random.choice(v, 50) for k, v in label_index.items()}

sampled_index = np.concatenate(list(label_index.values()))

virus_embeddings = virus_embeddings[sampled_index]
label = [label[i] for i in sampled_index]

embedding_df = pd.DataFrame({
    'x': virus_embeddings[:, 0],
    'y': virus_embeddings[:, 1],
    'z': virus_embeddings[:, 2],
    'Virus Type': label,
})

df = pd.DataFrame({
    "Virus Type": [x[0] for x in sorted(label_dict.items(), key=lambda x: x[1])],
})

app.layout = html.Div([
    html.H1("Viral Genome Classification"),
    html.Hr(),
    html.Div([
        html.H6("Upload your RNA Sequence."),
        html.H6("We will detect the virus for you."),
        html.A("How does this work", href="https://serious-cord-d5d.notion.site/Virus-Classification-Project-Report-f89b1f7d12a0401c8c1fce2a10117d83",
        style={'font-style':'italic'}),
    ]),
    html.Hr(),
    "Type your RNA Sequence: ",
    html.Div(dcc.Textarea(id='my-input', value='', placeholder="ACTG...")),
    html.Button("Go!", id='go-button'),
    html.Br(),
    html.Div(id='my-output'),
],
                      style={'text-align': 'center'})


def update_embedding_viz(input_value):
    emb = infer(input_value, tokenizer, model.embedding).numpy()
    emb_3d = pca.transform(emb)

    emb_df = pd.DataFrame({
        'x': emb_3d[:, 0],
        'y': emb_3d[:, 1],
        'z': emb_3d[:, 2],
        'Virus Type': "NEW"
    })

    plot_df = pd.concat([embedding_df, emb_df])

    new_ind = plot_df['Virus Type'] == "NEW"
    size = np.where(new_ind, 2, 0.5)
    fig = px.scatter_3d(plot_df,
                        x='x',
                        y='y',
                        z='z',
                        color='Virus Type',
                        size=size,
                        symbol=new_ind.astype(int),
                        symbol_sequence=['circle', 'square'],
                        opacity=0.8,
                        title="Virus Feautres 3D visualization")
    return dcc.Graph(id='3d-viz', figure=fig)


@app.callback(Output(component_id='my-output', component_property='children'),
              State(component_id='my-input', component_property='value'),
              Input(component_id='go-button', component_property='n_clicks'))
def update_output_div(input_value, n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    input_value = input_value.strip().upper()

    if set(input_value) != set('ACTG'):
        return html.P("Please input valid sequence containing only A, C, T, G",
                      style={'color': 'red'})

    probability = infer(input_value, tokenizer, model)
    df['Probability'] = probability
    fig = px.bar(df,
                 x="Virus Type",
                 y="Probability",
                 color="Virus Type",
                 title="Classification Result")
    return [dcc.Graph(id='example-graph', figure=fig), update_embedding_viz(input_value)]


if __name__ == '__main__':
    app.run_server()
