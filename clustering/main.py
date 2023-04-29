import itertools
import pandas
import numpy
import random
import os
import pickle


def generate_point(unlabeled_data: pandas.DataFrame, i: int) -> float:
    min = numpy.min(unlabeled_data.iloc[:, i])
    max = numpy.max(unlabeled_data.iloc[:, i])
    return random.uniform(min, max)


def generate_centroid(unlabeled_data: pandas.DataFrame) -> "list[float]":
    return [generate_point(unlabeled_data, i) for i in range(len(unlabeled_data.columns))]


class DataClusteringResult:
    def __init__(self, centroids: numpy.ndarray, labels: numpy.ndarray):
        self.centroids = centroids
        self.labels = labels


def kmeans(unlabeled_data: pandas.DataFrame, num_clusters: int, max_iterations: int) -> DataClusteringResult:
    centroids = numpy.array([generate_centroid(unlabeled_data) for _ in range(num_clusters)])

    (num_samples, _) = unlabeled_data.shape

    labels = numpy.repeat(0, num_samples)
    values_buffer = numpy.zeros(centroids.shape)
    distances_buffer = numpy.zeros(num_clusters)

    for i in range(max_iterations):
        # Assign each point to the nearest centroid
        for j in range(num_samples):
            values_buffer[:] = unlabeled_data.iloc[j]
            values_buffer -= centroids
            values_buffer *= values_buffer
            numpy.sum(values_buffer, axis=1, out=distances_buffer)
            labels[j] = numpy.argmin(distances_buffer)

        # Update centroids based on the assigned points
        for k in range(len(centroids)):
            points = unlabeled_data[labels == k]
            if len(points) > 0:
                # Have to reallocate the result, because pandas does not implement the out parameter 
                centroids[k:] = numpy.mean(points, axis=0)

    return DataClusteringResult(centroids, labels)


def main():
    dataset_file_name = 'irises.csv'
    data = pandas.read_csv(dataset_file_name)
    label = data.columns[-1]
    unlabeled_data = data.drop(label, axis=1)
    max_iterations = 100

    # check if cached file exists, with results
    # if not, run kmeans and serialize KMeansResult to a binary file
    # else, load KMeansResult from the binary file
    file_name = f'{dataset_file_name}_{max_iterations}_cache'
    kmeans_result: DataClusteringResult = None

    try:
        if os.path.exists(file_name):
            with open(file_name, 'rb') as file:
                kmeans_result = pickle.load(file)
    except OSError:
        pass

    if kmeans_result is None:
        num_clusters = data[label].unique().shape[0]
        kmeans_result = kmeans(unlabeled_data, num_clusters, max_iterations)
        with open(file_name, 'wb') as file:
            pickle.dump(kmeans_result, file)

    run_graph_app(unlabeled_data, kmeans_result, dataset_file_name)


import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


def run_graph_app(unlabeled_data: pandas.DataFrame, kmeans_clusters: DataClusteringResult, dataset_name: str):
    # Set up Dash app
    app = dash.Dash(__name__)

    app_layout = [
        html.H1(f'Clusterization of the dataset {dataset_name}'),
    ]
    graph = dcc.Graph(id='scatter-plot')

    features = unlabeled_data.columns
    data_dict = {label: unlabeled_data[label] for label in unlabeled_data.columns}
    data_dict['color'] = kmeans_clusters.labels

    if len(features) == 2:
        fig = px.scatter(data_dict, x=features[0], y=features[1], color='color', height=500)
        fig.update_layout(title=f'Clusters')
        graph.figure = fig

    elif len(features) == 3:
        fig = px.scatter_3d(data_dict, x=features[0], y=features[1], z=features[2], color='color', height=500)
        fig.update_layout(title=f'Clusters')
        graph.figure = fig

    else:
        feature_combinations = [[features[i] for i in x] for x in itertools.combinations(range(4), 3)]
        figs_cache = [None] * len(feature_combinations)

        # Define dimensions dropdown
        dimensions_dropdown = dcc.Dropdown(
            id='dimensions-dropdown',
            options=[{'label': str.join(", ", x), 'value': i}
                     for i, x in enumerate(feature_combinations)],
            value=0
        )
        app_layout.append(html.Label('Select dimensions to plot:'))
        app_layout.append(dimensions_dropdown)

        # Define callback to update graph based on dimensions selection
        @app.callback(
            Output('scatter-plot', 'figure'),
            [Input('dimensions-dropdown', 'value')]
        )
        def update_graph(index: int):
            if figs_cache[index] is not None:
                return figs_cache[index]

            dimensions = feature_combinations[index]
            fig = px.scatter_3d(data_dict, x=dimensions[0], y=dimensions[1], z=dimensions[2], color='color', height=500)
            fig.update_layout(title=f'Clusters of the Dimensions {str.join(", ", dimensions)}')
            figs_cache[index] = fig
            return fig

    app_layout.append(graph)
    app.layout = html.Div(app_layout)
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
