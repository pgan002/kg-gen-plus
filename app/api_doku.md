# KG Explorer App

This application provides two main functionalities:

1.  **Web APIs (at `/api`)**: A set of endpoints to generate graphs from text, and to parse and convert ontologies.
2.  **Graphical User UI (at `/ui`)**: A user interface to visually investigate the generated graphs.

## How to use the application

### 1. Generate a graph

Use the web APIs at `/api` to generate a graph from your text or ontology.

### 2. Visualize the graph

To investigate a graph visually, you first have to generate a graph and then call `/ui/add_graph` to add the graph to the visualizer.

After adding the graph, go to the `/ui` endpoint. In the user interface, you can open the graph by using the "Open Existing Graph" functionality.
