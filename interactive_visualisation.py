import networkx as nx
from tornado.ioloop import IOLoop
import numpy as np

from bokeh.models.widgets import Slider
from bokeh.models import (
    Plot,
    Range1d,
    MultiLine,
    Circle,
    HoverTool,
    TapTool,
    BoxSelectTool,
)
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import row, WidgetBox
from bokeh.server.server import Server

import pickle

io_loop = IOLoop.current()

with open("./data/network_data.pkl", "rb") as rfile:
    network_data_dict = pickle.load(rfile)
    # network_data_dict = [eval(k) for k in rfile]

unweighted_network_data = network_data_dict[0]["unweighted_network_data"]
network_edge_labels = network_data_dict[0]["network_edge_labels"]
network_node_labels = network_data_dict[0]["network_node_labels"]
number_agent_type = network_data_dict[0]["number_of_agents"]
max_time = len(number_agent_type)
# doc = output_file("bokeh/networkx_graph_demo.html")


def modify_network(doc):
    def make_dataset(time_iter):
        types = {"insurers": [], "reinsurers": [], "catbonds": []}
        unweighted_nx_network = nx.from_numpy_array(
            np.array(unweighted_network_data[time_iter])
        )
        nx.set_edge_attributes(
            unweighted_nx_network, network_edge_labels[time_iter], "categ"
        )
        nx.set_node_attributes(
            unweighted_nx_network, network_node_labels[time_iter], "id"
        )
        for i in range(number_agent_type[time_iter]["insurers"]):
            unweighted_nx_network.node[i]["type"] = "Insurer"
        for i in range(number_agent_type[time_iter]["reinsurers"]):
            unweighted_nx_network.node[i + number_agent_type[time_iter]["insurers"]][
                "type"
            ] = "Reinsurer"
        for i in range(number_agent_type[time_iter]["catbonds"]):
            unweighted_nx_network.node[
                i
                + number_agent_type[time_iter]["insurers"]
                + number_agent_type[time_iter]["reinsurers"]
            ]["type"] = "CatBond"
        nx.set_node_attributes(unweighted_nx_network, types, "type")
        return unweighted_nx_network

    def make_plot(unweighted_nx_network):
        plot = Plot(
            plot_width=600,
            plot_height=600,
            x_range=Range1d(-1.1, 1.1),
            y_range=Range1d(-1.1, 1.1),
        )
        plot.title.text = "Insurance Network Demo"
        graph_renderer = from_networkx(
            unweighted_nx_network, nx.kamada_kawai_layout, scale=1, center=(0, 0)
        )

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
        graph_renderer.node_renderer.selection_glyph = Circle(
            size=15, fill_color=Spectral4[2]
        )
        graph_renderer.node_renderer.hover_glyph = Circle(
            size=15, fill_color=Spectral4[1]
        )
        graph_renderer.node_renderer.glyph.properties_with_values()

        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="#CCCCCC", line_alpha=0.8, line_width=5
        )
        graph_renderer.edge_renderer.selection_glyph = MultiLine(
            line_color=Spectral4[2], line_width=5
        )
        graph_renderer.edge_renderer.hover_glyph = MultiLine(
            line_color=Spectral4[1], line_width=5
        )
        graph_renderer.edge_renderer.glyph.properties_with_values()

        graph_renderer.selection_policy = NodesAndLinkedEdges()
        node_hover_tool = HoverTool(tooltips=[("ID", "@id"), ("Type", "@type")])

        # graph_renderer.inspection_policy = EdgesAndLinkedNodes()
        # edge_hover_tool = HoverTool(tooltips=[('Category', '@categ')])

        plot.add_tools(node_hover_tool, TapTool(), BoxSelectTool())

        plot.renderers.append(graph_renderer)
        return plot

    def update(attr, old, new):
        doc.clear()
        network = make_dataset(0)
        if new > 0:
            new_network = make_dataset(new)
            network.update(new_network)

        timeselect_slider = Slider(
            start=0,
            end=max_time - 1,
            value=new,
            value_throttled=new,
            step=1,
            title="Time",
            callback_policy="mouseup",
        )
        timeselect_slider.on_change("value_throttled", update)

        p = make_plot(network)

        controls = WidgetBox(timeselect_slider)
        layout = row(controls, p)

        doc.add_root(layout)

    update("", old=-1, new=0)


network_app = Application(FunctionHandler(modify_network))

server = Server({"/": network_app}, io_loop=io_loop)
server.start()

if __name__ == "__main__":
    print("Opening Bokeh application on http://localhost:5006/")
    io_loop.add_callback(server.show, "/")
    io_loop.start()
