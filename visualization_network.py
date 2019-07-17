import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse


class ReinsuranceNetwork:
    def __init__(self, event_schedule=None):
        """Initialising method for ReinsuranceNetwork.
            No accepted values.
        This created the figure that the network will be displayed on so only called once, and only if show_network is
        True."""
        self.figure = plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
        self.save_data = {'unweighted_network': [], 'weighted_network': [], 'network_edgelabels': [], 'network_node_labels': [], 'number_of_agents': []}
        self.event_schedule = event_schedule

    def compute_measures(self):
        """Method to obtain the network distribution and print it.
            No accepted values.
            No return values."""
        #degrees = self.network.degree()
        degree_distr = dict(self.network.degree()).values()
        in_degree_distr = dict(self.network.in_degree()).values()
        out_degree_distr = dict(self.network.out_degree()).values()
        is_connected = nx.is_weakly_connected(self.network)
        #is_connected = nx.is_strongly_connected(self.network)  # must always be False
        try:
            node_centralities = nx.eigenvector_centrality(self.network)
        except:
            node_centralities = nx.betweenness_centrality(self.network)
        # TODO: and more, choose more meaningful ones...
        
        print("Graph is connected: ", is_connected, "\nIn degrees ", in_degree_distr, "\nOut degrees", out_degree_distr, \
              "\nCentralities", node_centralities)

    def update(self, insurancefirms, reinsurancefirms, catbonds):
        """Method to update the network.
            Accepts:
                insurancefirms: Type List of DataDicts.
                resinurancefirn.Type List of DataDicts.
                catbonds: Type List of DataDicts.
            No return values.
        This method is called from insurancesimulation for every iteration a network is to be shown. It takes the list
        of agents and creates both a weighted and unweighted networkx network with it."""
        self.insurancefirms = insurancefirms
        self.reinsurancefirms = reinsurancefirms
        self.catbonds = catbonds

        """obtain lists of operational entities"""
        op_entities = {}
        self.num_entities = {}
        for firmtype, firmlist in [("insurers", self.insurancefirms), ("reinsurers", self.reinsurancefirms), ("catbonds", self.catbonds)]:
            op_firmtype = [firm for firm in firmlist if firm.operational]
            op_entities[firmtype] = op_firmtype
            self.num_entities[firmtype] = len(op_firmtype)

        self.network_size = sum(self.num_entities.values())

        """Create weighted adjacency matrix and category edge labels"""
        weights_matrix = np.zeros(self.network_size ** 2).reshape(self.network_size, self.network_size)
        self.edge_labels = {}
        self.node_labels = {}
        for idx_to, firm in enumerate(op_entities["insurers"] + op_entities["reinsurers"]):
            self.node_labels[idx_to] = firm.id
            eolrs = firm.get_excess_of_loss_reinsurance()
            for eolr in eolrs:
                try:
                    idx_from = self.num_entities["insurers"] + (op_entities["reinsurers"] + op_entities["catbonds"]).index(eolr["reinsurer"])
                    weights_matrix[idx_from][idx_to] = eolr["value"]
                    self.edge_labels[idx_to, idx_from] = eolr["category"]
                except ValueError:
                    print("Reinsurer is not in list of reinsurance companies")

        """unweighted adjacency matrix"""
        adj_matrix = np.sign(weights_matrix)

        """define network"""
        self.network = nx.from_numpy_array(weights_matrix, create_using=nx.DiGraph())  # weighted
        self.network_unweighted = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())  # unweighted

        """Add this iteration of network data to be saved"""
        self.save_data['unweighted_network'].append(adj_matrix.tolist())
        self.save_data['weighted_network'].append(weights_matrix.tolist())
        self.save_data['network_edgelabels'].append(self.edge_labels)
        self.save_data['network_node_labels'].append(self.node_labels)
        self.save_data['number_of_agents'].append(self.num_entities)

    def visualize(self):
        """Method to add the network to the figure initialised in __init__.
            No accepted values.
            No return values.
        This method takes the network created in update method and then draws it onto the figure with edge labels
        corresponding to the category being reinsured, and adds a legend to indicate which node is insurer, reinsurer,
        or CatBond. This method allows the figure to be updated without a new figure being created or stopping the
        program."""
        plt.ion()   # Turns on interactive graph mode.
        firmtypes = np.ones(self.network_size)
        firmtypes[self.num_entities["insurers"]:self.num_entities["insurers"]+self.num_entities["reinsurers"]] = 0.5
        firmtypes[self.num_entities["insurers"]+self.num_entities["reinsurers"]:] = 1.3
        print("Number of insurers: %i, Number of Reinsurers: %i, CatBonds: %i"
              % (self.num_entities["insurers"], self.num_entities["reinsurers"], self.num_entities['catbonds']))

        # Either this or below create a network, this one has id's but no key.
        # pos = nx.spring_layout(self.network_unweighted)
        # nx.draw(self.network_unweighted, pos, node_color=firmtypes, with_labels=True, cmap=plt.cm.plasma)
        # nx.draw_networkx_edge_labels(self.network_unweighted, pos, self.edge_labels, font_size=5)

        "Draw Network"
        pos = nx.spring_layout(self.network_unweighted)
        nx.draw_networkx_nodes(self.network_unweighted, pos, list(range(self.num_entities["insurers"])),
                               node_color='b', node_size=50, alpha=0.9, label='Insurer')
        nx.draw_networkx_nodes(self.network_unweighted, pos, list(range(self.num_entities["insurers"], self.num_entities["insurers"]+self.num_entities["reinsurers"])),
                               node_color='r', node_size=50, alpha=0.9, label='Reinsurer')
        nx.draw_networkx_nodes(self.network_unweighted, pos, list(range(self.num_entities["insurers"] + self.num_entities["reinsurers"], self.num_entities["insurers"] + self.num_entities["reinsurers"] + self.num_entities['catbonds'])),
                               node_color='g', node_size=50, alpha=0.9, label='CatBond')
        nx.draw_networkx_edges(self.network_unweighted, pos, width=1.0, alpha=0.5, node_size=50)
        nx.draw_networkx_edge_labels(self.network_unweighted, pos, self.edge_labels, font_size=5)
        nx.draw_networkx_labels(self.network_unweighted, pos, self.node_labels, font_size=20)
        plt.legend(scatterpoints=1, loc='upper right')
        plt.axis('off')
        plt.show()

        """Update figure"""
        self.figure.canvas.flush_events()
        self.figure.clear()

    def save_network_data(self):
        with open("data/network_data.dat", "w") as wfile:
            wfile.write(str(self.save_data) + "\n")
            wfile.write(str(self.event_schedule) + "\n")


class LoadNetwork:
    def __init__(self, network_data, num_iter):
        """Initialises LoadNetwork class.
            Accepts:
                network_data: Type List. Contains a DataDict of the network data, and a list of events.
                num_iter: Type Integer. Used to tell animation how many frames it should have.
            No return values.
        This class is given the loaded network data and then uses it to create an animated network."""
        self.figure = plt.figure(num=None, figsize=(10, 8), dpi=100, facecolor='w', edgecolor='k')
        self.unweighted_network_data = network_data[0]["unweighted_network"]
        # self.weighted_network_data = network_data[0]["weighted_network"]           # Unused for now
        self.network_edge_labels = network_data[0]["network_edgelabels"]
        self.network_node_labels = network_data[0]["network_node_labels"]
        self.number_agent_type = network_data[0]["number_of_agents"]
        self.event_schedule = network_data[1]
        self.num_iter = num_iter

        self.all_events = []
        for categ in self.event_schedule:
            self.all_events += categ
        self.all_events.sort()

    def update(self, i):
        """Method to update network animation.
            Accepts:
                i: Type Integer, iterator.
            No return values.
        This method is called from matplotlib.animate.FuncAnimation to update the plot to the next time iteration."""
        self.figure.clear()
        plt.suptitle('Network Timestep %i' % i)
        unweighted_nx_network = nx.from_numpy_array(np.array(self.unweighted_network_data[i]))
        pos = nx.shell_layout(unweighted_nx_network)

        nx.draw_networkx_nodes(unweighted_nx_network, pos, list(range(self.number_agent_type[i]["insurers"])),
                               node_color='b', node_size=50, alpha=0.9, label='Insurer')
        nx.draw_networkx_nodes(unweighted_nx_network, pos, list(
            range(self.number_agent_type[i]["insurers"],
                  self.number_agent_type[i]["insurers"] + self.number_agent_type[i]["reinsurers"])),
                               node_color='r', node_size=50, alpha=0.9, label='Reinsurer')
        nx.draw_networkx_nodes(unweighted_nx_network, pos, list(
            range(self.number_agent_type[i]["insurers"] + self.number_agent_type[i]["reinsurers"],
                  self.number_agent_type[i]["insurers"] + self.number_agent_type[i]["reinsurers"] +
                  self.number_agent_type[i]['catbonds'])),
                               node_color='g', node_size=50, alpha=0.9, label='CatBond')
        nx.draw_networkx_edges(unweighted_nx_network, pos, width=1.0, alpha=0.5, node_size=50)

        nx.draw_networkx_edge_labels(self.unweighted_network_data[i], pos, self.network_edge_labels[i], font_size=5)
        nx.draw_networkx_labels(self.unweighted_network_data[i], pos, self.network_node_labels[i], font_size=10)

        while self.all_events[0] == i:
            plt.title('EVENT!')
            self.all_events = self.all_events[1:]

        plt.legend()
        plt.axis('off')

    def animate(self):
        """Method to create animation.
            No accepted values.
            No return values."""
        self.network_ani = animation.FuncAnimation(self.figure, self.update, frames=self.num_iter, repeat=False,
                                                   interval=20, save_count=self.num_iter)

    def save_network_animation(self):
        """Method to save animation as MP4.
            No accepted values.
            No return values."""
        self.network_ani.save("data/animated_network.mp4", writer="ffmpeg", dpi=200, fps=5)


if __name__ == "__main__":
    # Use argparse to handle command line arguments
    parser = argparse.ArgumentParser(description='Plot the network of the insurance sector')
    parser.add_argument("--save", action="store_true", help="Save the network as an mp4")
    parser.add_argument("--number_iterations", type=int, help="number of frames for animation")
    args = parser.parse_args()

    if args.number_iterations:
        num_iter = args.number_iterations
    else:
        num_iter = 100

    # Access stored network data
    with open("data/network_data.dat", "r") as rfile:
        network_data_dict = [eval(k) for k in rfile]

    # Load network data and create animation data for given number of iterations
    loaded_network = LoadNetwork(network_data_dict, num_iter=num_iter)
    loaded_network.animate()

    # Either display or save network, dependant on args
    if args.save:
        loaded_network.save_network_animation()
    else:
        plt.show()
