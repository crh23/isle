import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class ReinsuranceNetwork:
    def __init__(self):
        """Initialising method for ReinsuranceNetwork.
            No accepted values.
        This created the figure that the network will be displayed on so only called once, and only if show_network is
        True."""
        self.figure = plt.figure(
            num=None, figsize=(10, 8), dpi=100, facecolor="w", edgecolor="k"
        )

    def compute_measures(self):
        """Method to obtain the network distribution and print it.
            No accepted values.
            No return values."""
        # degrees = self.network.degree()
        degree_distr = dict(self.network.degree()).values()
        in_degree_distr = dict(self.network.in_degree()).values()
        out_degree_distr = dict(self.network.out_degree()).values()
        is_connected = nx.is_weakly_connected(self.network)
        # is_connected = nx.is_strongly_connected(self.network)  # must always be False
        try:
            node_centralities = nx.eigenvector_centrality(self.network)
        except:
            node_centralities = nx.betweenness_centrality(self.network)
        # TODO: and more, choose more meaningful ones...

        print(
            "Graph is connected: ",
            is_connected,
            "\nIn degrees ",
            in_degree_distr,
            "\nOut degrees",
            out_degree_distr,
            "\nCentralities",
            node_centralities,
        )

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
        for firmtype, firmlist in [
            ("insurers", self.insurancefirms),
            ("reinsurers", self.reinsurancefirms),
            ("catbonds", self.catbonds),
        ]:
            op_firmtype = [firm for firm in firmlist if firm.operational]
            op_entities[firmtype] = op_firmtype
            self.num_entities[firmtype] = len(op_firmtype)

        self.network_size = sum(self.num_entities.values())

        """Create weighted adjacency matrix and category edge labels"""
        weights_matrix = np.zeros(self.network_size ** 2).reshape(
            self.network_size, self.network_size
        )
        self.edge_labels = {}
        for idx_to, firm in enumerate(
            op_entities["insurers"] + op_entities["reinsurers"]
        ):
            eolrs = firm.get_excess_of_loss_reinsurance()
            for eolr in eolrs:
                # pdb.set_trace()
                try:
                    idx_from = self.num_entities["insurers"] + (
                        op_entities["reinsurers"] + op_entities["catbonds"]
                    ).index(eolr["reinsurer"])
                    weights_matrix[idx_from][idx_to] = eolr["value"]
                    self.edge_labels[idx_to, idx_from] = eolr["category"]
                except ValueError:
                    print("Reinsurer is not in list of reinsurance companies")

        """unweighted adjacency matrix"""
        adj_matrix = np.sign(weights_matrix)

        """define network"""
        self.network = nx.from_numpy_array(
            weights_matrix, create_using=nx.DiGraph()
        )  # weighted
        self.network_unweighted = nx.from_numpy_array(
            adj_matrix, create_using=nx.DiGraph()
        )  # unweighted

    def visualize(self):
        """Method to add the network to the figure initialised in __init__.
            No accepted values.
            No return values.
        This method takes the network created in update method and then draws it onto the figure with edge labels
        corresponding to the category being reinsured, and adds a legend to indicate which node is insurer, reinsurer,
        or CatBond. This method allows the figure to be updated without a new figure being created or stopping the
        program."""
        plt.ion()  # Turns on interactive graph mode.
        firmtypes = np.ones(self.network_size)
        firmtypes[
            self.num_entities["insurers"] : self.num_entities["insurers"]
            + self.num_entities["reinsurers"]
        ] = 0.5
        firmtypes[
            self.num_entities["insurers"] + self.num_entities["reinsurers"] :
        ] = 1.3
        print(
            "Number of insurers: %i, Number of Reinsurers: %i, CatBonds: %i"
            % (
                self.num_entities["insurers"],
                self.num_entities["reinsurers"],
                self.num_entities["catbonds"],
            )
        )

        # Either this or below create a network, this one has id's but no key.
        # pos = nx.spring_layout(self.network_unweighted)
        # nx.draw(self.network_unweighted, pos, node_color=firmtypes, with_labels=True, cmap=plt.cm.plasma)
        # nx.draw_networkx_edge_labels(self.network_unweighted, pos, self.edge_labels, font_size=5)

        "Draw Network"
        pos = nx.spring_layout(self.network_unweighted)
        nx.draw_networkx_nodes(
            self.network_unweighted,
            pos,
            list(range(self.num_entities["insurers"])),
            node_color="b",
            node_size=50,
            alpha=0.9,
            label="Insurer",
        )
        nx.draw_networkx_nodes(
            self.network_unweighted,
            pos,
            list(
                range(
                    self.num_entities["insurers"],
                    self.num_entities["insurers"] + self.num_entities["reinsurers"],
                )
            ),
            node_color="r",
            node_size=50,
            alpha=0.9,
            label="Reinsurer",
        )
        nx.draw_networkx_nodes(
            self.network_unweighted,
            pos,
            list(
                range(
                    self.num_entities["insurers"] + self.num_entities["reinsurers"],
                    self.num_entities["insurers"]
                    + self.num_entities["reinsurers"]
                    + self.num_entities["catbonds"],
                )
            ),
            node_color="g",
            node_size=50,
            alpha=0.9,
            label="CatBond",
        )
        nx.draw_networkx_edges(
            self.network_unweighted, pos, width=1.0, alpha=0.5, node_size=50
        )
        nx.draw_networkx_edge_labels(
            self.network_unweighted, pos, self.edge_labels, font_size=5
        )
        plt.legend(scatterpoints=1, loc="upper right")
        plt.axis("off")
        plt.show()

        """Update figure"""
        self.figure.canvas.flush_events()
        self.figure.clear()
