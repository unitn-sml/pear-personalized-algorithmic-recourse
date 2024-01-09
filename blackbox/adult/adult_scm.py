from recourse_fare.utils.wfare_utils.structural_weights import StructuralWeights

import numpy as np

import networkx as nx

DEFAULT_NODES = [
            "age",
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country"]

DEFAULT_EDGES_ADULT = [
                ("age", "workclass"),
                ("age", "education"),
                ("age", "relationship"),
                ("age", "occupation"),
                ("race", "occupation"),
                ("race", "education"),
                ("race", "relationship"),
                ("native_country", "occupation"),
                ("native_country", "relationship"),
                ("native_country", "education"),
                ("sex", "occupation"),
                ("sex", "education"),
                ("sex", "relationship"),
                ("relationship", "marital_status"),
                ("education", "workclass"),
                ("workclass", "occupation"),
                ("hours_per_week", "occupation"),
                ("workclass", "hours_per_week"),
                ("workclass", "capital_gain"),
                ("capital_gain", "capital_loss")
                ]

class AdultSCM(StructuralWeights):

    def __init__(self, preprocessor, remove_edges: list=None):
        self.preprocessor = preprocessor

        if remove_edges:
            EDGES = DEFAULT_EDGES_ADULT.copy()
            for e in remove_edges:
                EDGES.remove(e)
        else:
            EDGES = DEFAULT_EDGES_ADULT

        super().__init__(DEFAULT_NODES, EDGES)

    def _feature_mapping(self, features: dict) -> dict:
        """This feature maps accomodate for binary features.
        """
        return self.preprocessor.transform_dict(features, type="raw")

    def compute_cost(self, node: str, new_value, features: dict, weights: dict) -> float:
        """
        Compute the cost of changing one feature given
        :param name:
        :param new_value:
        :param features:
        :return:
        """
        
        # Apply feature mapping to all the features and to the new value
        features_tmp = features.copy()
        features_tmp[node] = new_value

        features = self._feature_mapping(features)
        new_features = self._feature_mapping(features_tmp)

        if node in self.preprocessor.categorical:
            old_value_encoded = 0
            new_value_encoded = 1
        elif node in self.preprocessor.continuous:
            new_value_encoded = new_features.get(node)
            old_value_encoded = features.get(node)
        else:
            print("Missing feature in the graph!")   

        # Compute the cost given the parents
        cost = 0
        parent_edges = self.scm.predecessors(node)
        for parent in parent_edges:
            
            if parent in self.preprocessor.categorical:
                parent_value = 1
            else:
                parent_value = features.get(parent)
            
            cost += weights.get((parent, node))*(1+parent_value)
        
        # Check the variation
        variation = np.abs(new_value_encoded - old_value_encoded)
        variation = 1 if variation == 0 else variation

        # Return the cost plus the variation of the current value
        # If the clip the cost to be positive. A negative cost does not make sense.
        assert variation != 0, (new_value_encoded, old_value_encoded, node)
        return max(1, cost + variation * weights.get((node, node)))

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['mathtext.fontset'] = 'cm'
    sns.set(rc={'figure.figsize':(8,5)})
    sns.set(font_scale=1.5)
    sns.set_style("ticks", {'axes.grid' : False})
    sns.set_palette(sns.color_palette("colorblind"))
    plt.rcParams['lines.markersize'] = 12

    plt.rcParams.update({
        "text.usetex": True
    })

    DRAWING_MAPPING = {
        "age": "AGE",
        "workclass": "WO",
        "education": "EDU",
        "marital_status": "MA",
        "occupation": "OCC",
        "relationship": "REL",
        "race": "R",
        "sex": "SEX",
        "capital_gain": "CA_G",
        "capital_loss": "CA_L",
        "hours_per_week": "HO",
        "native_country": "NA"
    }

    scm = AdultSCM(None)
    renamed_graph = nx.relabel_nodes(scm.scm, DRAWING_MAPPING)
    nx.draw_networkx(renamed_graph,
                     with_labels=True,
                     node_color="white",
                     edgecolors="black",
                     font_size=20,
                     arrowsize=20,
                     node_size=2500, pos = nx.circular_layout(renamed_graph))
    sns.despine(bottom = True, left = True)
    plt.tight_layout()
    plt.savefig("adult_cg.png", dpi=300)


