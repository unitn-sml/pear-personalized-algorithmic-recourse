from recourse_fare.utils.wfare_utils.structural_weights import StructuralWeights

import numpy as np

import networkx as nx

DEFAULT_NODES = [
            'RevolvingUtilizationOfUnsecuredLines',
            'age', 
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
            ]

DEFAULT_EDGES_GIVEMECREDIT = [
                ("age", "MonthlyIncome"),
                ("age", "DebtRatio"),
                ("age", "NumberOfDependents"),
                ("age", "NumberOfOpenCreditLinesAndLoans"),
                ("age", "NumberRealEstateLoansOrLines"),
                ("MonthlyIncome", "NumberOfOpenCreditLinesAndLoans"),
                ("MonthlyIncome", "NumberRealEstateLoansOrLines"),
                ("MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines"),
                ("RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse"),
                ("RevolvingUtilizationOfUnsecuredLines", "NumberOfTime60-89DaysPastDueNotWorse"),
                ("RevolvingUtilizationOfUnsecuredLines", "NumberOfTimes90DaysLate"),
                ("NumberOfOpenCreditLinesAndLoans", "NumberOfTime30-59DaysPastDueNotWorse"),
                ("NumberOfOpenCreditLinesAndLoans", "NumberOfTime60-89DaysPastDueNotWorse"),
                ("NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate"),
                ("NumberRealEstateLoansOrLines", "NumberOfTime30-59DaysPastDueNotWorse"),
                ("NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse"),
                ("NumberRealEstateLoansOrLines", "NumberOfTimes90DaysLate"),
                ("NumberOfOpenCreditLinesAndLoans", "DebtRatio"),
                ("NumberRealEstateLoansOrLines", "DebtRatio"),
                ]

class GiveMeCreditSCM(StructuralWeights):

    def __init__(self, preprocessor, remove_edges: list=None):
        self.preprocessor = preprocessor

        if remove_edges:
            EDGES = DEFAULT_EDGES_GIVEMECREDIT.copy()
            for e in remove_edges:
                EDGES.remove(e)
        else:
            EDGES = DEFAULT_EDGES_GIVEMECREDIT

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
        'RevolvingUtilizationOfUnsecuredLines': "ROU",
        'age': "AGE", 
        'NumberOfTime30-59DaysPastDueNotWorse': "30L",
        'DebtRatio': "DBR",
        'MonthlyIncome': "INC",
        'NumberOfOpenCreditLinesAndLoans': "CAL",
        'NumberOfTimes90DaysLate': "90L",
        'NumberRealEstateLoansOrLines': "NEL",
        'NumberOfTime60-89DaysPastDueNotWorse': "60L",
        'NumberOfDependents': "DP"
    }

    scm = GiveMeCreditSCM(None)
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
    plt.savefig("givemecredit_cg.png", dpi=300)


