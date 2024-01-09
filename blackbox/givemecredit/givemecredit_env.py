from recourse_fare.environment_w import EnvironmentWeights

from blackbox.givemecredit.givemecredit_scm import GiveMeCreditSCM

from collections import OrderedDict
from typing import Any

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class GiveMeCreditEnv(EnvironmentWeights):

    def __init__(self, features: dict, weights: dict, model: Any,
                 preprocessor: Any,
                 remove_edges: list=None,
                 agnostic: bool=False,
                 model_type="svc"):
        
        self.model_type = model_type
        self.agnostic = agnostic

        # Preprocessor element. Please have a look below to understand how it is called.
        self.preprocessor = preprocessor

        # The maximum length of an intervention. It considers also the STOP action.
        self.max_intervention_depth = 6

        # Dictionary specifying, for each action, the corresponding implementation of
        # such action in the environment.
        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                "BALANCE": self._change_balance,
                                                "PAST_DUE_30": self._change_past_due_30,
                                                "DEBT_RATIO": self._change_debt_ratio,
                                                "INCOME": self._change_income,
                                                "OPEN_LOANS": self._change_open_loans,
                                                "PAST_DUE_90": self._change_past_due_90,
                                                "OPEN_REAL_ESTATE": self._change_open_real_estate,
                                                "PAST_DUE_60": self._change_past_due_60,
                                                }.items()))
        
        # Dictionary specifying, for each action, the corresponding precondition which needs 
        # to be verified to be able to apply such action. For example, if I already have a bachelor's
        # degree, the action "change_education(high-school diploma)" would be meaningless.
        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._placeholder_stop,
                                                        "BALANCE": self._change_balance_p,
                                                        "PAST_DUE_30": self._change_past_due_30_p,
                                                        "DEBT_RATIO": self._change_debt_ratio_p,
                                                        "INCOME": self._change_income_p,
                                                        "OPEN_LOANS": self._change_open_loans_p,
                                                        "PAST_DUE_90": self._change_past_due_90_p,
                                                        "OPEN_REAL_ESTATE": self._change_open_real_estate_p,
                                                        "PAST_DUE_60": self._change_past_due_60_p,
                                                        'INTERVENE': self._placeholder_stop
                                                        }.items()))

        # Function which validate the environment and it checks if we reached recourse.
        self.prog_to_postcondition = self._intervene_postcondition

        # Programs library. It contains all the available actions the user can perform. For each action, we need to
        # specify three things: the index (it goes from 0 to n), the level and the argument type that function accepts.
        #
        # Here we have two small caveats:
        #  * level: 1 represent the program we want to learn (INTERVENE), 0 represents the action we can take and
        # -1 represents the stop action, which is called to signal the termination of an intervention;
        # * The program library MUST contain the STOP and INTERVENE programs as defined below;
        self.programs_library = OrderedDict(sorted({'STOP': {'index': 0, 'level': -1, 'args': 'NONE'},
                                                    "BALANCE": {'index': 1, 'level': 0, 'args': 'PERC'},
                                                    "PAST_DUE_30": {'index': 2, 'level': 0, 'args': 'DAYS_LATE'},
                                                    "DEBT_RATIO": {'index': 3, 'level': 0, 'args': 'PERC'},
                                                    "INCOME": {'index': 4,'level': 0, 'args': 'INCOME'},
                                                    "OPEN_LOANS": {'index': 5,'level': 0, 'args': 'OPEN_LOANS_REAL_ESTATE'},
                                                    "PAST_DUE_90": {'index': 6,'level': 0, 'args': 'DAYS_LATE'},
                                                    "OPEN_REAL_ESTATE": {'index': 7,'level': 0, 'args': 'OPEN_LOANS_REAL_ESTATE'},
                                                    "PAST_DUE_60": {'index': 8,'level': 0, 'args': 'DAYS_LATE'},
                                                    'INTERVENE': {'index': 9,'level': 1, 'args': 'NONE'}}.items()))

        # The available arguments. For each type, we need to specify a list of potential values. Each action will be
        # tied to the correspoding type. 
        # The arguments need to contain the NONE type, with a single value 0.
        self.arguments = OrderedDict(sorted({
                                                #"PERC": [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9 -0.1, -0.2, -0.3, -0.5, -0.7, -0.8, -0.9],
                                                "PERC": list(np.linspace(0.01, 1, num=20))+list(-np.linspace(0.01, 1.0, num=20)),
                                                "DAYS_LATE": [-i for i in range(1,25)],
                                                "OPEN_LOANS_REAL_ESTATE": [-i for i in range(1,25)],
                                                "INCOME": list(np.linspace(1, 10000, num=25))+list(-np.linspace(1, 10000, num=25)),
                                                "NONE": [0]
                                            }.items()))

        self.program_feature_mapping = {
            "BALANCE": 'RevolvingUtilizationOfUnsecuredLines',
            "PAST_DUE_30": 'NumberOfTime30-59DaysPastDueNotWorse',
            "DEBT_RATIO": 'DebtRatio',
            "INCOME": 'MonthlyIncome',
            "OPEN_LOANS": 'NumberOfOpenCreditLinesAndLoans',
            "PAST_DUE_60": 'NumberOfTime60-89DaysPastDueNotWorse',
            "OPEN_REAL_ESTATE": 'NumberRealEstateLoansOrLines',
            "PAST_DUE_90": 'NumberOfTimes90DaysLate',
        }

        # Create the baseline graph and corrupt it if needed.
        scm = GiveMeCreditSCM(preprocessor, remove_edges)

        # Call parent constructor
        super().__init__(features, weights, scm, model, self.prog_to_func, self.prog_to_precondition, self.prog_to_postcondition,
                         self.programs_library, self.arguments, self.max_intervention_depth, prog_to_cost=None,
                         program_feature_mapping=self.program_feature_mapping
                         )

    # Some utilites functions

    def reset_to_state(self, state):
        self.features = state.copy()

    def get_stop_action_index(self):
        return self.programs_library["STOP"]["index"]

    ### ACTIONS

    def _stop(self, arguments=None):
       return True

    def _change_balance(self, arguments=None):
        self.features["RevolvingUtilizationOfUnsecuredLines"] += arguments
    
    def _change_past_due_30(self, arguments=None):
        self.features["NumberOfTime30-59DaysPastDueNotWorse"] += arguments
    
    def _change_debt_ratio(self, arguments=None):
        self.features["DebtRatio"] += arguments
    
    def _change_income(self, arguments=None):
        self.features["MonthlyIncome"] += arguments
    
    def _change_open_loans(self, arguments=None):
        self.features["NumberOfOpenCreditLinesAndLoans"] += arguments
    
    def _change_past_due_90(self, arguments=None):
        self.features["NumberOfTimes90DaysLate"] += arguments
    
    def _change_open_real_estate(self, arguments=None):
        self.features["NumberRealEstateLoansOrLines"] += arguments
    
    def _change_past_due_60(self, arguments=None):
        self.features["NumberOfTime60-89DaysPastDueNotWorse"] += arguments

    ### ACTIONA PRECONDTIONS

    def _change_balance_p(self, arguments=None):
        return self.features.get('RevolvingUtilizationOfUnsecuredLines')+arguments >= 0.0 and self.features.get('RevolvingUtilizationOfUnsecuredLines')+arguments <= 1.0
    
    def _change_past_due_30_p(self, arguments=None):
        return self.features.get("NumberOfTime30-59DaysPastDueNotWorse")+arguments >= 0.0
    
    def _change_debt_ratio_p(self, arguments=None):
        return self.features.get('DebtRatio')+arguments >= 0.0
    
    def _change_income_p(self, arguments=None):
        return self.features.get("MonthlyIncome")+arguments >= 0.0
    
    def _change_open_loans_p(self, arguments=None):
        return self.features.get("NumberOfOpenCreditLinesAndLoans")+arguments >= 0.0
    
    def _change_past_due_90_p(self, arguments=None):
        return self.features.get("NumberOfTimes90DaysLate")+arguments >= 0.0
    
    def _change_open_real_estate_p(self, arguments=None):
        return self.features.get('NumberRealEstateLoansOrLines')+arguments >= 0.0
    
    def _change_past_due_60_p(self, arguments=None):
        return self.features.get("NumberOfTime60-89DaysPastDueNotWorse")+arguments >= 0.0

    def _placeholder_stop(self, args=None):
        return True

    ### POSTCONDITIONS

    def _intervene_postcondition(self, init_state, current_state):
        # We basically check if the model predicts a 0 (which means
        # recourse) given the current features. We discard the 
        # init_state
        obs = self.preprocessor.transform(
            pd.DataFrame.from_records(
                [current_state]
            )
        )
        if self.model_type != "nn":
            return self.model.predict(obs)[0] == 1
        else:
            with torch.no_grad():
                self.model.eval()
                output = self.model(torch.FloatTensor(obs)).round().numpy()
                return output[0][0] == 1

    ## OBSERVATIONS
    def get_observation(self):
        obs = self.preprocessor.transform_dict(self.features)
        if self.agnostic:
             return torch.FloatTensor(obs)
        else:
            costs = self.get_list_of_costs()
            return torch.FloatTensor(np.concatenate([obs, costs]))
    
    ## FASTER LIST OF COSTS
    def get_list_of_costs(self):

        lists_of_costs = []

        # For each available program and argument, compute the cost
        for program in self.programs_library:

            if program == "INTERVENE" or program == "STOP":
                continue

            all_available_args = self.arguments.get(self.programs_library.get(program).get("args"))
            
            current_average_cost = []
            available_args = []

            prog_idx = self.prog_to_idx.get(program)

            available_args = [
              (self.inverse_complete_arguments.get(argument).get(program, None), argument) for argument in all_available_args if self.can_be_called(prog_idx, self.inverse_complete_arguments.get(argument).get(program, None))
            ]
            
            if len(available_args) > 0:
                if isinstance(available_args[0][1], str):
                    current_average_cost.append(self.get_cost(prog_idx, available_args[0][0]))
                else:
                    available_args = sorted(available_args, key=lambda x: x[1])
                    current_average_cost.append(self.get_cost(prog_idx, available_args[0][0]))
                    current_average_cost.append(self.get_cost(prog_idx, available_args[-1][0]))

            lists_of_costs.append(np.mean(current_average_cost) if len(current_average_cost) > 0 else -1)
        
        # Standardize the costs
        lists_of_costs = np.array(lists_of_costs)
        mask_not_available = np.where(lists_of_costs >= 0, 1, 0)
        mask_available = np.where(lists_of_costs < 0, 1, 0)

        max_costs = lists_of_costs.max()
        
        lists_of_costs = mask_not_available*lists_of_costs + mask_available*lists_of_costs.max()
        
        lists_of_costs = -lists_of_costs

        lists_of_costs = lists_of_costs - max_costs
        lists_of_costs = np.exp(lists_of_costs*0.1)
        lists_of_costs = lists_of_costs / lists_of_costs.sum()

        return lists_of_costs