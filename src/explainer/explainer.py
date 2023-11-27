import numpy as np
import pandas as pd
from explainer.searcher import Searcher
from explainer.single_variable_explainer import SingleVariableExplainer

class Explainer:
    def __init__(
        self,
        underlying_model,
        training_dataset,
        target_variable, 
        variables: list = None,
        sampling_method = 'uniform',       
        bounding_method = 'meanstd',
        std_dev = 1,
        number_samples = 10,
        regressor = 'gaussian_process',
        changeability_scores = None
    ):
        '''
            This class implements the process of chaining together single variable explainers to explain the output of the overall model. It uses the Searcher class to determine the order in which to evaluate the single variable explainers.

            More details are provided in our paper on this.

            *Required parameters*
            - underlying_model: the model to be explained
            - training_dataset: the dataset used to train the model
            - target_variable: the variable to be explained

            *Optional parameters*
            - variables: list of variables to search over
            - sampling_method: method to use for sampling the data (default: 'uniform'), see explainer.single_variable_explainer for more details
            - bounding_method: method to use for bounding the data (default: 'meanstd'), see explainer.single_variable_explainer for more details
            - std_dev: standard deviation to use for bounding the data (default: 1), see explainer.single_variable_explainer for more details
            - number_samples: number of samples to use for bounding the data (default: 10), see explainer.single_variable_explainer for more details
            - regressor: regressor to use for bounding the data (default: 'gaussian_process'), see explainer.single_variable_explainer for more details
            - changeability_scores: dictionary of changeability scores for each variable (default: None)

            "Methods"
            - explain: method to explain the output of the model at a given input point
        '''
        self.underlying_model = underlying_model
        self.training_dataset = training_dataset
        self.target_variable = target_variable
        self.variables = variables
        self.sampling_method = sampling_method
        self.bounding_method = bounding_method
        self.std_dev = std_dev
        self.number_samples = number_samples
        self.regressor = regressor

        if changeability_scores is None:
            self.changeability_scores = {
                col: 1
                for col in variables
            } 
        


    def explain(
        self, 
        explanation_point,
        initial_classification: int = None,
        additional_threshold = 0
    ):
        '''
            This method explains the output of the model at a given input point.

            *Required parameters*
            - explanation_point: the point to be explained

            *Optional parameters*
            - initial_classification: the initial classification of the point as an integer (default: None)
            - additional_threshold: additional threshold to use for determining the solution (default: 0)
        '''
        if initial_classification is None:
            initial_classification = self.underlying_model.predict(pd.DataFrame(explanation_point[self.underlying_model.feature_names_in_]).T)
        searcher = Searcher(self.variables)
        evaluation_points_dict = {tuple(): explanation_point}

        initial_point = pd.DataFrame(explanation_point).T[self.underlying_model.feature_names_in_]
        initial_point.columns = self.underlying_model.feature_names_in_

        extrema_dict = {tuple(): self.underlying_model.predict_proba(initial_point)[0][1]}
        previous_scores = {tuple(): 0}

        found = False
        try: 
            while not found:
                current_node = searcher.current_node
                evaluation_nodes = searcher.get_next_evaluation_set()
                evaluation_point = evaluation_points_dict[current_node]

        
                single_variable_explainers = {
                    node: SingleVariableExplainer(
                        underlying_model = self.underlying_model,
                        target_variable = self.target_variable,
                        explainable_variable = node[-1],
                        explanation_point = evaluation_point,
                        training_dataset = self.training_dataset,
                        sampling_method = self.sampling_method,
                        bounding_method = self.bounding_method,
                        std_dev = self.std_dev,
                        number_samples = self.number_samples,
                        regressor = self.regressor
                    )
                    for node in evaluation_nodes                
                }
                
                arg_extrema = {node: single_variable_explainers[node].get_arg_extrema(initial_classification=initial_classification) for node in evaluation_nodes}
                val_extrema = {node: single_variable_explainers[node].get_val_extrema(initial_classification=initial_classification) for node in evaluation_nodes}

                scores = {
                    node: (
                        self.changeability_scores[node[-1]] * 
                        single_variable_explainers[node].get_layer_score(initial_classification=initial_classification)
                    ) + previous_scores[node[:-1]] 
                    for node in evaluation_nodes
                }
                previous_scores.update(scores)

                if initial_classification==0: masked = {k: v for k,v in scores.items() if val_extrema[k] > 0.5 + additional_threshold}
                else: masked = {k: v for k,v in scores.items() if val_extrema[k] < 0.5 - additional_threshold}

                if len(masked.keys())>0:
                    found = True
                    solution = max(masked, key=masked.get)
                        
                for key, value in arg_extrema.items():
                    previous_value = evaluation_points_dict[key[:-1]].copy()
                    previous_value[key[-1]] = value
                    evaluation_points_dict[key] = previous_value

                    extrema_dict[key] = val_extrema[key]

                searcher.select_next_node(scores)

            if found:
                return solution, extrema_dict

        except ValueError:
            return extrema_dict