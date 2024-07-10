import traceback
import numpy as np
import pandas as pd
import time

from tuneable_counterfactuals_explainer.searcher import Searcher
from tuneable_counterfactuals_explainer.scorer import BaseScorer, BasicScorer
from tuneable_counterfactuals_explainer.single_variable_explainer import SingleVariableExplainer

from tqdm import tqdm
from typing import Union,Dict,Tuple

import multiprocessing


def generator(args):
    SingleVariableExplainer(**args)
class Explainer:
    def __init__(
        self,
        underlying_model,
        training_dataset,
        target_variable, 
        variables: list = None,
        sampling_method = 'uniform',       
        bounding_method = 'meanstd',
        override_variable_bounds: Union[Dict[str, Tuple[int, int]], Tuple[int, int]] = None,
        probability_prediction_function = None,
        class_prediction_function = None,
        std_dev = 1,
        quantiles = 0.1,
        number_samples = 10,
        regressor = 'gaussian_process',
        scorer: Union[str, BaseScorer] = 'basic',
        changeability_scores = None,
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
            - override_variable_bounds: dictionary of bounds to use for each variable (default: None), see explainer.single_variable_explainer for more details
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
        self.sampling_method = sampling_method
        self.bounding_method = bounding_method
        self.std_dev = std_dev
        self.quantiles = quantiles
        self.number_samples = number_samples
        self.regressor = regressor
        self.override_variable_bounds = override_variable_bounds

        if variables is None:
            if changeability_scores is not None:
                self.variables = list(changeability_scores.keys())
            else:
                self.variables = list(self.training_dataset.columns)
                if self.target_variable in self.variables:
                    self.variables.remove(self.target_variable)

        if changeability_scores is None:
            self.changeability_scores = {
                col: 1
                for col in self.variables
            }
        else: 
            self.changeability_scores = changeability_scores

        if isinstance(scorer, str):
            if scorer == 'basic':
                self.scorer = BasicScorer()
            else:
                raise ValueError('scorer must be one one [\'basic\'] or a BaseScorer object')
        
        if probability_prediction_function is None:
            self.probability_prediction_function = lambda x: self.underlying_model.predict_proba(x)
        else:
            self.probability_prediction_function = probability_prediction_function


        if class_prediction_function is None:
            self.class_prediction_function = lambda x: self.underlying_model.predict(x)
        else:
            self.class_prediction_function = class_prediction_function
        
    def handle_variable_bounds(
        self,
        variable
    ):
        if self.override_variable_bounds is None:
            return None
        elif isinstance(self.override_variable_bounds, tuple):
            return self.override_variable_bounds
        elif isinstance(self.override_variable_bounds, dict):
            if variable in self.override_variable_bounds.keys():
                return self.override_variable_bounds[variable]
        return None


    def explain(
        self, 
        explanation_point,
        initial_classification: int = None,
        additional_threshold = 0,
        store_historical_explainers = False,
        do_parallel = False,
        store_historical_times = False
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
            initial_classification = int(self.class_prediction_function(pd.DataFrame(explanation_point[self.underlying_model.feature_names_in_]).T))
        searcher = Searcher(self.variables)
        evaluation_points_dict = {tuple(): explanation_point}

        initial_point = pd.DataFrame(explanation_point).T[self.underlying_model.feature_names_in_]
        initial_point.columns = self.underlying_model.feature_names_in_

        extrema_dict = {tuple(): self.probability_prediction_function(initial_point)[0][1]}
        previous_scores = {tuple(): 0}
        

        if do_parallel:
            p = multiprocessing.Pool()

        found = False
        historical_scores = []
        if store_historical_times: historical_times = []
        if store_historical_explainers: historical_explainers = {}
        with tqdm(total=len(self.variables)) as pbar:
            current_time = time.time()
            try: 
                while not found:
                    current_node = searcher.current_node
                    evaluation_nodes = searcher.get_next_evaluation_set()
                    evaluation_point = evaluation_points_dict[current_node]
                    
                    if do_parallel:
                        args_generator = [
                            {
                                'underlying_model': self.underlying_model,
                                'target_variable': self.target_variable,
                                'explainable_variable': node[-1],
                                'explanation_point': evaluation_point,
                                'training_dataset': self.training_dataset,
                                'variable_bounds': self.handle_variable_bounds(node[-1]),
                                'sampling_method': self.sampling_method,
                                'bounding_method': self.bounding_method,
                                'std_dev': self.std_dev,
                                'quantiles': self.quantiles,
                                'number_samples': self.number_samples,
                                'regressor': self.regressor
                            }
                            for node in evaluation_nodes
                        ]
                        single_variable_explainers = dict(
                            zip(
                                evaluation_nodes,
                                tqdm(p.imap(generator, args_generator))
                            )
                        )
                    
                    
                    else:
                        single_variable_explainers = {
                            node: SingleVariableExplainer(
                                underlying_model = self.underlying_model,
                                target_variable = self.target_variable,
                                explainable_variable = node[-1],
                                explanation_point = evaluation_point,
                                training_dataset = self.training_dataset,
                                variable_bounds = self.handle_variable_bounds(node[-1]),
                                sampling_method = self.sampling_method,
                                bounding_method = self.bounding_method,
                                std_dev = self.std_dev,
                                number_samples = self.number_samples,
                                regressor = self.regressor,
                                probability_prediction_function = self.probability_prediction_function,
                                class_prediction_function = self.class_prediction_function
                            )
                            for node in evaluation_nodes
                        }
                    
                    arg_extrema = {node: single_variable_explainers[node].get_arg_extrema(initial_classification=initial_classification) for node in evaluation_nodes}
                    val_extrema = {node: single_variable_explainers[node].get_val_extrema(initial_classification=initial_classification) for node in evaluation_nodes}

                    scores = {
                        node: (
                            self.changeability_scores[node[-1]] * 
                            self.scorer.get_score(single_variable_explainers[node], initial_classification=initial_classification)
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
                        previous_value[key[-1]] = np.float32(value)
                        evaluation_points_dict[key] = previous_value

                        extrema_dict[key] = val_extrema[key]

                    searcher.select_next_node(scores)
                    historical_scores.append(max(scores.values()))
                    pbar.update(1)
                    if store_historical_explainers: historical_explainers.update(single_variable_explainers)
                    if store_historical_times: historical_times.append(time.time() - current_time)
                if found:
                    if store_historical_explainers: 
                        if store_historical_times: 
                            return solution, single_variable_explainers[solution], extrema_dict, historical_scores, historical_explainers, historical_times
                        else:
                            return solution, single_variable_explainers[solution], extrema_dict, historical_scores, historical_explainers
                    if store_historical_times: 
                        return solution, single_variable_explainers[solution], extrema_dict, historical_scores, historical_times
                    return solution, single_variable_explainers[solution], extrema_dict, historical_scores

            except ValueError as e:
                print(e)
                return extrema_dict