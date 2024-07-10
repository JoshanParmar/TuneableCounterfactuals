import pandas as pd
import tuneable_counterfactuals_explainer.single_variable_explainer as sve
class BaseScorer():
    def get_initial_probability(
        self,
        single_variable_explainer
    ):
        initial_point = pd.DataFrame(single_variable_explainer.explanation_point).T[single_variable_explainer.underlying_model.feature_names_in_]
        initial_point.columns = single_variable_explainer.underlying_model.feature_names_in_
        return single_variable_explainer.probability_prediction_function(initial_point)[0][1]
    

    def get_score(
        self,
        *args,
        **kwargs   
    ):
        raise NotImplementedError
    
class BasicScorer(BaseScorer):
    def get_score(
        self,
        single_variable_explainer,
        initial_classification: int = None,
        resolution: int = 100
    ):
        initial_probability = self.get_initial_probability(single_variable_explainer)
        final_probability = single_variable_explainer.get_extrema(return_val=True, initial_classification=initial_classification, resolution=resolution)[1]
        return abs(final_probability - initial_probability)
