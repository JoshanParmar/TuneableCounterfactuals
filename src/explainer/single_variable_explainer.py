from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression


class SingleVariableExplainer():
    def __init__(
        self,
        model,
        target_variable,
        explainable_variable,
        explanation_point,
        training_dataset = None,
        variable_bounds = None,
        sampling_method = 'uniform',
        bounding_method = 'minmax',
        quantiles = 0.1,
        std_dev = 1,
        number_samples = 10,
        regressor = 'gaussian_process'
    ) -> None:
        if (training_dataset is None) and (variable_bounds is None):
            raise ValueError("Either training_dataset or variable_bounds must be provided")
        
        if sampling_method not in ['uniform', 'random']:
            raise ValueError("sampling_method must be either 'uniform' or 'random'")
        
        if sampling_method == 'random':
            if training_dataset is None:
                raise ValueError("training_dataset must be provided if sampling_method is 'random'")
        
        if bounding_method not in ['minmax', 'quantile', 'meanstd']:
            raise ValueError("bounding_method must be either 'minmax', 'quantile' or 'meanstd'")
        
        if regressor not in ['gaussian_process', 'linear']:
            raise ValueError("regressor must be either 'gaussian_process' or 'linear'")
        
        
        self.model = model
        self.target_variable = target_variable
        self.explainable_variable = explainable_variable
        self.regressor_type = regressor
        self.explanation_point = explanation_point

        if variable_bounds is None:
            if bounding_method=='minmax':
                variable_bounds = [
                    training_dataset[explainable_variable].min(),
                    training_dataset[explainable_variable].max()
                ]
            elif bounding_method=='quantile':
                variable_bounds = [
                    training_dataset[explainable_variable].quantile(quantiles),
                    training_dataset[explainable_variable].quantile(1-quantiles)
                ]
            elif bounding_method=='meanstd':
                variable_bounds = [
                    training_dataset[explainable_variable].mean()-(std_dev*training_dataset[explainable_variable].std()),
                    training_dataset[explainable_variable].mean()+(std_dev*training_dataset[explainable_variable].std())
                ]


        if sampling_method == 'uniform':
            self.samples = np.linspace(variable_bounds[0], variable_bounds[1], number_samples)
        
        if sampling_method == 'random':
            self.samples = np.random.choice(
                self.training_dataset[target_variable][(self.training_dataset[target_variable]>=variable_bounds[0]) & (self.training_dataset[target_variable]<=variable_bounds[1])],
                number_samples,
            )

        if regressor == 'gaussian_process':
            self.regressor = GaussianProcessRegressor(
                normalize_y=True,
                kernel=RBF(length_scale=self.samples.std(), length_scale_bounds='fixed'),
            )
        elif regressor == 'linear':
            self.regressor = LinearRegression()
    
        self.explanation_dataset = pd.DataFrame(explanation_point).T.iloc[[0 for _ in range(number_samples)]].reset_index().drop(columns=['index'])[model.feature_names_in_]
        self.explanation_dataset[explainable_variable] = self.samples
        self.explanation_dataset['_prediction'] = self.model.predict_proba(self.explanation_dataset)[:, 1]
        
        self.regressor.fit(self.explanation_dataset[[explainable_variable]], self.explanation_dataset['_prediction']) 
        self.variable_bounds = variable_bounds

    def plot(
        self,
        plot_resolution: int = 100
    ):
        X = np.linspace(
            self.variable_bounds[0],
            self.variable_bounds[1],
            plot_resolution
        )
        X_predict = pd.DataFrame(X)
        X_predict.columns = [self.explainable_variable]
        Y = self.regressor.predict(X_predict)

        self.explanation_dataset.plot(self.explainable_variable, '_prediction', kind='scatter')
        plt.plot(X,Y)
        
        if self.regressor_type == 'gaussian_process':
            _, Y_std = self.regressor.predict(X_predict, return_std=True)
            plt.fill_between(
                X,
                Y-Y_std,
                Y+Y_std,
                alpha=0.3
            )

        plt.xlabel(self.explainable_variable)
        plt.ylabel('Predicted probability')
        initial_point = pd.DataFrame(self.explanation_point).T[self.model.feature_names_in_]
        initial_point.columns = self.model.feature_names_in_
        
        plt.scatter(
            initial_point[self.explainable_variable],
            self.model.predict_proba(initial_point)[0][1],
            color='red',
            zorder=10
        )
