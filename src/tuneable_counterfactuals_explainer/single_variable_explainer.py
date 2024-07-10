from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression

from typing import Union

class SingleVariableExplainer():
    def __init__(
        self,
        underlying_model,
        target_variable,
        explainable_variable,
        explanation_point,
        training_dataset = None,
        variable_bounds = None,
        sampling_method = 'uniform',
        bounding_method = 'meanstd',
        quantiles = 0.1,
        std_dev = 1,
        number_samples = 10,
        regressor : Union[str, RegressorMixin]  = 'gaussian_process',
        probability_prediction_function = None,
        class_prediction_function = None,
        
    ) -> None:
        '''
            This class fits a simple, interpretable model to the to the underlying model to explain the effect of a single variable on the underlying model's prediction.

            Samples will be taken (following provided rules) for the explanation variable. We will use the underlying model to make predictions for each sample, leaving every other variable as it is in the training dataset. By fitting a simple model to this data, we can explain the effect of the variable on the underlying model's prediction.

            *Required Variables*
            underlying_model: SKLearn Model 
                The underlying model to be explained

            target_variable: str 
                The target variable of the underlying model

            explainable_variable: str
                The variable to be explained

            *Optional Variables*
            training_dataset: pd.DataFrame (optional)
                The training dataset used to train the underlying model. If not provided, variable_bounds must be provided.

            variable_bounds: list (optional)
                The expected bounds of the variable to be explained. This should be a two variable list, with the first element being the lower bound and the second element being the upper bound. If not provided, the bounds will be calculated from the training dataset. One of training_dataset or variable_bounds must be provided.

            sampling_method: str (optional)
                The method used to sample the variable to be explained. This should be either 'uniform' or 'random'. If not provided, 'uniform' will be used.

                Sampling method 'uniform' will sample the variable to be explained uniformly across the variable bounds - either provided manually, or using bounding rules on the training data (see below).

                Sampling method 'random' will sample the variable to be explained randomly from the training dataset. This will only work if training_dataset is provided.
            
            bounding_method: str (optional)
                The method used to calculate the variable bounds if they are not provided manually. This should be either 'minmax', 'quantile' or 'meanstd'. If not provided, 'minmax' will be used.

                minmax: The variable bounds will be the minimum and maximum values of the variable to be explained in the training dataset.

                quantile: The variable bounds will be the quantiles of the variable to be explained in the training dataset. The quantiles used are provided by the quantiles variable.

                meanstd: The variable bounds will be the mean plus or minus a number of standard deviations of the variable to be explained in the training dataset. The number of standard deviations used is provided by the std_dev variable.

            quantiles: float (optional)
                The quantiles used to calculate the variable bounds if bounding_method is 'quantile'. This should be a float between 0 and 1. If not provided, 0.1 will be used.

            std_dev: float (optional)
                The number of standard deviations used to calculate the variable bounds if bounding_method is 'meanstd'. This should be a float. If not provided, 1 will be used.

            number_samples: int (optional)
                The number of samples to be taken to explain the variable. This should be an integer. If not provided, 10 will be used.

            regressor: str or SKLearn Regressor (optional)
                The regressor used to fit the simple model. This should be either 'gaussian_process', 'linear' or a SKLearn Regressor. If not provided, 'gaussian_process' will be used.

            *Methods*
            plot: Plots the simple model and the underlying model's predictions for the samples taken.

            get_extrema: Returns the extrema of the simple model. This can be either the maximum or minimum value of the simple model, depending on the initial classification of the underlying model's prediction for the explanation point.

            get_arg_extrema: Returns the value of the explanation variable that causes the most change in the models output. This can be either the value of the explanation variable that causes the maximum or minimum value of the simple model, depending on the initial classification of the underlying model's prediction for the explanation point.

            get_val_extrema: Returns the model's output at the explanation variable that causes the most change in the models output. This can be either the maximum or minimum value of the simple model, depending on the initial classification of the underlying model's prediction for the explanation point.

        '''
        
        # Check input
        if (training_dataset is None) and (variable_bounds is None):
            raise ValueError("Either training_dataset or variable_bounds must be provided")
        
        if sampling_method not in ['uniform', 'random']:
            raise ValueError("sampling_method must be either 'uniform' or 'random'")
        
        if sampling_method == 'random':
            if training_dataset is None:
                raise ValueError("training_dataset must be provided if sampling_method is 'random'")
        
        if bounding_method not in ['minmax', 'quantile', 'meanstd']:
            raise ValueError("bounding_method must be either 'minmax', 'quantile' or 'meanstd'")
        
        if isinstance(regressor, RegressorMixin):
            if regressor not in ['gaussian_process', 'linear']:
                raise ValueError("regressor must be either 'gaussian_process' or 'linear'")
        
        
        # Set attributes
        self.underlying_model = underlying_model
        self.target_variable = target_variable
        self.explainable_variable = explainable_variable
        self.regressor_type = regressor
        self.explanation_point = explanation_point
        self.training_dataset = training_dataset


        # Handle variable bounds
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
        self.variable_bounds = variable_bounds

        # Do Sampling
        if sampling_method == 'uniform':
            self.samples = np.linspace(variable_bounds[0], variable_bounds[1], number_samples)
        
        if sampling_method == 'random':
            self.samples = np.random.choice(
                self.training_dataset[target_variable][(self.training_dataset[target_variable]>=variable_bounds[0]) & (self.training_dataset[target_variable]<=variable_bounds[1])],
                number_samples,
            )

        # Set up regressor
        if isinstance(regressor, RegressorMixin):
            self.regressor = regressor.copy()
        else:
            if regressor == 'gaussian_process':
                self.regressor = GaussianProcessRegressor(
                    normalize_y=True,
                    kernel=RBF(length_scale=self.samples.std(), length_scale_bounds='fixed'),
                )
            elif regressor == 'linear':
                self.regressor = LinearRegression()
    
        if probability_prediction_function is None:
            self.probability_prediction_function = lambda x: self.underlying_model.predict_proba(x)
            self.to_double_index = True
        else:
            self.probability_prediction_function = probability_prediction_function
            self.to_double_index = False

        
        if class_prediction_function is None:
            self.class_prediction_function = lambda x: self.underlying_model.predict(x)
        else:
            self.class_prediction_function = class_prediction_function

        # Create explanation dataset
        self.explanation_dataset = pd.DataFrame(explanation_point).T.iloc[[0 for _ in range(number_samples)]].reset_index().drop(columns=['index'])[underlying_model.feature_names_in_]
        self.explanation_dataset[explainable_variable] = self.samples
        self.explanation_dataset['_prediction'] = self.probability_prediction_function(self.explanation_dataset[self.underlying_model[0].feature_names_in_])[:, 1]
        
        # Fit regressor
        self.regressor.fit(self.explanation_dataset[[explainable_variable]], self.explanation_dataset['_prediction']) 



    def plot(
        self,
        initial_classification: int = None,
        plot_resolution: int = 100,
        ax=None,
        show_arrow: bool = True,
    ):
        '''
            Plots the simple model and the underlying model's predictions for the samples taken.

            *Optional Variables*
            plot_resolution: int (optional)
                The number of points to be plotted for the simple model. This should be an integer. If not provided, 100 will be used.
        '''
        if initial_classification is None:
            initial_classification = int(self.class_prediction_function(pd.DataFrame(self.explanation_point[self.underlying_model.feature_names_in_]).T))
        if ax is None:
            fig, ax = plt.subplots()
        X = np.linspace(
            self.variable_bounds[0],
            self.variable_bounds[1],
            plot_resolution
        )
        X_predict = pd.DataFrame(X)
        X_predict.columns = [self.explainable_variable]
        Y = self.regressor.predict(X_predict)

        self.explanation_dataset.plot(self.explainable_variable, '_prediction', kind='scatter', ax=ax)
        ax.plot(X,Y)
        
        if self.regressor_type == 'gaussian_process':
            _, Y_std = self.regressor.predict(X_predict, return_std=True)
            ax.fill_between(
                X,
                Y-Y_std,
                Y+Y_std,
                alpha=0.3
            )

        ax.set_xlabel(self.explainable_variable)
        ax.set_ylabel('Predicted probability')
        initial_point = pd.DataFrame(self.explanation_point).T[self.underlying_model.feature_names_in_]
        initial_point.columns = self.underlying_model.feature_names_in_
        
        ax.scatter(
            initial_point[self.explainable_variable],
            self.probability_prediction_function(initial_point)[0][1],
            color='red',
            zorder=10
        )

        ax.scatter(
            self.get_arg_extrema(initial_classification=initial_classification),
            self.get_val_extrema(initial_classification=initial_classification),
            color='green',
            zorder=10
        )
        if show_arrow:
            ax.arrow(
                (initial_point[self.explainable_variable] + 0.02*(self.get_arg_extrema(initial_classification=initial_classification) - initial_point[self.explainable_variable])).values[0],
                self.probability_prediction_function(initial_point)[0][1] + 0.02*(self.get_val_extrema(initial_classification=initial_classification) - self.underlying_model.predict_proba(initial_point)[0][1]),
                (0.92*(self.get_arg_extrema(initial_classification=initial_classification) - initial_point[self.explainable_variable])).values[0],
                0.92*(self.get_val_extrema(initial_classification=initial_classification) - self.probability_prediction_function(initial_point)[0][1]),
                color='grey',
                alpha=0.5,
                width=(0.02*(self.get_val_extrema(initial_classification=initial_classification) - self.probability_prediction_function(initial_point)[0][1])),
                # head_width=0.05,
                head_length=abs(0.05*(0.8*(self.get_arg_extrema(initial_classification=initial_classification) - initial_point[self.explainable_variable])).values[0]),
            )

    def get_extrema(
        self,
        return_val: bool = True,
        initial_classification: int = None,
        resolution = 100
    ):
        '''
            Get's the value of the explanation variable the causes the most change in the models output

            *Optional Variables*
            return_val: bool (optional)
                If True, also returns the value of the explanation variable that causes the most change in the models output. If False, only returns the value of the explanation variable that causes the most change in the models output.

            initial_classification: int (optional)
                The initial classification of the underlying model's prediction for the explanation point. This should be either 0 or 1. If not provided, the underlying model will be used to classify the explanation point.
            
            resolution: int (optional)
                The number of points to be used in the evaluation of the simple model. This should be an integer. If not provided, 100 will be used.
        '''
        if initial_classification is None:
            initial_classification = int(self.class_prediction_function(pd.DataFrame(self.explanation_point[self.underlying_model.feature_names_in_]).T))
        
        X = np.linspace(
            self.variable_bounds[0],
            self.variable_bounds[1],
            resolution
        )
        X_predict = pd.DataFrame(X)
        X_predict.columns = [self.explainable_variable]
        Y = self.regressor.predict(X_predict)

        if initial_classification==0:
            val = np.max(Y)
            arg = X[np.argmax(Y)]
        elif initial_classification==1:
            val = np.min(Y)
            arg = X[np.argmin(Y)]
        else:
            print(initial_classification)

        if return_val:
            return arg, val
        else:
            return arg
        
    def get_arg_extrema(
        self,
        initial_classification: int = None,
        resolution = 100
    ):
        '''
            Get's the value of the explanation variable the causes the most change in the models output

            *Optional Variables*
            initial_classification: int (optional)
                The initial classification of the underlying model's prediction for the explanation point. This should be either 0 or 1. If not provided, the underlying model will be used to classify the explanation point.
            
            resolution: int (optional)
                The number of points to be used in the evaluation of the simple model. This should be an integer. If not provided, 100 will be used.
        '''
        return self.get_extrema(
            return_val=False,
            initial_classification=initial_classification,
            resolution=resolution
        )
    
    def get_val_extrema(
        self,
        initial_classification: int = None,
        resolution = 100
    ):
        '''
            Get's the model's output at the explanation variable the causes the most change in the models output

            *Optional Variables*
            initial_classification: int (optional)
                The initial classification of the underlying model's prediction for the explanation point. This should be either 0 or 1. If not provided, the underlying model will be used to classify the explanation point.
            
            resolution: int (optional)
                The number of points to be used in the evaluation of the simple model. This should be an integer. If not provided, 100 will be used.
        '''
        return self.get_extrema(
            return_val=True,
            initial_classification=initial_classification,
            resolution=resolution
        )[1]
    
    