import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def do_plot_heatmap(df, prediction_col, target_col):
    t_df = df[[prediction_col, target_col]].value_counts().reset_index().pivot(index=prediction_col, columns=target_col)
    t_df = t_df/t_df.sum().sum()
    seaborn.heatmap(t_df, annot=True)
    plt.show()


def create_summary(df, features, target_var, number=10):
    plot_groupby = [feature if len(df[feature].unique())<30 else pd.cut(df[feature], np.linspace(df[feature].min(), df[feature].max()), number) for feature in features]
    plot_ranges = [feature for feature in features if len(df[feature].unique())>=30]

    plot_df = df[features+[target_var]].groupby(plot_groupby, observed=False).agg(['mean', 'count']).dropna()
    plot_df = pd.DataFrame(plot_df[target_var]).reset_index()
    
    for feature in plot_ranges:
        plot_df[feature] = [x.mid for x in plot_df[feature]]
    
    plot_df = plot_df[plot_df['count']>2]
    plot_df = plot_df.drop('count', axis=1)
    
    return plot_df


def do_plot_decision_boundary(model, df, feature_x, feature_y, target_var):
    ax = plt.subplot(1, 1, 1)
    cm = plt.cm.RdBu
    features = [feature_x, feature_y]
    DecisionBoundaryDisplay.from_estimator(model, df[features], ax=ax,alpha=0.3, cmap=cm)
    
    plot_df = create_summary(df, features, target_var, number=50)
    
    ax.scatter(
        plot_df[feature_x],
        plot_df[feature_y],
        c=plot_df['mean'],
        cmap=cm,
        alpha=0.5
    )
    plt.show()


def make_model(
    df: pd.DataFrame,
    model: any,
    target_variable: str,
    independent_variables: list,
    test_size: float = 0.2,
    plot_heatmap: bool = False,
    plot_decision_boundary: bool = False,
    random_state: int = 0,
    pre_processing: dict = None,
    dropna: bool = True,
    include_standard_scaler: bool = True,
    return_training_set: bool = False,
):
    if (len(independent_variables)!=2) and (plot_decision_boundary):
        raise ValueError("Only 2 independent variables are allowed for plotting decision boundary")
    

    # Splitting the data into train and test
    train_X, test_X, train_y, test_y = train_test_split(
        df[independent_variables], 
        df[target_variable], 
        test_size=test_size, 
        random_state=random_state
    )

    if pre_processing is not None:
        for col in independent_variables:
            if col in pre_processing.keys():
                if type(pre_processing[col]) is str:
                    if pre_processing[col] == 'fill_with_mean':
                        train_X[col] = train_X[col].fillna(train_X[col].mean())
                else:
                    train_X[col] = pre_processing[col](train_X[col])

    if dropna:
        train_X.dropna(inplace=True)
        test_X.dropna(inplace=True)
        train_y = train_y.loc[train_X.index]
        test_y = test_y.loc[test_X.index]

    # Fitting the model
    if include_standard_scaler:
        model = make_pipeline(StandardScaler(), model)
    model.fit(train_X, train_y)
    
    train_X['prediction'] = model.predict(train_X)
    test_X['prediction'] = model.predict(test_X)

    train_X[target_variable] = train_y
    test_X[target_variable] = test_y

    if plot_heatmap:
        do_plot_heatmap(test_X, target_variable, 'prediction')
    
    if plot_decision_boundary:
        do_plot_decision_boundary(model, train_X, independent_variables[0], independent_variables[1], target_variable)

    if return_training_set:
        return model, train_X

    else:
        return model