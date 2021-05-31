#Z0096

from measure import model_report

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


#################### Model telco_churn Data ####################


def baseline_model(X, y, strategy='most_frequent', random_state=19):
    '''
    '''

    # assign baseline model and fit to data
    baseline = DummyClassifier(strategy=strategy, random_state=random_state)
    baseline.fit(X, y)
    # assign baseline predictions
    y_baseline = baseline.predict(X)
    # print baseline accuracy score and first ten values for training data
    print(f'''
               Baseline Accuracy Score: {baseline.score(X, y):.2%}
        First Ten Baseline Predictions: {y_baseline[:10]}
        ''')

    return baseline, y_baseline


def tree_model(X, y, 
               criterion='gini',
               splitter='best',
               max_depth=None,
               min_samples_split=2,
               min_samples_leaf=1,
               min_weight_fraction_leaf=0.0,
               max_features=None,
               random_state=19,
               max_leaf_nodes=None,
               min_impurity_decrease=0.0,
               min_impurity_split=None,
               class_weight=None,
               ccp_alpha=0.0):
    '''
    '''

    # assign model and fit to data
    model = DecisionTreeClassifier(
               criterion=criterion,
               splitter=splitter,
               max_depth=max_depth,
               min_samples_split=min_samples_split,
               min_samples_leaf=min_samples_leaf,
               min_weight_fraction_leaf=min_weight_fraction_leaf,
               max_features=max_features,
               random_state=random_state,
               max_leaf_nodes=max_leaf_nodes,
               min_impurity_decrease=min_impurity_decrease,
               min_impurity_split=min_impurity_split,
               class_weight=class_weight,
               ccp_alpha=ccp_alpha)
    model.fit(X, y)
    # assign model predictions
    y_pred = model.predict(X)
    # print model metrics
    model_report(y, y_pred)

    return model, y_pred


def forest_model(X, y,
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=19,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
    '''
    '''

    # assign model and fit to data
    model = RandomForestClassifier(
                 n_estimators=n_estimators,
                 criterion=criterion,
                 max_depth=max_depth,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 bootstrap=bootstrap,
                 oob_score=oob_score,
                 n_jobs=n_jobs,
                 random_state=random_state,
                 verbose=verbose,
                 warm_start=warm_start,
                 class_weight=class_weight,
                 ccp_alpha=ccp_alpha,
                 max_samples=max_samples)
    model.fit(X, y)
    # assign model predictions
    y_pred = model.predict(X)
    # print model metrics
    model_report(y, y_pred)

    return model, y_pred


def knn_model(X, y,
              n_neighbors=5,
              weights='uniform',
              algorithm='auto',
              leaf_size=30,
              p=2,
              metric='minkowski',
              metric_params=None,
              n_jobs=None):
    '''
    '''

    # assign model and fit to data
    model = KNeighborsClassifier(
              n_neighbors=n_neighbors,
              weights=weights,
              algorithm=algorithm,
              leaf_size=leaf_size,
              p=p,
              metric=metric,
              metric_params=metric_params,
              n_jobs=n_jobs)
    model.fit(X, y)
    # assign model predictions
    y_pred = model.predict(X)
    # print model metrics
    model_report(y, y_pred)

    return model, y_pred


def logit_model(X, y,
              penalty='l2',
              dual=False,
              tol=0.0001,
              C=1.0,
              fit_intercept=True,
              intercept_scaling=1,
              class_weight=None,
              random_state=19,
              solver='lbfgs',
              max_iter=100,
              multi_class='auto',
              verbose=0,
              warm_start=False,
              n_jobs=None,
              l1_ratio=None):
    '''
    '''

    # assign model and fit to data
    model = LogisticRegression(
              penalty=penalty,
              dual=dual,
              tol=tol,
              C=C,
              fit_intercept=fit_intercept,
              intercept_scaling=intercept_scaling,
              class_weight=class_weight,
              random_state=random_state,
              solver=solver,
              max_iter=max_iter,
              multi_class=multi_class,
              verbose=verbose,
              warm_start=warm_start,
              n_jobs=n_jobs,
              l1_ratio=l1_ratio)
    model.fit(X, y)
    # assign model predictions
    y_pred = model.predict(X)
    # print model metrics
    model_report(y, y_pred)

    return model, y_pred


def validate(X, y, model):
    '''
    '''

    # assign model predictions on validate data
    y_pred = model.predict(X)
    # print model metrics on validate data
    model_report(y, y_pred)

    return y_pred


def final_test(X, y, model):
    '''
    '''

    # assign model predictions on test data
    y_pred = model.predict(X)
    # print model metrics on test data
    model_report(y, y_pred)

    return y_pred
