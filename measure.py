#Z0096

import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.metrics import confusion_matrix, classification_report


#################### Measure telco_churn Data ####################


def two_sample_ttest(a, b, null_hyp, alpha=0.05, equal_var=True,
                        alternative='two-sided'):
    '''

    Perform T-Test using scipy.stats.ttest, prints whether to reject or accept
    the null hypothesis, as well as alpha, p-value, and t-value

    '''

    t, p = ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    # print alpha and p-value
    print(f'''
  alpha: {alpha}
p-value: {p:.1g}''')
    # print if our p-value is less than our significance level
    if p < alpha:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we must reject the null hypothesis
        that {null_hyp}.
        ''')
    # print if our p-value is greater than our significance level
    else:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we fail to reject the null hypothesis
        that {null_hyp}.    
        ''')


def chi_test(cat, target, alpha=0.05):
    '''
    '''

    # set observed DataFrame with crosstab
    observed = pd.crosstab(cat, target)
    a = observed.iloc[0,0]
    b = observed.iloc[0,1]
    c = observed.iloc[1,0]
    d = observed.iloc[1,1]
    # assign returned values from chi2_contigency
    chi2, p, degf, expected = chi2_contingency(observed)
    # set expected DataFrame from returned array
    expected = pd.DataFrame(expected)
    a2 = expected.iloc[0,0]
    b2 = expected.iloc[0,1]
    c2 = expected.iloc[1,0]
    d2 = expected.iloc[1,1]
    # set null hypothesis
    null_hyp = f'{target.name} is independent of {cat.name}'
    # print alpha and p-value
    print(f'''
  alpha: {alpha}
p-value: {p:.1g}''')
    # print if our p-value is less than our significance level
    if p < alpha:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we must reject the null hypothesis
        that {null_hyp}.''')
    # print if our p-value is greater than our significance level
    else:
        print(f'''
        Due to our p-value of {p:.1g} being less than our significance level of {alpha}, we fail to reject the null hypothesis
        that {null_hyp}.''')
    # print observed and expected DataFrames side by side
    print(f'''
                       ** Observed **                        |       ** Expected **
                       --------------------------------------|--------------------------------------
                                     No Churn    Churn       |                     No Churn    Churn
                                                             |       
                       No Fiber      {a:<10.0f}  {b:<10.0f}  |       No Fiber      {a2:<10.0f}  {b2:<10.0f}
                                                             |       
                          Fiber      {c:<10.0f}  {d:<10.0f}  |          Fiber      {c2:<10.0f}  {d2:<10.0f}
    ''')


def cmatrix(y_true, y_pred):
    '''
    '''

    # define confusion matrix, convert to dataframe
    cmatrix = confusion_matrix(y_true, y_pred)
    cmatrix = pd.DataFrame(confusion_matrix(y_true, y_pred),
                           index=['True Retain', 'True Churn'],
                           columns=['Predict Retain', 'Predict Churn'])
    # assign TN, FN, TP, FP
    true_neg = cmatrix.iloc[0, 0]
    false_neg = cmatrix.iloc[0, 1]
    true_pos = cmatrix.iloc[1, 0]
    false_pos = cmatrix.iloc[1, 1]
    #do math to find rates
    tpr = true_pos / (true_pos + false_neg)
    tnr = true_neg / (true_neg + false_pos)
    fpr = 1 - tnr
    fnr = 1 - tpr
    cmatrix_dict = {'tpr':tpr, 'tnr':tnr, 'fpr':fpr, 'fnr':fnr}

    return cmatrix_dict


def model_report(y_true, y_pred):
    '''
    '''

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    cmatrix_dict = cmatrix(y_true, y_pred)
    print(f'''
            *** Model  Report ***  
            ---------------------              
 _____________________________________________
|            Positive Case: Churn  (1)        |
|            Negative Case: Retain (0)        |
|---------------------------------------------|
|                 Accuracy: {report_dict['accuracy']:>8.2%}          |
|       True Positive Rate: {cmatrix_dict['tpr']:>8.2%}          |
|      False Positive Rate: {cmatrix_dict['fpr']:>8.2%}          |
|       True Negative Rate: {cmatrix_dict['tnr']:>8.2%}          |
|      False Negative Rate: {cmatrix_dict['fnr']:>8.2%}          |
|                Precision: {report_dict['1']['precision']:>8.2%}          |
|                   Recall: {report_dict['1']['recall']:>8.2%}          |
|                 F1-Score: {report_dict['1']['f1-score']:>8.2%}          |
|                                             |
|         Positive Support: {report_dict['1']['support']:>8}          |
|         Negative Support: {report_dict['0']['support']:>8}          |
|            Total Support: {report_dict['macro avg']['support']:>8}          |
|_____________________________________________|
''')
