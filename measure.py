#Z0096


import scipy.stats as stats


#################### Measure telco_churn Data ####################


def two_sample_ttest(a, b, alpha, null_hyp, equal_var=True,
                        alternative='two-sided'):
    '''

    Perform T-Test using scipy.stats.ttest, prints whether to reject or accept
    the null hypothesis, as well as alpha, p-value, and t-value

    '''

    t, p = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)

    print(f'''
  alpha: {alpha}
p-value: {p:.1g}
t-value: {t:.1g}''')
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
