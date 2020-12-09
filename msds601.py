import copy
from math import *
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.stats import kstest
from scipy.stats import boxcox
import scipy.linalg as linalg
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

class msds601():
    def __init__(self, filename, y, X):
        self.filename = filename 
        self.df = pd.read_csv(filename)
        self.y = y
        self.X = X
    
    def slr(self):
        if len(self.X) > 1:
            print('ERROR: More than 1 predictors.')
            return
        y = self.y
        x = self.X[0]
        model = smf.ols('y ~ x', data=self.df).fit()
        return model
    
    def mlr(self):
        y = self.y
        X = self.X
        modelstr = 'y ~ x1'
        for index, x in enumerate(self.X):
            if index != 0:
                modelstr += f' + x{index+1}'
            locals()['x' + str(index+1)] = x
        model = smf.ols(modelstr, data=self.df).fit()
        return model 
    
    def logit(self):
        y = self.y
        X = self.X
        df = self.df
        if set(y) != set([0, 1]):
            print('ERROR: The response variable is not dummy.')
            return
        if len(X) == 1:
            modelstr = 'y ~ x'
            x = self.X[0]
        else:
            modelstr = 'y ~ x1'
            for index, x in enumerate(self.X):
                if index != 0:
                    modelstr += f' + x{index+1}'
                locals()['x' + str(index+1)] = x
        model = smf.glm(modelstr, data=df, family=sm.families.Binomial()).fit()
        return model
    
    def linXy(self):
        y = self.y
        X = self.X
        predictors = [np.ones(len(y))]
        for x in X:
            predictors.append(list(x))
        X = np.array(predictors)
        y = np.array(y)
        return y, X.T
    
    def OLSE(self):
        y = self.y
        X = self.X
        y, X = self.linXy()
        Xt = X.T
        b = np.dot(np.dot(linalg.inv(np.dot(Xt, X)), Xt), y)
        return b
    
    def resid(self):
        y = self.y
        X = self.X
        y, X = self.linXy()
        b = self.OLSE()
        e = y - np.dot(X, b)
        return e
    
    def Hat_matrix(self):
        y = self.y
        X = self.X
        y, X = self.linXy()
        H = np.dot(np.dot(X, linalg.inv(np.dot(X.T, X))), X.T)
        return H  
    
    def SSE(self):
        y = self.y
        X = self.X
        y, X = self.linXy()
        H = self.Hat_matrix()
        SSE = np.dot(np.dot(y.T, (np.identity(len(y)) - H)), y)
        return SSE
    
    def SSR(self):
        y = self.y
        X = self.X
        y, X = self.linXy()
        Jn = np.ones(len(y)**2).reshape(len(y),len(y))
        H = self.Hat_matrix()
        SSR = np.dot(np.dot(y.T, H - Jn * 1 / len(y)), y) 
        return SSR
    
    def SST(self):
        y = self.y
        X = self.X
        y, X = self.linXy()
        Jn = np.ones(len(y)**2).reshape(len(y),len(y))
        SST = np.dot(np.dot(y.T, np.identity(len(y)) - Jn * 1 / len(y)),y) 
        return SST
    
    def ANOVA(self, type_=1):
        if len(self.X) == 1:
            model = self.slr()
        else:
            model = self.mlr()
        print(sm.stats.anova_lm(model, typ=type_))
        return 
    
    def MSE(self):
        n = len(self.y)
        k = len(self.X) + 1
        return self.SSE() / (n - k)
    
    def MSR(self):
        k = len(self.X) + 1
        return self.SSR() / (k - 1)
    
    def MST(self):
        n = len(self.y)
        return self.SST() / (n - 1)
    
    def F_stats(self, alpha=0.05):
        n = len(self.y)
        k = len(self.X) + 1
        F = self.MSR()/self.MSE()
        pval = 1 - stats.f.cdf(F, k-1, n-k) 
        return F, pval
    
    def R_square(self, adjusted=False):
        n = len(self.y)
        k = len(self.X) + 1
        if adjusted == True:
            adjR2 = 1 - self.MSE()/self.MST()
            return adjR2
        else:
            R2 = self.SSR()/self.SST()
            return R2
    
    def se(self):
        y, X = self.linXy()
        return stats.sem(X)
    
    def vif(self):
        y = self.y
        X = self.X
        modelstr = 'y ~ x1'
        for index, x in enumerate(self.X):
            if index != 0:
                modelstr += f' + x{index+1}'
            locals()['x' + str(index+1)] = x
        y, X = dmatrices(modelstr, data=self.df, return_type='dataframe')
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        vif["Multicollinearity"] = vif["VIF Factor"] >= 10
        return vif
        
    def ex_studti(self):
        df = self.df
        model = self.mlr()
        infl = model.get_influence()
        ex_studti = infl.resid_studentized_external

        n = len(ex_studti)
        k = model.df_model + 1

        seuil_stud = scipy.stats.t.ppf(0.975, df=n-k-1)
        atyp_stud = np.abs(ex_studti) >= seuil_stud

        ex_studenti = pd.DataFrame()
        ex_studenti['Index of Position'] = df.index[atyp_stud]
        ex_studenti['External Studendized Residual'] = ex_studti[atyp_stud]
        residuals = model.resid
        ex_studenti['Original Residuals'] = list(residuals[atyp_stud])
        return ex_studenti
    
    def cooks_distance(self):
        df = self.df
        model = self.mlr()
        infl = model.get_influence()
        inflsum = infl.summary_frame()

        reg_cook = inflsum.cooks_d
        atyp_cook = np.abs(reg_cook) >= 4/len(self.y)

        cooki = pd.DataFrame()
        cooki['Index of Position'] = df.index[atyp_cook]
        cooki['Cook\'s Distance'] = list(reg_cook[atyp_cook])
        return cooki
    
    def influential_points(self):
        df = self.df
        ex_studenti = self.ex_studti()
        cooki = self.cooks_distance()
        influencial_points = set(ex_studenti['Index of Position']).union(set(cooki['Index of Position']))
        return list(influencial_points)
    
    def drop_influential_points(self, inplace=False):
        df = self.df
        influencial_points = self.influential_points()
        df_droppoints = self.df
        for i in influencial_points:
            df_droppoints = df_droppoints.drop(index=i, axis =1)
        if inplace == True:
            self.df = df_droppoints.reset_index(drop=True)
            return
        else:
            return df_droppoints.reset_index(drop=True)
            
    def fitted_resid_plot(self):
        model = self.mlr()
        fittedvals = model.fittedvalues
        residuals = model.resid

        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(fittedvals, residuals, alpha=0.4, edgecolors='none')
        ax.hlines(0, xmin=min(fittedvals), xmax=max(fittedvals), colors='r', alpha=0.4)
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Fitted Values vs. Residuals")
        plt.show()
        
    def BP_test(self):
        model = self.mlr()
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        labels = ['BP Statistic', 'BP-Test p-value']
        bp_result = dict(zip(labels, bp_test))
        if bp_result['BP-Test p-value'] < 0.05:
            print('Reject H0, there’s a significant heteroscedasticity problem at the 95% confidence level.')
        else:
            print('Can\'t reject H0, there’s no significant heteroscedasticity problem at the 95% confidence level.')
        return bp_result
    
    def skewness(self):
        model = self.mlr()
        plt.hist(model.resid)
        return stats.skew(model.resid)
        
    def kurtosis(self):
        model = self.mlr()
        plt.hist(model.resid)
        return stats.kurtosis(model.resid)
    
    def qqplot(self):
        model = self.mlr()
        fig = sm.qqplot(model.resid)
        return
    
    def ominibus(self):
        model = self.mlr()
        KstestResult = kstest(model.resid,'norm')
        if KstestResult[1] < 0.05:
            print('Reject H0, the model is not normality at 95% confidence level.')
        else:
            print('Can\'t reject H0, the model is normality at 95% confidence level.')
        labels = ['Ominibus Statistic', 'Ominibus-Test p-value']
        Ominibus_result = dict(zip(labels, KstestResult))
        return Ominibus_result
        
    def jb_test(self): 
        model = self.mlr()
        JBresult = stats.jarque_bera(model.resid)
        if JBresult[1] < 0.05:
            print('Reject H0, the model is not normality at 95% confidence level.')
        else:
            print('Can\'t reject H0, the model is normality at 95% confidence level.')
        labels = ['JB Statistic', 'JB-Test p-value']
        JB_result = dict(zip(labels, JBresult))
        return JB_result
    
    def SubsetSelection(self, keyword):
        '''
        This is a function for the subset model selection. The
        output of this function is a dataframe with the stats 
        of every subset. 
        The input df is our input dataframe, and the input keyword
        is the name of the dependent variable.
        '''
        df = self.df
        selection = pd.DataFrame()
        predictors = list(df.keys())
        predictors.remove(keyword)
        subsets = [[]]

        for predictor in predictors:
            prev = copy.deepcopy(subsets)
            [k.append(predictor) for k in subsets]
            subsets.extend(prev)

        predstr = [' + '.join(i) for i in subsets if i]
        varnum = [len(i) for i in subsets if i]

        model_full = smf.ols(f'{keyword} ~ {predstr[0]}', data=df).fit()
        n = len(df)
        k = len(df.keys())
        MSE_full = sum(model_full.resid**2)/(n-k)

        r2 = []
        adjr2 = []
        aic = []
        bic = []
        cplist = []
        for stritem in predstr:
            model = smf.ols(f'{keyword} ~ {stritem}', data=df).fit()

            r2.append(model.rsquared)
            adjr2.append(model.rsquared_adj)
            aic.append(model.aic)
            bic.append(model.bic)

            p = len(stritem.split(' + ')) + 1
            cp = sum(model.resid**2)/MSE_full - (n - 2*p)
            cplist.append(cp)

        selection['Vars'] = varnum
        selection['R-Sq'] = r2
        selection['Adj R-Sq'] = adjr2
        selection['AIC'] = aic
        selection['BIC'] = bic
        selection['Mallows\'s Cp'] = cplist
        selection['Predictors'] = predstr

        selection = selection.sort_values(by=['Vars'], ascending=True)

        return selection
    
    def ForwardSelection(df, keyword, method='R_sq_adj'):
        '''
        The df input is the our dataframe for model selection.
        The keyword input is the name of the dependent variable.
        The method input is the measure for model selection:
        - AIC: 'aic'
        - BIC: 'bic'
        - Adjusted R Square: 'R_sq_adj'
        - Mallows's Cp: 'cp'
        The output of this model is a plot of the method value by
        the number of the predictors we choose.
        This function returns the final model by forward selection.
        '''
        
        selection = pd.DataFrame()
        predictors = list(df.keys())
        predictors.remove(keyword)

        subsets = [[]]

        for predictor in predictors:
            prev = copy.deepcopy(subsets)
            [k.append(predictor) for k in subsets]
            subsets.extend(prev)

        predstr = [' + '.join(i) for i in subsets if i]

        model_full = smf.ols(f'{keyword} ~ {predstr[0]}', data=df).fit()
        n = len(df)
        k = len(df.keys())
        MSE_full = sum(model_full.resid**2)/(n-k)

        measure_values = []
        step_values = []

        for predictor in predictors:
            modelstr = keyword + ' ~ ' + predictor
            model = smf.ols(f'{modelstr}', data=df).fit()

            if method == 'aic':
                measure_values.append(model.aic)
            elif method == 'bic':
                measure_values.append(model.bic)
            elif method == 'R_sq_adj':
                measure_values.append(model.rsquared_adj)
            elif method == 'cp':
                p = 2
                cp = sum(model.resid**2)/MSE_full - (n - 2*p)
                measure_values.append(cp)

        step_values.append(max(measure_values))
        best = predictors[measure_values.index(max(measure_values))]
        predictors.remove(best)
        keyword = keyword + ' ~ ' + best

        while predictors:

            measure_values = []

            for predictor in predictors:
                modelstr = keyword + ' + ' + predictor
                model = smf.ols(f'{modelstr}', data=df).fit()

                if method == 'aic':
                    measure_values.append(model.aic)
                elif method == 'bic':
                    measure_values.append(model.bic)
                elif method == 'R_sq_adj':
                    measure_values.append(model.rsquared_adj)
                elif method == 'cp':
                    p = len(modelstr.split(' + ')) + 1
                    cp = sum(model.resid**2)/MSE_full - (n - 2*p)
                    measure_values.append(cp)

            step_values.append(max(measure_values))
            best = predictors[measure_values.index(max(measure_values))]
            predictors.remove(best)
            keyword = keyword + ' + ' + best

        fig, ax = plt.subplots(figsize=(3,3))

        ax.plot(range(1, len(keyword.split(' + ')) + 1), step_values)
        ax.set_xlabel('Number of the predictors')
        ax.set_ylabel(method)

        plt.show()

        if method == 'R_sq_adj':
            best_model = ' + '.join(keyword.split(' + ')[:step_values.index(max(step_values))+1])
        else:
            best_model = ' + '.join(keyword.split(' + ')[:step_values.index(min(step_values))+1])

        return best_model
    
    def MLE(self):
        model = self.logit()
        return np.array(model.params)[1: ]
    
    def odds_ratio(self):
        model = self.logit()
        coef = np.array(model.params)
        odds_ratio = np.exp(coef)
        return np.array(odds_ratio)
    
    def odds(self):
        model = self.logit()
        b = np.array(model.params)
        y, X = self.linXy()
        f = np.dot(b, X.T)
        odds = np.exp(f)
        return odds
    
    def logit_probability(self):
        odds = self.odds()
        prob = [i/(1+i) for i in odds]
        return prob
        
    def deviance(self):
        model = self.logit()
        deviance = model.deviance
        return deviance
