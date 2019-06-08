import React, { Component } from "react";
import ReactMarkdown from "react-markdown";
import CodeBlock from "./CodeBlock";

// HEADER TO START THE TUTORIAL
const inputHeader01 = `
## Using Autoimpute to Compare Imputation Methods

This tutorial examimes the effect of imputation methods from the \`Autoimpute\` package. The tutorial includes:  

1. Generating Data  
2. Imputation Methods  
3. Impact of imputation on Covariance and Correlation  

### 1. Generating Data
* In the section below, we generate two variables, **x** and **y**, that are positively correlated but display heteroscedasticity.  
* Thus, **x** and **y** get larger (or smaller) together. As a result, the variance between the two variables is not constant.     
* We then introduce 30% missingness within **y**. **x** remains completely observed.  
* The code on the left generates the data and creates a function to visualize the data.  
* The code on the right displays the resulting dataframe and its associated plot.   
`

// FIRST CODE BLOCK
const inputCode1 = `
\`\`\`python
'''Simulating data and defining a joint plot b/w two variables'''

# plotting specification and imports needed
%matplotlib inline
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# seed to follow along
np.random.seed(654654)

# generate 1500 data points
N = np.arange(1500)

# create correlated, heteroskedastic random variables
a = 0
b = 1
eps = np.array([norm(0, n).rvs() for n in N])
y = (a + b*N + eps) / 100                         
x = (N + norm(10, 10).rvs(len(N))) / 100
 
# 30% missingness created artificially
y[binom(1, 0.3).rvs(len(N)) == 1] = np.nan

# collect results in a dataframe 
data_het_miss = pd.DataFrame({"y": y, "x": x})

# create a scatter plot function to display the results of our dataframe
def scatter_dists(data, x="x", y="y", a=0.5, joints_color="navy", 
                  markers="o", marginals=dict(rug=True, kde=True)):

    sns.set(context="talk")
    joint_kws = dict(
        facecolor=joints_color,
        edgecolor=joints_color,
        marker=markers
    )
    sns.jointplot(
        x=x, y=y, data=data, alpha=a, height=8.27,
        joint_kws=joint_kws, marginal_kws=marginals
    )
\`\`\`
`

const df_header01 = `
\`\`\`python
data_het_miss.head(7)
\`\`\`
`

const df_scatter01 = `
\`\`\`python
scatter_dists(data_het_miss)
\`\`\`
`

const meanImp = `

### 2. Imputation Methods
In this section, we examine what happens under the hood when we use each imputation method. To do so, we use a **visualization method native to the Autoimpute package** that imputes a dataset via single imputation and plots the resulting scatter plot between two features within the data. In this case, **x** is our predictor, and **y** is our imputed variable. The imputations appear in red, while the observed data points are navy. The function plots the marginal distribution of each feature as well. We note **the differences between the marginal distributions in subsequent plots and the fully observed plot above**.


\`\`\`python
from autoimpute.imputations import SingleImputer
from autoimpute.visuals import plot_imp_scatter
\`\`\`

#### a) Mean imputation using observed mean of Y
Mean imputation is quick and effective, but it heavily distorts both the relationship between **x** and **y** and the distribution of **y** itself. When we impute with the mean, the distribution of **y** becomes almost bimodal. The density of the distribution shifts to the mean, creating an artificial peak that would not have existed had we observed **y**. Mean imputation also completely disregards the structure of the data and the relationship between **x** and **y**. This consequence is obvious, given that mean imputation is a univariate method.


\`\`\`python
plot_imp_scatter(data_het_miss, "x", "y", "mean")
\`\`\`

`

const normImp = `

#### b) Imputation with draws from Random Normal distribution of Y:
We get more realistic imputations by taking random draws from a normal distribution with mean and variance from observed **y**. The imputations below capture the variance in **y**. That being said, they still fail to take into consideration the relationship between **y** and **x**, and the imputations don't consider **where **y** varies** (i.e. at larger values of **x**). 


\`\`\`python
plot_imp_scatter(data_het_miss, "x", "y", "norm")
\`\`\`

`

const linearImp = `

#### c) Imputation from Linear Interpolation
Linear interpolation provides the most realistic imputations we've seen so far, but looks can be deceiving. Linear Interpolation **still considers **y** on its own. Had the data been shuffled, imputations would be scattered incorrectly.** Thus, linear interpolation not only disregards **x** but is also sensitive to the **row position** of the data in **y** instead of its actual structure. While this may make sense for time series data, it does not in this situation.


\`\`\`python
plot_imp_scatter(data_het_miss, "x", "y", "interpolate")
\`\`\`

`

const lsImp = `

#### d) Imputation using predictions from least squares linear regression
As with mean imputation, linear regression imputation does not capture variance whatsoever because all imputations are considered "best fit" point estimates. As a result, all imputations lie along a line. Linear regression imputation does improve upon mean imputation by capturing the correlation between **x** and **y**, but linear regression imputation artificially deflates variance and inflates the relationship between features.

\`\`\`python
plot_imp_scatter(data_het_miss, "x", "y", "least squares")
\`\`\`

`

const stochImp = `

#### e) Imputation using predictions from least squares + random draw from distribution of residuals 
Linear regression with stochastic error is the best imputation we've seen yet. As with linear regression, it captures the correlation between the features, and it accurately depicts the variance between them as well. Unfortunately, it has the same problem as norm imputation. It does not consider where and how **y** varies with **x**. It considers total variance only, and it assumes variance is constant at all points.


\`\`\`python
plot_imp_scatter(data_het_miss, "x", "y", "stochastic")
\`\`\`


`

const pmmImp = `

#### f) Predictive Mean Matching Imputation - Nearest 5 Neighbors
Finally, PMM is **the optimal and most flexible method for our dataset.** PMM is more complex, so let's review its steps:  

1. Fit a linear regression on observed data. Keep the predictions and the regression coefficients handy.  
2. Fit a bayesian model on the observed data, using the coefficients from step 1 as inputs to the priors of the bayesian model.  
    - Note: passing coefficients to priors is optional.  
    - We can use uninformative priors instead, but convergence may be much slower.  
    - Passing coefficients may speed up MCMC methods to generate posterior predictive distributions.  
    - If the linear model is poor, however, the coefficients may do more harm than good.  
3. Make a random draw from the **posterior predictive distribution** of the coefficients to produce a new set of coefficients.  
    - This is a random draw using the means and the covariance matrix of the of the regression coefficients.  
    - The random draw creates sufficient variability and is the **first step to generalise to multiple imputation**.  
4. Using the coefficents produced from the random draw, predict missing values for **y** from the corresonding features **x**.  
5. For each predicted **y**, find the **n** closest predictions **y_pred** from the original observed linear regression.  
6. Take a random draw from the actual, observed **y** values that correspond to each **y_pred**, and impute.  

The steps above summarize predictive mean matching. It borrows ideas from linear models to capture the correlation between features, and it borrows strategies from hot-deck methods like knn by sampling from "near neighbors". As a result, we get realistic imputations, and we obey the **covariance structure** of the dataset. Additionally, we preserve the **heteroskedasticity of the data**. There are downsides to PMM, but they are not in scope for this tutorial.


\`\`\`python
plot_imp_scatter(data_het_miss, "x", "y", "pmm")
\`\`\`

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:03<00:00, 1505.87draws/s]



`

const covCorrHeader = `

### 3. Impact of each Strategy on Covariance and Correlation
Although we can assess accuracy if we conduct a simulation (as we know the true values of **y** where **y** is missing), this assessment is misguided. The goal of imputation is to **preserve the structure of the data**, not correctly guess the true value of the missing points. With real-world data, we often do not have and never will have "true" values. Therefore, accuracy is not the best objective. Instead, we want to preserve the covariance (and thus correlation) between features after imputation takes place. If we preserve the data's structure, we can more safely assume that our imputations are plausible. Now, this objective assumes we are working with data that is **at least Missing at Random (MAR)**, as do all the methods utilized in this tutorial. If data is missing not at random, preservation may actually indicate poor imputation.


\`\`\`python
def cov_corr(df):
    cov = df.cov()
    corr = df.corr()
    cov_corr_df = pd.concat([cov, corr], axis=1)
    cov_corr_df.columns = columns=["cov_x", "cov_y", "corr_x", "corr_y"]
    return cov_corr_df
\`\`\`

#### Observed Only
Because we assume data is missing at random or missing completely at random, we can use the observed points to determine the initial correlation and covariance of the data. We calculate these matrices after using each imputation method. 

`

const meanCovCorr = `

#### a) Mean Imputation
Mean imputation depresses the covariance and correlation between features. This should make sense intuitively, as mean imputation is univariate. It ignores **x** all together, and it even ignores variance in **y**. As a result, covariance and correlation drop substantially.


\`\`\`python
mean_ft = SingleImputer(strategy="mean").fit_transform(data_het_miss)
cov_corr(mean_ft)
\`\`\`

`

const normCovCorr = `

#### b) Norm Imputation
Norm imutation draws random values from a normal distribution of observed **y**. Because values are drawn at random, and because draws do not consider **x** whatsoever, the correlation and covariance drops. This should also make sense intuitively. Random draws should never help correlation or covariance, or else randomness would not be true.


\`\`\`python
norm_ft = SingleImputer(strategy="norm").fit_transform(data_het_miss)
cov_corr(norm_ft)
\`\`\`

`

const linearCovCorr = `

#### c) Linear Interpolation Imputation
Linear interpoloation does a good job preserving the structure of the data, although it artificially inflates both to a minor degree. This inflation occurs because linear interpolation has no stochastic component.


\`\`\`python
linear_ft = SingleImputer(strategy="interpolate").fit_transform(data_het_miss)
cov_corr(linear_ft)
\`\`\`

`

const lsCovCorr = `

#### d) Least Squares Imputation
Least squares imputes using predictions from the line of best fit. There is no stochastic component, and the imputations respect the slope of the fitted line. Therefore, correlation and covariance are artificially inflated to unnaceptable levels.


\`\`\`python
lm_ft = SingleImputer(strategy="least squares").fit_transform(data_het_miss)
cov_corr(lm_ft)
\`\`\`

`

const stochCovCorr = `

#### e) Stochastic Imputation
Stochastic imputation is least squares imputation combined with norm imputation, where the mean and variance of the normal distribution are now the mean and variance of the errors from the regression fit. The stochastic component brings the correlation and covariance back to acceptable levels.


\`\`\`python
stoch_ft = SingleImputer(strategy="stochastic").fit_transform(data_het_miss)
cov_corr(stoch_ft)
\`\`\`


`

const pmmCovCorr = `

#### f) Predictive Mean Matching Imputation
Predictive mean matching does a great job respecting the correlation and covariance of the data. Each measure is a bit lower than the observed metrics, but the difference is simply be related to the random selections of one PMM iteration rather than a character flaw of the imputation method itself. The matrices below confirm the good fit we visualized above.


\`\`\`python
pmm_ft = SingleImputer(strategy="pmm").fit_transform(data_het_miss)
cov_corr(pmm_ft)
\`\`\`

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:04<00:00, 1331.54draws/s]
    The acceptance probability does not match the target. It is 0.8898599588203708, but should be close to 0.8. Try to increase the number of tuning steps.

`

class Comparing extends Component {
    render() {
      return (
          <div className="comparing-imputation-methods">
            <div className="cim-header-01">
                <ReactMarkdown source={inputHeader01} escapeHtml={false} renderers={{code: CodeBlock}} />
            </div>
            <div className="cim-code-1">
                <ReactMarkdown source={inputCode1} escapeHtml={false} renderers={{code: CodeBlock}} />
            </div>
            <div className="cim-table-code-1">
              <ReactMarkdown source={df_header01} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" className="dataframe">
                <thead>
                    <tr>
                    <th>x</th>
                    <th>y</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                    <td>0.295288</td>
                    <td>0.000000</td>
                    </tr>
                    <tr>
                    <td>0.086210</td>
                    <td>0.002255</td>
                    </tr>
                    <tr>
                    <td>0.227369</td>
                    <td>0.003869</td>
                    </tr>
                    <tr>
                    <td>0.194216</td>
                    <td>0.046370</td>
                    </tr>
                    <tr>
                    <td>0.094630</td>
                    <td>0.081440</td>
                    </tr>
                    <tr>
                    <td>0.292320</td>
                    <td>NaN</td>
                    </tr>
                    <tr>
                    <td>0.198131</td>
                    <td>0.067222</td>
                    </tr>
                </tbody>
              </table>
            </div>
            <div className="cim-df-scatter-01">
                <ReactMarkdown source={df_scatter01} escapeHtml={false} renderers={{code: CodeBlock}} />
                <img alt="scatter01" src="https://kearnz.github.io/autoimpute-tutorials/img/comparing/cim-df-scatter01.png"></img>
            </div>
            <div>
              <ReactMarkdown source={meanImp} escapeHtml={false} renderers={{code: CodeBlock}} />
              <img alt="cim-mean" src="https://kearnz.github.io/autoimpute-tutorials/img/comparing/cim-mean-imp.png"></img>
              <ReactMarkdown source={normImp} escapeHtml={false} renderers={{code: CodeBlock}} />
              <img alt="cim-norm" src="https://kearnz.github.io/autoimpute-tutorials/img/comparing/cim-norm-imp.png"></img>
              <ReactMarkdown source={linearImp} escapeHtml={false} renderers={{code: CodeBlock}} />
              <img alt="cim-linear" src="https://kearnz.github.io/autoimpute-tutorials/img/comparing/cim-linear-imp.png"></img>
              <ReactMarkdown source={lsImp} escapeHtml={false} renderers={{code: CodeBlock}} />
              <img alt="cim-ls" src="https://kearnz.github.io/autoimpute-tutorials/img/comparing/cim-ls-imp.png"></img>
              <ReactMarkdown source={stochImp} escapeHtml={false} renderers={{code: CodeBlock}} />
              <img alt="cim-stoch" src="https://kearnz.github.io/autoimpute-tutorials/img/comparing/cim-stoch-imp.png"></img>
              <ReactMarkdown source={pmmImp} escapeHtml={false} renderers={{code: CodeBlock}} />
              <img alt="cim-pmm" src="https://kearnz.github.io/autoimpute-tutorials/img/comparing/cim-pmm-imp.png"></img>
              <ReactMarkdown source={covCorrHeader} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" class="dataframe">
                <thead>
                  <tr>
                    <th>feature</th>
                    <th>covariance x</th>
                    <th>covariance y</th>
                    <th>correlation x</th>
                    <th>correlation y</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>x</td>
                    <td>18.782294</td>
                    <td>16.859482</td>
                    <td>1.000000</td>
                    <td>0.410591</td>
                  </tr>
                  <tr>
                    <td>y</td>
                    <td>16.859482</td>
                    <td>90.140563</td>
                    <td>0.410591</td>
                    <td>1.000000</td>
                  </tr>
                </tbody>
              </table>
              <ReactMarkdown source={meanCovCorr} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" class="dataframe">
                <thead>
                  <tr>
                    <th>feature</th>
                    <th>covariance x</th>
                    <th>covariance y</th>
                    <th>correlation x</th>
                    <th>correlation y</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>x</td>
                    <td>18.782294</td>
                    <td>12.023207</td>
                    <td>1.000000</td>
                    <td>0.346017</td>
                  </tr>
                  <tr>
                    <td>y</td>
                    <td>12.023207</td>
                    <td>64.283030</td>
                    <td>0.346017</td>
                    <td>1.000000</td>
                  </tr>
                </tbody>
              </table>
              <ReactMarkdown source={normCovCorr} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" class="dataframe">
                <thead>
                  <tr>
                    <th>feature</th>
                    <th>covariance x</th>
                    <th>covariance y</th>
                    <th>correlation x</th>
                    <th>correlation y</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>x</td>
                    <td>18.782294</td>
                    <td>12.190499</td>
                    <td>1.000000</td>
                    <td>0.298202</td>
                  </tr>
                  <tr>
                    <td>y</td>
                    <td>12.190499</td>
                    <td>88.976049</td>
                    <td>0.298202</td>
                    <td>1.000000</td>
                  </tr>
                </tbody>
              </table>
              <ReactMarkdown source={linearCovCorr} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" class="dataframe">
                <thead>
                  <tr>
                    <th>feature</th>
                    <th>covariance x</th>
                    <th>covariance y</th>
                    <th>correlation x</th>
                    <th>correlation y</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>x</td>
                    <td>18.782294</td>
                    <td>16.995538</td>
                    <td>1.000000</td>
                    <td>0.428826</td>
                  </tr>
                  <tr>
                    <td>y</td>
                    <td>16.995538</td>
                    <td>83.629567</td>
                    <td>0.428826</td>
                    <td>1.000000</td>
                  </tr>
                </tbody>
              </table>
              <ReactMarkdown source={lsCovCorr} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" class="dataframe">
                <thead>
                <tr>
                  <th>feature</th>
                  <th>covariance x</th>
                  <th>covariance y</th>
                  <th>correlation x</th>
                  <th>correlation y</th>
                </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>x</td>
                    <td>18.782294</td>
                    <td>16.929476</td>
                    <td>1.000000</td>
                    <td>0.471275</td>
                  </tr>
                  <tr>
                    <td>y</td>
                    <td>16.929476</td>
                    <td>68.705310</td>
                    <td>0.471275</td>
                    <td>1.000000</td>
                  </tr>
                </tbody>
              </table>
              <ReactMarkdown source={stochCovCorr} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" class="dataframe">
                <thead>
                  <tr>
                    <th>feature</th>
                    <th>cov_x</th>
                    <th>cov_y</th>
                    <th>corr_x</th>
                    <th>corr_y</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>x</td>
                    <td>18.782294</td>
                    <td>17.246360</td>
                    <td>1.000000</td>
                    <td>0.422592</td>
                  </tr>
                  <tr>
                    <td>y</td>
                    <td>17.246360</td>
                    <td>88.675623</td>
                    <td>0.422592</td>
                    <td>1.000000</td>
                  </tr>
                </tbody>
              </table>
              <ReactMarkdown source={pmmCovCorr} escapeHtml={false} renderers={{code: CodeBlock}} />
              <table border="1" class="dataframe">
                <thead>
                  <tr>
                    <th>feature</th>
                    <th>cov_x</th>
                    <th>cov_y</th>
                    <th>corr_x</th>
                    <th>corr_y</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>x</td>
                    <td>18.782294</td>
                    <td>15.999986</td>
                    <td>1.000000</td>
                    <td>0.380924</td>
                  </tr>
                  <tr>
                    <td>y</td>
                    <td>15.999986</td>
                    <td>93.932024</td>
                    <td>0.380924</td>
                    <td>1.000000</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
      );
    }
  }
   
  export default Comparing;