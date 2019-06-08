import React, { Component } from "react";
import ReactMarkdown from "react-markdown";
import CodeBlock from "./CodeBlock";

const header = `


## Getting the Most out of the Imputer Classes: Part III
---
This tutorial is part III of a comprehensive overview of \`Autoimpute\` Imputers. It includes:  

1. Prepping Environment and Creating Data  
2. A Quick Review of the \`SingleImputer\`  
3. Issues with Single Imputation  
4. Multiple Imputation in \`Autoimpute\`  
5. \`MultipleImputer\` under the Hood  
6. Considerations during Multiple Imputation  

### 1. Prepping Environment and Creating Data
As with most tutorials, we begin by prepping our environment. Here, we import familiar packages for data analysis as well as the \`SingleImputer\` and \`MultipleImputer\` from \`Autoimpute\`. We also import visualization methods native to \`Autoimpute\` that help us visually understand the multiple imputation framework. Finally, we generate sample data, sticking with two variables, predictor **x** and response **y**. Only **y** has observations with missing values.


\`\`\`python
# imports for analyzing missing data
%matplotlib inline
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
from autoimpute.imputations import SingleImputer, MultipleImputer
from autoimpute.visuals import plot_imp_dists, plot_imp_boxplots, plot_imp_swarm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(context="talk", rc={'figure.figsize':(11.7,8.27)})

# helper functions used throughout this project
print_header = lambda msg: print(f"{msg}\\n{'-'*len(msg)}")

# seed to follow along
np.random.seed(654654)

# generate 400 data points
N = np.arange(400)

# helper function for this data
vary = lambda v: np.random.choice(np.arange(v))

# create correlated, random variables
a = 2
b = 1/2
eps = np.array([norm(0, vary(30)).rvs() for n in N])
y = (a + b*N + eps) / 100                         
x = (N + norm(10, vary(250)).rvs(len(N))) / 100
 
# 30% missing in y
y[binom(1, 0.3).rvs(len(N)) == 1] = np.nan

# collect results in a dataframe 
data_miss = pd.DataFrame({"y": y, "x": x})
sns.scatterplot(x="x", y="y", data=data_miss)
plt.show()
\`\`\`

`

const partTwo = `

### 2. A Quick Review of the SingleImputer
The dataset we create above is missing roughly 30% of the values in **y**. As we've seen in the first two tutorials, we can use the \`SingleImputer\` in \`Autoimpute\` to replace the missing values with plausible imputations. In the code below, we create a default instance of the \`SingleImputer\` and assign it the variable name \`si\`. Because **y** is numeric, \`si\` implements the predictive mean matching algorithm by default to generate imputations when we call its \`fit_transform\` method. 


\`\`\`python
# create an instance of the single imputer and impute the data
si = SingleImputer()
si_data_full = si.fit_transform(data_miss)

# print the results
print_header("Results from SingleImputer running PMM on column y one time")
conc = pd.concat([data_miss.head(20), si_data_full.head(20)], axis=1)
conc.columns = ["x", "y_orig", "x_imp", "y_imp"]
conc[["x", "y_orig", "y_imp"]]
\`\`\`

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:02<00:00, 2214.42draws/s]


    Results from SingleImputer running PMM on column y one time
    -----------------------------------------------------------


`

const partThreeFour = `

Next, we compare the first 20 records after imputation to the original dataset. As expected, the new dataset has no missing values. In the following code block, we print the index of the records in **y** that have been imputed. Because **x** had no missing data, no imputations were made. **y**, on the other hand, contains 124 imputations.


\`\`\`python
print_header("Index of imputed values in each column")
print(si.imputed_)
print()
print_header("Number of imputations in each column")
print([f"{k}: {len(v)}" for k, v in si.imputed_.items()])
\`\`\`

    Index of imputed values in each column
    --------------------------------------
    {'x': [], 'y': [0, 6, 9, 14, 16, 18, 20, 26, 29, 30, 41, 43, 47, 49, 54, 56, 59, 66, 67, 71, 73, 75, 80, 83, 86, 87, 88, 93, 98, 100, 116, 117, 118, 121, 122, 128, 132, 134, 138, 142, 146, 147, 149, 153, 159, 168, 171, 177, 178, 180, 182, 183, 185, 186, 187, 191, 192, 195, 199, 200, 201, 204, 209, 225, 228, 231, 233, 235, 236, 237, 242, 245, 247, 249, 250, 252, 259, 266, 269, 271, 272, 273, 276, 279, 284, 290, 296, 298, 302, 308, 311, 313, 318, 320, 321, 323, 325, 326, 327, 330, 332, 335, 339, 341, 346, 351, 353, 354, 358, 359, 363, 365, 368, 374, 379, 380, 382, 383, 384, 389, 392, 396, 398, 399]}
    
    Number of imputations in each column
    ------------------------------------
    ['x: 0', 'y: 124']


### 3. Issues with Single Imputation
After performing single imputation, we now have a complete dataset. We may feel that we are ready to deploy machine learning models because all the records are complete. This assumption, however, glosses over a key problem with single imputation.

Single imputation produces a set of single values for each missing observation in each imputed variable. In our case, the set represents "point estimates" for each missing observation within **y**. If we use a single-imputed dataset downstream in a machine learning model such as linear or logistic regression, the model is not be able to differentiate real values from imputed values and treats both the same. Unfortunately, this process disregards the uncertain nature that governs the true value of the each imputed record. **Instead, we should treat each missing value as a random variable**. Therefore, each missing value should be represented by a distribution of potentially viable imputations rather than a single point estimate. In contrast, observed values have no uncertainty, and therefore, their variance is naturally zero. We do not need to represent observed data using a distribution because we know each observed value with certainty. But the imputed value of missing data points could change if we run the imputation procedure again. Therefore, we should not treat missing data points the same as we treat observed data points, even after imputation takes place.

Whether imputed values change and by how much depends on the imputation method used and the relationship between features in the observed samples. We must consider these factors when assessing the performance of imputation algorithms in the mulitple imputation framework. What's important to remmeber now is the way in which we approach missing data points before and after imputation. Single imputation treats missing data points the same as it does observed data points, while multiple imputation treats missing observations as random variables governed by a distribution of possible imputed values. To retain the variance associated with a respective distribution, we must use multiple imputation instead of single imputation.

### 4. Multiple Imputation in Autoimpute
Multiple imputation strives to solve the issues with single imputation. When we peform multiple imputation, we impute the same dataset multiple times, and we store a separate copy of each imputed dataset. The difference between imputed values in each dataset depends on the structure of the observed data and the imputation algorithm used. Regardless, multiple imputation gives us a framework with which we can treat each missing record as a random variable. We can produce multiple imputations for each missing value and then assess the variance that results from mutliple imputations.

In the code below, we create an instance of the \`MultipleImputer\`, named \`mi\`. We then apply \`mi\` 5 times (the default in \`Autoimpute\`). We store the results in another variable, named \`mi_data_full\`. The variable represents a list of tuples, where the first value in each tuple is the imputation number (1-5 in this case) and the second value in each tuple is the imputed dataset for a given imputation. We then concatenate the **y** Series from each imputation and compare the results to the original dataset. We notice that the imputed values for **y** are often different between the 5 imputations. 


\`\`\`python
# create an instance of the multiple imputer and impute the data
mi = MultipleImputer(return_list=True)
mi_data_full = mi.fit_transform(data_miss)

# print the results
print_header("Results from MultipleImputer running PMM on column y five times")
imps = pd.concat([mi_data_full[i][1]["y"].to_frame() for i in range(len(mi_data_full))], axis=1)
conc = pd.concat([data_miss.head(20), imps.head(20)], axis=1)
conc.columns = ["x", "y_orig", "y_1", "y_2", "y_3", "y_4", "y_5"]
conc
\`\`\`

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:03<00:00, 1964.50draws/s]
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:03<00:00, 1974.05draws/s]
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:02<00:00, 2105.42draws/s]
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:02<00:00, 2150.44draws/s]
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:02<00:00, 2114.16draws/s]


    Results from MultipleImputer running PMM on column y five times
    ---------------------------------------------------------------



`

const distPlot = `


The differences between the imputed values demonstrate the uncertainty we have in the true value of each missing data point. We can visually depict the differences between each imputation using a number of helper methods native to \`Autoimpute\`. In the code sections below, we demonstrate four plots to help us understand the differences between the observed data and the full data after each imputation round.

#### Distribution Plot
The distribution plot shows the kernel density estimate of the observed data as well as each complete dataset after each imputation round. Note the distributions are not the same. The differences result from the fact that each missing data point generates different imputed values at each iteration.


\`\`\`python
plot_imp_dists(mi_data_full, mi, "y")
\`\`\`


`

const boxPlot = `

#### Box Plot
The boxplot also shows the observed distribution for **y** and the distribution after each imputation. These boxplots are another way to visualize the distribution of a variable and how that distribution changes when multiple imputations take place, each of which treats a missing data point as a random variable. 


\`\`\`python
plot_imp_boxplots(mi_data_full, mi, "y", side_by_side=True)
\`\`\`

`

const swarmPlot = `

#### Swarm Plot
The swarm plot helps visualize the actual "point estimate" imputed records take in each imputation round. It most clearly demonstrates the differences between each imputation iteration. Some imputed values are the same from iteration to iteration, but others are clearly distinct. For example, imputation round 1 and 5 (below) contain an imputed value at the the upper tail of the distribution of **y**, while imputation round 3 and 4 contain an imputed value at the bottom tail of the sitribution of **y**. 


\`\`\`python
plot_imp_swarm(mi_data_full, mi, "y")
\`\`\`


`

const partFiveSix = `

### 5. MultipleImputer under the Hood
In the sections above, we focused on the effect of the multiple imputation procedure. We skip over the implementation details because we want to emphasize how multiple imputation handles each missing record. We stress that missing records are now treated as random variables, and we demonstrate that each imputation round essentially draws a record from the distribution that governs each missing record. In this section, we examine how \`Autoimpute\` implements multiple imputation and how the \`MultipleImputer\` works under the hood. We begin by printing \`mi\` to the console.


\`\`\`python
print_header("Instance of the MultipleImputer")
mi
\`\`\`

    Instance of the MultipleImputer
    -------------------------------

    MultipleImputer(imp_kwgs=None, n=5, predictors='all', return_list=True,
            seed=None, strategy='default predictive', visit='default')



The \`MultipleImputer\` has some familiar arguments. Specifically, it includes the \`imp_kwgs\`, \`predictors\`, and \`strategy\` arguments that we go over in detail in part II These arguments appear in both the \`SingleImputer\` and \`MultipleImputer\` because the latter simply implements independent instances of the former for each of the \`n\` imputations it performs. Let's observe the \`statistics_\` property of the \`MultipleImputer\`.


\`\`\`python
print_header("Statistics property of the MultipleImputer")
mi.statistics_
\`\`\`

    Statistics property of the MultipleImputer
    ------------------------------------------

    {1: SingleImputer(copy=True, imp_kwgs=None, predictors={'x': 'all', 'y': 'all'},
            seed=None, strategy='default predictive', visit='default'),
     2: SingleImputer(copy=True, imp_kwgs=None, predictors={'x': 'all', 'y': 'all'},
            seed=None, strategy='default predictive', visit='default'),
     3: SingleImputer(copy=True, imp_kwgs=None, predictors={'x': 'all', 'y': 'all'},
            seed=None, strategy='default predictive', visit='default'),
     4: SingleImputer(copy=True, imp_kwgs=None, predictors={'x': 'all', 'y': 'all'},
            seed=None, strategy='default predictive', visit='default'),
     5: SingleImputer(copy=True, imp_kwgs=None, predictors={'x': 'all', 'y': 'all'},
            seed=None, strategy='default predictive', visit='default')}


The \`MultipleImputer\` creates a new instance of the \`SingleImputer\` class for each imputation. Therefore, the key of the \`statistics_\` dictionary represents the imputation round, and the value represents the \`SingleImputer\` that handles a given iteration. **The imp_kwgs and strategy arguments are the same for each iteration, but the predictors argument can change**. While we refrain from going into detail in this tutorial, we could pass a list to the \`predictors\` argument, and each imputation round would use the predictors in the respective position of the list. The \`imp_kwgs\` and \`strategy\` must be the same, however, to make the imputation model comparable across each imputation round.

The only new argument to familarize ourselves with is \`n\`: the number of imputation rounds we want to perform. We discuss considerations regarding the value of \`n\` in the next section. For now, note that \`n=1\` is the same as single imputation. As \`n\` approaches infinity, we should expect each missing data point to contain enough samples to construct its posterior distribution.

### 6. Considerations during Multiple Imputation

The multiple imputation procedure essentially performs single imputation numerous times to generate multiple copies of the same dataset with (potentially) different values for the originally missing records. While multiple imputation accounts for the uncertainty in imputations, the method comes with its own challenges that one must consider.

#### How many imputations?
The most obvious question that comes up during multiple imputation is how many imputations to perform. If we perform 1, we are essentially mimicking the behavior of the \`SingleImputer\`. If we perform more than 1, we should see some variance between each iteration. But how many do we need to perform for that variance to be "sufficient"? This question is a tough one to answer, and various literature covers the subject in more detail. \`Autoimpute\` defaults the number of imputations to 5. An end user can control the number of imputations through the **n** argument in the \`MultipleImputer\`. Note that the number of imputations generally increases the time it takes the \`Imputer\` to run. Additionally, setting **n** too high can consume a lot of memory, as the \`Imputer\` returns **n** datasets. To handle this, \`Autoimpute\` lazily evaluates imputations by default, although this behavior can be controlled through the \`return_list\` argument. Set it equal to \`True\` to return all imputations at once.

#### What can change between imputations?
To generate variance between imputations, the end user can change the way in which each imputation round proceeds. The end user cannot change the strategy or the imp_kwgs between iterations, as each algorithm needs to be the same in order to make imputations comparable accross imputation rounds. That being said, the end user can change the **predictors** from round to round. Additionally, the end user can specify a **visit sequence** that shuffles the order in which columns are imputed. While "left-to-right" is the only method supported right now, we plan to support shuffling soon.

#### How do we analyze multiply imputed data?
Throughout the tutorial, we note that muliple imputation is necessary to retain the variance that occurs from the imputation procedure. That being said, it's not clear at this point what the end user should do with 5 datasets that have different values. For instance, how should one analyze multiply imputed data in a supervised machine learning pipeline? Luckily, \`Autoimpute\` extends linear and logistic regression to apply to multiply imputed data. Two classes - the \`MiLinearRegression\` and \`MiLogisticRegression\` - take in a missing dataset, produce multiple imputations, and return a supervised machine learning model that pools parameters automatically and provides summary parameter diagnostics. These classes are the subject of a subsequent tutorial.


`

class ImputerIII extends Component {
    render() {
      return (
        <div className="imputer-III">
          <ReactMarkdown source={header} escapeHtml={false} renderers={{code: CodeBlock}} />
          <img alt="imputer-III-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/imputer/imputer-III-scatter.png"></img>
          <ReactMarkdown source={partTwo} escapeHtml={false} renderers={{code: CodeBlock}} />
          <table border="1" class="dataframe">
          <thead>
            <tr>
              <th>x</th>
              <th>y original</th>
              <th>y imputed</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0.661921</td>
              <td>NaN</td>
              <td>0.826949</td>
            </tr>
            <tr>
              <td>-0.190313</td>
              <td>0.130094</td>
              <td>0.130094</td>
            </tr>
            <tr>
              <td>-1.055594</td>
              <td>0.289676</td>
              <td>0.289676</td>
            </tr>
            <tr>
              <td>0.118157</td>
              <td>0.354103</td>
              <td>0.354103</td>
            </tr>
            <tr>
              <td>-0.149828</td>
              <td>-0.009910</td>
              <td>-0.009910</td>
            </tr>
            <tr>
              <td>-0.545609</td>
              <td>0.002967</td>
              <td>0.002967</td>
            </tr>
            <tr>
              <td>0.401257</td>
              <td>NaN</td>
              <td>0.311536</td>
            </tr>
            <tr>
              <td>0.609314</td>
              <td>0.058539</td>
              <td>0.058539</td>
            </tr>
            <tr>
              <td>-1.053159</td>
              <td>0.063187</td>
              <td>0.063187</td>
            </tr>
            <tr>
              <td>-0.015037</td>
              <td>NaN</td>
              <td>0.118795</td>
            </tr>
            <tr>
              <td>-0.134157</td>
              <td>0.086663</td>
              <td>0.086663</td>
            </tr>
            <tr>
              <td>-0.221629</td>
              <td>0.303863</td>
              <td>0.303863</td>
            </tr>
            <tr>
              <td>0.525348</td>
              <td>0.092465</td>
              <td>0.092465</td>
            </tr>
            <tr>
              <td>0.900651</td>
              <td>0.068560</td>
              <td>0.068560</td>
            </tr>
            <tr>
              <td>0.801863</td>
              <td>NaN</td>
              <td>0.139153</td>
            </tr>
            <tr>
              <td>0.787607</td>
              <td>0.029584</td>
              <td>0.029584</td>
            </tr>
            <tr>
              <td>-1.613273</td>
              <td>NaN</td>
              <td>0.063187</td>
            </tr>
            <tr>
              <td>0.502887</td>
              <td>0.417373</td>
              <td>0.417373</td>
            </tr>
            <tr>
              <td>1.596454</td>
              <td>NaN</td>
              <td>0.864781</td>
            </tr>
            <tr>
              <td>1.106348</td>
              <td>0.068833</td>
              <td>0.068833</td>
            </tr>
          </tbody>
        </table>
        <ReactMarkdown source={partThreeFour} escapeHtml={false} renderers={{code: CodeBlock}} />
        <table border="1" class="dataframe">
          <thead>
            <tr>
              <th>x</th>
              <th>y original</th>
              <th>y imp 1</th>
              <th>y imp 2</th>
              <th>y imp 3</th>
              <th>y imp 4</th>
              <th>y imp 5</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0.661921</td>
              <td>NaN</td>
              <td>0.058539</td>
              <td>0.826949</td>
              <td>0.826949</td>
              <td>0.417373</td>
              <td>0.029584</td>
            </tr>
            <tr>
              <td>-0.190313</td>
              <td>0.130094</td>
              <td>0.130094</td>
              <td>0.130094</td>
              <td>0.130094</td>
              <td>0.130094</td>
              <td>0.130094</td>
            </tr>
            <tr>
              <td>-1.055594</td>
              <td>0.289676</td>
              <td>0.289676</td>
              <td>0.289676</td>
              <td>0.289676</td>
              <td>0.289676</td>
              <td>0.289676</td>
            </tr>
            <tr>
              <td>0.118157</td>
              <td>0.354103</td>
              <td>0.354103</td>
              <td>0.354103</td>
              <td>0.354103</td>
              <td>0.354103</td>
              <td>0.354103</td>
            </tr>
            <tr>
              <td>-0.149828</td>
              <td>-0.009910</td>
              <td>-0.009910</td>
              <td>-0.009910</td>
              <td>-0.009910</td>
              <td>-0.009910</td>
              <td>-0.009910</td>
            </tr>
            <tr>
              <td>-0.545609</td>
              <td>0.002967</td>
              <td>0.002967</td>
              <td>0.002967</td>
              <td>0.002967</td>
              <td>0.002967</td>
              <td>0.002967</td>
            </tr>
            <tr>
              <td>0.401257</td>
              <td>NaN</td>
              <td>0.766211</td>
              <td>0.766211</td>
              <td>0.469095</td>
              <td>0.030568</td>
              <td>0.932830</td>
            </tr>
            <tr>
              <td>0.609314</td>
              <td>0.058539</td>
              <td>0.058539</td>
              <td>0.058539</td>
              <td>0.058539</td>
              <td>0.058539</td>
              <td>0.058539</td>
            </tr>
            <tr>
              <td>-1.053159</td>
              <td>0.063187</td>
              <td>0.063187</td>
              <td>0.063187</td>
              <td>0.063187</td>
              <td>0.063187</td>
              <td>0.063187</td>
            </tr>
            <tr>
              <td>-0.015037</td>
              <td>NaN</td>
              <td>0.071464</td>
              <td>0.071464</td>
              <td>0.208594</td>
              <td>0.256477</td>
              <td>0.134639</td>
            </tr>
            <tr>
              <td>-0.134157</td>
              <td>0.086663</td>
              <td>0.086663</td>
              <td>0.086663</td>
              <td>0.086663</td>
              <td>0.086663</td>
              <td>0.086663</td>
            </tr>
            <tr>
              <td>-0.221629</td>
              <td>0.303863</td>
              <td>0.303863</td>
              <td>0.303863</td>
              <td>0.303863</td>
              <td>0.303863</td>
              <td>0.303863</td>
            </tr>
            <tr>
              <td>0.525348</td>
              <td>0.092465</td>
              <td>0.092465</td>
              <td>0.092465</td>
              <td>0.092465</td>
              <td>0.092465</td>
              <td>0.092465</td>
            </tr>
            <tr>
              <td>0.900651</td>
              <td>0.068560</td>
              <td>0.068560</td>
              <td>0.068560</td>
              <td>0.068560</td>
              <td>0.068560</td>
              <td>0.068560</td>
            </tr>
            <tr>
              <td>0.801863</td>
              <td>NaN</td>
              <td>0.029584</td>
              <td>0.139153</td>
              <td>0.139153</td>
              <td>0.736597</td>
              <td>0.549241</td>
            </tr>
            <tr>
              <td>0.787607</td>
              <td>0.029584</td>
              <td>0.029584</td>
              <td>0.029584</td>
              <td>0.029584</td>
              <td>0.029584</td>
              <td>0.029584</td>
            </tr>
            <tr>
              <td>-1.613273</td>
              <td>NaN</td>
              <td>0.289676</td>
              <td>0.063187</td>
              <td>0.424593</td>
              <td>0.235977</td>
              <td>0.424593</td>
            </tr>
            <tr>
              <td>0.502887</td>
              <td>0.417373</td>
              <td>0.417373</td>
              <td>0.417373</td>
              <td>0.417373</td>
              <td>0.417373</td>
              <td>0.417373</td>
            </tr>
            <tr>
              <td>1.596454</td>
              <td>NaN</td>
              <td>0.315037</td>
              <td>0.776069</td>
              <td>0.088169</td>
              <td>0.458626</td>
              <td>1.192725</td>
            </tr>
            <tr>
              <td>1.106348</td>
              <td>0.068833</td>
              <td>0.068833</td>
              <td>0.068833</td>
              <td>0.068833</td>
              <td>0.068833</td>
              <td>0.068833</td>
            </tr>
          </tbody>
        </table>
        <ReactMarkdown source={distPlot} escapeHtml={false} renderers={{code: CodeBlock}} />
        <img alt="imputer-III-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/imputer/imputer-III-dist.png"></img>
        <ReactMarkdown source={boxPlot} escapeHtml={false} renderers={{code: CodeBlock}} />
        <img alt="imputer-III-box" src="https://kearnz.github.io/autoimpute-tutorials/img/imputer/imputer-III-box.png"></img>
        <ReactMarkdown source={swarmPlot} escapeHtml={false} renderers={{code: CodeBlock}} />
        <img alt="imputer-III-swarm" src="https://kearnz.github.io/autoimpute-tutorials/img/imputer/imputer-III-swarm.png"></img>
        <ReactMarkdown source={partFiveSix} escapeHtml={false} renderers={{code: CodeBlock}} />
        </div>
        );
    }
  }
   
  export default ImputerIII;