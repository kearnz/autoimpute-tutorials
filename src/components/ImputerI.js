import React, { Component } from "react";
import ReactMarkdown from "react-markdown";
import CodeBlock from "./CodeBlock";

const header = `
## Getting the Most out of the Imputer Classes: Part I
---
This tutorial is part I of a comprehensive overview of \`Autoimpute\` Imputers. It includes:  

1. Motivation for Imputation in the First Place  
2. The Design Considerations behind Autoimpute Imputers  

### 1. Motivation for Imputation in the First Place
Let's revisit why multiple imputation is necessary. A user wants to perform analysis on a dataset using some sort of **analysis model** such as linear regression or logistic regression. The dataset of interest contains one or more predictors, **X**, and some response **y**. The analysis model produces a function that best explains the relationship between **X** and **y**. Let's generate some sample data below. To keep things simple, our data set contains just one predictor, **x**, and a response **y**.
`

const codePlot = `

\`\`\`python
# imports
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sns.set(context="talk", rc={'figure.figsize':(11.7,8.27)})

# helper functions used throughout this project
print_header = lambda msg: print(f"{msg}\\n{'-'*len(msg)}")

# seed to follow along
np.random.seed(654654)

# generate 1500 data points
N = np.arange(1500)

# helper function for this data
vary = lambda v: np.random.choice(np.arange(v))

# create correlated, random variables
a = 2
b = 1/2
eps = np.array([norm(0, vary(50)).rvs() for n in N])
y = (a + b*N + eps) / 100                         
x = (N + norm(10, vary(250)).rvs(len(N))) / 100
 
# 20% missing in x, 30% missing in y
x[binom(1, 0.2).rvs(len(N)) == 1] = np.nan
y[binom(1, 0.3).rvs(len(N)) == 1] = np.nan

# collect results in a dataframe 
data_miss = pd.DataFrame({"y": y, "x": x})
sns.scatterplot(x="x", y="y", data=data_miss)
plt.show()
\`\`\`

`

const partOne = `
The plot suggests a linear relationship may exist between **x** and **y**. Let's fit a linear model to estimate that relationship.


\`\`\`python
from sklearn.linear_model import LinearRegression

# prep for regression
X = data_miss.x.values.reshape(-1, 1) # reshape because one feature only
y = data_miss.y
lm = LinearRegression()

# try to fit the model
print_header("Fitting linear model to estimate relationship between X and y")
try:
    lm.fit(X, y)
except ValueError as ve:
    print(f"{ve.__class__.__name__}: {ve}")
\`\`\`

    Fitting linear model to estimate relationship between X and y
    -------------------------------------------------------------
    ValueError: Input contains NaN, infinity or a value too large for dtype('float64').


#### What Happened?
\`sklearn\` threw a ValueError when we tried to fit a linear regression. The error occurred because our **dataset has missing data**. In our case, 20% of observations are missing in our predictor, and 30% of observations are missing in our response. \`sklearn\` cannot fit the analysis model unless our dataset is complete! That's no good - we can't model the relationship in our data, nor can we make predictions when new data arrives.

#### What do we Do?
In order to proceed, we **need to handle the missing data in some way**. One option is simply removing records with missing data. This strategy allows the analysis model to run, but it may negatively affect the inference from that model. For now, we'll recommend against dropping missing records.

If we don't drop missing records, **then we must impute them**. Imputing data means coming up with plausible values to fill in missing records, which we must do to enable our analysis model to run. **Performing imputations is the primary concern of Autoimpute**. The next section introduces Autoimpute Imputers and familiarizes the reader with Autoimpute's package design.

`

const partTwoOne = `

### 2. The Design Considerations behind Autoimpute Imputers
We designed Autoimpute Imputers with three goals in mind:  
* **Make Imputation Easy**. Imputation can be done in one line of code.  
* **Make Imputation Familiar to Python Users.** Autoimpute Imputers follows the design patterns of sklearn.  
* **Make Imputation Flexible**. Use an Imputer's default arguments or fine-tune Imputers case-by-case.  

Let's explore each objective. We'll use the same dataset as above and stick with the \`SingleImputer\` for now. The \`MutlipleImputer\` extends the \`SingleImputer\` and therefore contains all the arguments that the \`SingleImputer\` does. Therefore, we will explain design considerations using the \`SingleImputer\`. In Part III of this series, we'll address additional arguments of and considerations for the \`MultipleImputer\`.

#### 2.1. Make Imputation Easy

As promised, we can impute a dataset with exactly one line of code. In the code section below, we'll demonstrate the one line of code that imputes all missing data in a dataset. First, we'll observe how many records are missing. Then, we'll perform imputation and verify that missing records have been imputed.


\`\`\`python
# amount of missing data before imputation
print_header("Amount of data missing before imputation takes place")
pd.DataFrame(data_miss.isnull().sum(), columns=["records missing"]).T
\`\`\`

    Amount of data missing before imputation takes place
    ----------------------------------------------------

`

const partTwoTwo = `

\`\`\`python
from autoimpute.imputations import SingleImputer
print_header("Imputing missing data in one line of code with the default SingleImputer")
data_imputed_once = SingleImputer().fit_transform(data_miss)
print("Imputation Successful!")
\`\`\`

    Imputing missing data in one line of code with the default SingleImputer
    ------------------------------------------------------------------------


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:03<00:00, 1912.67draws/s]
    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [σ, beta, alpha]
    Sampling 4 chains: 100%|██████████| 6000/6000 [00:03<00:00, 1722.73draws/s]


    Imputation Successful!



\`\`\`python
# amount of missing data before imputation
print_header("Amount of data missing after imputation takes place")
pd.DataFrame(data_imputed_once.isnull().sum(), columns=["records missing"]).T
\`\`\`

    Amount of data missing after imputation takes place
    ---------------------------------------------------


`

const partTwoThree = `

### 2.2 Make Imputation Familiar to Python Users
Autoimpute follows \`sklearn\` API design. This design choice comes with a number of benefits:  
* If you are familiar with \`sklearn\`, there is essentially no learning curve to start using \`Autoimpute\`.  
* Imputers inherit from sklearn's BaseEstimator and TransformerMixin and leverage methods from these Parent classes.  
* Imputers can \`fit_transform\` the same dataset, or \`fit\` and \`Imputer\` to \`transform\` new data with the same features.  
* Imputers fit in sklearn Pipelines (although Autoimpute includes ML models designed for multiply imputed data).  

The code segments below demonstrate Autoimpute's ease of use and familiarity.

First, Autoimpute Imputers inherit from \`BaseEstimator\` and \`TransformerMixin\` classes from \`sklearn\`.


\`\`\`python
print_header("Parent Classes to SingleImputer and MultipleImpuer")
print(list(map(lambda cls_: cls_.__name__, SingleImputer().__class__.__bases__)))
\`\`\`

    Parent Classes to SingleImputer and MultipleImpuer
    --------------------------------------------------
    ['BaseImputer', 'BaseEstimator', 'TransformerMixin']


As with \`sklearn\` \`Transformers\`, Autoimpute Imputers set smart defaults for all its arguments. Imputers leverage the \`__repr__\` special method inherited from the \`BaseEstimator\`, so users can quickly examine the default values an \`Imputer\` sets.


\`\`\`python
SingleImputer()
\`\`\`




    SingleImputer(copy=True, imp_kwgs=None, predictors='all', seed=None,
           strategy='default predictive', visit='default')



Because Autoimpute Imputers are valid \`sklearn\` \`Transformers\`, they implement the \`fit\`, \`transform\`, and \`fit_transform\` methods, which should be familiar to anyone who has preprocessed data using \`sklearn\`. The \`fit\` step returns an instance of the \`Imputer\` class, and the \`transform\` step returns a **transformed dataset**. Imputers can fit and transform a dataset in one go by using \`fit_transform\`. We demonstrate this process below, transforming our missing dataset using mean imputation.


\`\`\`python
print_header("Original dataset with missing values")
data_miss.head(10)
\`\`\`

    Original dataset with missing values
    ------------------------------------



`

const partTwoFour = `

\`\`\`python
print_header("Transforming the missing dataset with mean imputation")
imputer = SingleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data_miss)
data_imputed.head(10)
\`\`\`

    Transforming the missing dataset with mean imputation
    -----------------------------------------------------


`

const partTwoFive = `

The transformed dataset contains imputations in place of previously missing values. Here, we used mean imputation, although we offer many imputation strategies that may be more appropriate. We show these strategies in Part II of this series. We can **easily retrieve the index of the the imputed values** should we want to assess the imputations themselves. The index of imputed values live in the Imputer's \`imputed_\` attribute. As with \`sklearn\`, public attributes of Imputers contain an underscore suffix.

The \`imputed_\` attribute returns a dictionary, where each key is a column and its value is a list with the index of each imputation for that column. These indices represent where data was originally missing but is now imputed. We can use these indices to find the location of imputations within a transformed dataset. The code below accesses the first 5 imputations for **x** and **y**. Because we used mean imputation, the imputation values are the same. In this case, the **record with index 5** had missing values for both **x** and **y**.


\`\`\`python
print_header("Showing the first 5 imputations for column x")
data_imputed.loc[imputer.imputed_['x'], 'x'].head()
\`\`\`

    Showing the first 5 imputations for column x
    --------------------------------------------

    5     7.577981
    13    7.577981
    24    7.577981
    29    7.577981
    30    7.577981
    Name: x, dtype: float64




\`\`\`python
print_header("Showing the first 5 imputations for column y")
data_imputed.loc[imputer.imputed_['y'], 'y'].head()
\`\`\`

    Showing the first 5 imputations for column y
    --------------------------------------------

    5     3.747201
    7     3.747201
    8     3.747201
    13    3.747201
    18    3.747201
    Name: y, dtype: float64



#### 2.3. Make Imputation Flexible
To make Imputers powerful, we need to make them flexible. While Autoimpute Imputers set default values for their arguments, the Imputers' arguments give users full control of how to impute each column within a dataset. Let's see what arguments Imputers take. The code below prints each argument's name and its default value.


\`\`\`python
print_header("Printing arguments for the SingleImputer as well as their default values")
for k,v in SingleImputer().get_params().items():
    print(f"Argument: {k}; Default: {v}")
\`\`\`

    Printing arguments for the SingleImputer as well as their default values
    ------------------------------------------------------------------------
    Argument: copy; Default: True
    Argument: imp_kwgs; Default: None
    Argument: predictors; Default: all
    Argument: seed; Default: None
    Argument: strategy; Default: default predictive
    Argument: visit; Default: default


Below is a brief description of each argument. Part II of this series explains how to use these arguments to control imputation.  
* **copy**: Whether or not to copy the dataset during the transform place. If False, imputations happen in place.  
* **imp_kwgs**: Dictionary of arguments to fine-tune imputation on a specific column or for a specific imputation strategy.  
* **predictors**: Which columns to use to predict imputations for a given column's imputation model.  
* **seed**: Seed number makes imputations reproducible.  
* **strategy**: The imputation strategy to use. Can specify for all columns or each column individually.  
* **visit**: The order in which columns should be "visited" or imputed.  


This concludes Part I of this series. This tutorial is merely an introduction to the \`Autoimpute\` Imputers. We motivated the need for Autoimpute Imputers and introduced the package design at the highest level, but we barely scratched the surface for what Imputers can do. In the code segment above, we peeked at the arguments an \`Imputer\` takes. These arguments give the user full control over the imputation process and hold the power and flexibility behind the \`Imputer\` classes. Part II of this series walks through these arguments in depth, giving users examples of how the arguments tailor imputation to fit specific needs.


`

class ImputerI extends Component {
    render() {
      return (
        <div className="imputer-I">
            <ReactMarkdown source={header} escapeHtml={false} renderers={{code: CodeBlock}} />
            <ReactMarkdown source={codePlot} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="imputer-I-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/imputer/imputer-I-scatter.png"></img>
            <ReactMarkdown source={partOne} escapeHtml={false} renderers={{code: CodeBlock}} />
            <ReactMarkdown source={partTwoOne} escapeHtml={false} renderers={{code: CodeBlock}} />
            <table border="1" className="dataframe">
                <thead>
                <tr>
                    <th>label</th>
                    <th>x</th>
                    <th>y</th>
                </tr>
                </thead>
                <tbody>
                    <tr>
                    <td><b>records missing</b></td>
                    <td>285</td>
                    <td>491</td>
                    </tr>
                </tbody>
            </table>
            <ReactMarkdown source={partTwoTwo} escapeHtml={false} renderers={{code: CodeBlock}} />
            <table border="1" className="dataframe">
            <thead>
                <tr>
                <th>label</th>
                <th>x</th>
                <th>y</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                <td><b>records missing</b></td>
                <td>0</td>
                <td>0</td>
                </tr>
            </tbody>
            </table>
            <ReactMarkdown source={partTwoThree} escapeHtml={false} renderers={{code: CodeBlock}} />
            <table border="1" className="dataframe">
                <thead>
                    <tr>
                    <th>x</th>
                    <th>y</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                    <td>2.534781</td>
                    <td>0.257901</td>
                    </tr>
                    <tr>
                    <td>-0.118755</td>
                    <td>-0.114591</td>
                    </tr>
                    <tr>
                    <td>-1.184612</td>
                    <td>0.018801</td>
                    </tr>
                    <tr>
                    <td>1.442059</td>
                    <td>0.047037</td>
                    </tr>
                    <tr>
                    <td>0.109537</td>
                    <td>0.229042</td>
                    </tr>
                    <tr>
                    <td>NaN</td>
                    <td>NaN</td>
                    </tr>
                    <tr>
                    <td>-0.962892</td>
                    <td>0.000090</td>
                    </tr>
                    <tr>
                    <td>-0.028426</td>
                    <td>NaN</td>
                    </tr>
                    <tr>
                    <td>1.949358</td>
                    <td>NaN</td>
                    </tr>
                    <tr>
                    <td>-1.728996</td>
                    <td>0.068058</td>
                    </tr>
                </tbody>
            </table>
            <ReactMarkdown source={partTwoFour} escapeHtml={false} renderers={{code: CodeBlock}} />
            <table border="1" className="dataframe">
                <thead>
                    <tr>
                    <th>x</th>
                    <th>y</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                    <td>2.534781</td>
                    <td>0.257901</td>
                    </tr>
                    <tr>
                    <td>-0.118755</td>
                    <td>-0.114591</td>
                    </tr>
                    <tr>
                    <td>-1.184612</td>
                    <td>0.018801</td>
                    </tr>
                    <tr>
                    <td>1.442059</td>
                    <td>0.047037</td>
                    </tr>
                    <tr>
                    <td>0.109537</td>
                    <td>0.229042</td>
                    </tr>
                    <tr>
                    <td>7.577981</td>
                    <td>3.747201</td>
                    </tr>
                    <tr>
                    <td>-0.962892</td>
                    <td>0.000090</td>
                    </tr>
                    <tr>
                    <td>-0.028426</td>
                    <td>3.747201</td>
                    </tr>
                    <tr>
                    <td>1.949358</td>
                    <td>3.747201</td>
                    </tr>
                    <tr>
                    <td>-1.728996</td>
                    <td>0.068058</td>
                    </tr>
                </tbody>
            </table>
            <ReactMarkdown source={partTwoFive} escapeHtml={false} renderers={{code: CodeBlock}} />
        </div>
        );
    }
  }

  export default ImputerI;