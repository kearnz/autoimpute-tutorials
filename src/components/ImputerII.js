import React, { Component } from "react";
import ReactMarkdown from "react-markdown";
import CodeBlock from "./CodeBlock";

const header = `

## Getting the Most out of the Imputer Classes: Part II
---
This tutorial is part II of a comprehensive overview of \`Autoimpute\` Imputers. It includes:

1. Creating a Toy Dataset for the \`SingleImputer\`  
2. How the \`SingleImputer\` Works under the Hood  
3. Customizing a \`SingleImputer\` through its Arguments  

### 1. Creating a Toy Dataset for the SingleImputer
This tutorial utilizes the toy dataset created below. Note that the dataset has no missing values. This is fine because we will not perform imputations. Rather, we are interested in how the \`SingleImputer\` works under the hood and how we can control the \`SingleImputer\` to fit imputation models capable of imputing missing data. While the imputations themselves may be of primary interest to the end user, they are simply the output from a fitted imputation model. Therefore, this tutorial places emphasis on how \`Autoimpute\` Imputers **fit imputation models**.


\`\`\`python
# imports to create toy df
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# helper functions used throughout this project
print_header = lambda msg: print(f"{msg}\\n{'-'*len(msg)}")

# dataframe with columns as random selections from various arrays
toy_df = pd.DataFrame({
    "age": np.random.choice(np.arange(20,80), 50),
    "gender": np.random.choice(["Male","Female"], 50),
    "employment": np.random.choice(["Unemployed","Employed", "Part Time", "Self-Employed"], 50),
    "salary": np.random.choice(np.arange(50_000, 1_000_000), 50),
    "weight": np.random.choice(np.arange(100, 300, 0.1), 50),
})

# helper functions used throughout this project
print_header("Creating a toy dataset for demonstration purposes")
toy_df.head()
\`\`\`

    Creating a toy dataset for demonstration purposes
    -------------------------------------------------

`

const partOneTwo = `

Our toy dataframe has 5 columns of mixed types. \`age\`, \`salary\`, and \`weight\` are numeric, while \`gender\` and \`employment\` are categorical. For the numeric variables, \`age\` and \`salary\` take integer values, while \`weight\` is float. For the categorical variables, \`gender\` is binary while \`employment\` is multiclass.  As we'll see later in this tutorial, column types matter for imputation. Imputers handle numeric and categorical columns, but users must be aware of which imputation methods apply to which data types.

### 2. How the SingleImputer Works under the Hood
A lot has to happen behind the scenes for the \`SingleImputer\` to meet Autoimpute's design goals addressed in Part I, section 2. This section peeks under the hood of the \`SingleImputer\` to explore how it makes the imputation process easy, familiar, and flexible. Additionally, users are more adequately prepared to utilize Imputers if they have a solid understanding of \`Imputer\` mechanics. In this section, we cover:  
* How the \`SingleImputer\` Fits the Imputation Model  
* What are "series-imputers"?  
* The Design Patterns behind Imputers  

#### 2.1 How the SingleImputer Fits the Imputation Model
Recall from Part I, section 2 that \`Autoimpute\` Imputers set smart defaults for each argument. Therefore, we can instantiate and fit an imputer that can handle any dataset we give it. Let's create a default instance of the \`SingleImputer\` class and fit it to our toy dataset.


\`\`\`python
from autoimpute.imputations import SingleImputer
si = SingleImputer()

print_header("Fit Method returning instance of the SingleImputer class")
si.fit(toy_df)
\`\`\`

    Fit Method returning instance of the SingleImputer class
    --------------------------------------------------------

    SingleImputer(copy=True, imp_kwgs=None, predictors='all', seed=None,
           strategy='default predictive', visit='default')



In the code above, we create a default instance of the \`SingleImputer\` and assign it the variable name \`si\`. Note that because we did not specify a strategy, the \`SingleImputer\` set the strategy to **default predictive** for each column in the dataset. We'll explore the \`strategy\` argument in more detail in section 3 of this tutorial.

We then fit \`si\` to our toy dataset. This fitting process is easy and should be quite familiar. As with \`sklearn\`, our \`SingleImputer\` uses default arguments to fit the dataset and returns an instance of the \`Imputer\` class itself when the fitting process is complete (The \`BaseEstimator\` \`__repr__\` prints the instance and its arguments to the console). **But what actually happened when we fit the Imputer?** Just like \`sklearn\` \`Transformers\`, the \`SingleImputer\` generates a \`statistics_\` attribute when the \`fit\` method is called. In our case, the \`statistics_\` attribute stores the imputation model built for each column we want to impute. We can access the \`statistics_\` of \`si\` to explore what happened after the \`fit\` process completes and the imputation model(s) are created.


\`\`\`python
print_header("Statistics generated from fit method of the SingleImputer")
si.statistics_
\`\`\`

    Statistics generated from fit method of the SingleImputer
    ---------------------------------------------------------

    {'age': DefaultPredictiveImputer(cat_imputer=MultinomialLogisticImputer(),
                  cat_kwgs=None,
                  num_imputer=PMMImputer(am=48.53533000546936, asd=10,
           bm=array([-1.21117e-05,  8.27985e-03, -5.43102e+00,  1.04216e+01,
             9.20132e+00,  7.98604e+00]),
           bsd=10, fill_value='random', init='auto', neighbors=5, sample=1000,
           sig=1, tune=1000),
                  num_kwgs=None),
     'employment': DefaultPredictiveImputer(cat_imputer=MultinomialLogisticImputer(),
                  cat_kwgs=None,
                  num_imputer=PMMImputer(am=None, asd=10, bm=None, bsd=10, fill_value='random', init='auto',
           neighbors=5, sample=1000, sig=1, tune=1000),
                  num_kwgs=None),
     'gender': DefaultPredictiveImputer(cat_imputer=MultinomialLogisticImputer(),
                  cat_kwgs=None,
                  num_imputer=PMMImputer(am=None, asd=10, bm=None, bsd=10, fill_value='random', init='auto',
           neighbors=5, sample=1000, sig=1, tune=1000),
                  num_kwgs=None),
     'salary': DefaultPredictiveImputer(cat_imputer=MultinomialLogisticImputer(),
                  cat_kwgs=None,
                  num_imputer=PMMImputer(am=887784.0482856919, asd=10,
           bm=array([  -3475.07484,    -351.17341,  -90835.31078, -197110.63707,
            -128389.04932,   83669.62554]),
           bsd=10, fill_value='random', init='auto', neighbors=5, sample=1000,
           sig=1, tune=1000),
                  num_kwgs=None),
     'weight': DefaultPredictiveImputer(cat_imputer=MultinomialLogisticImputer(),
                  cat_kwgs=None,
                  num_imputer=PMMImputer(am=203.908957825882, asd=10,
           bm=array([ 9.18857e-02, -1.35828e-05, -1.92486e+00,  2.90271e+00,
             2.08853e+00,  8.95211e+00]),
           bsd=10, fill_value='random', init='auto', neighbors=5, sample=1000,
           sig=1, tune=1000),
                  num_kwgs=None)}



While the \`statistics_\` attribute is quite dense in the default case, it demonstrates what a \`SingleImputer\` does by default after fitting an \`Imputer\`. \`si.statistics_\` returns a dictionary, where the key is a column we are imputing in our dataset and the value is a **series-imputer** that corresponds to the strategy set for the given column (or key). Each series-imputer contains within itself the **imputation model** for the given column it is called upon to fit. The next section discusses the concept of a series-imputer in more detail.

#### 2.2 What are "series-imputers"?
The **series-imputer** is a critical component of how \`Autoimpute\` Imputers work under the hood. From the the end-user's perspective, series-imputers are simply workers behind the scenes that implement a given imputation model. From Autoimpute's pespective, series-imputers are classes that implement the "imputation interface" and therefore can be used interchangeably within the \`SingleImputer\` when different strategies are passed.

The \`strategy\` a \`SingleImputer\` assigns to each column in a dataset maps to a **series-imputer class prefixed by the strategy's name**. These series-imputers are in the \`autoimpute.imputations.series\` folder, but they are hidden from end users because they aren't meant for public use. The \`SingleImptuer\` and \`MultipleImputer\` are the main classes \`Autoimpute\` exposes to end users because they are robust and work with \`DataFrames\`. Under the hood, however, both of these \`Imptuers\` rely on series-imputers to do all of the actual work once a dataset is ready for imputation models to be fit.

We'll explain the importance of and reasoning behind this design pattern in the next section (2.3). For now, let's return to our \`si\` \`Imputer\` and explore the series-imputer used by default for each column. We'll focus on stats for \`employment\` first.


\`\`\`python
# get the series imputer for employment column
emp_series_imputer = si.statistics_["employment"]

print_header("The series-imputer for the 'employment' column in the toy dataset")
emp_series_imputer
\`\`\`

    The series-imputer for the 'employment' column in the toy dataset
    -----------------------------------------------------------------

    DefaultPredictiveImputer(cat_imputer=MultinomialLogisticImputer(),
                 cat_kwgs=None,
                 num_imputer=PMMImputer(am=None, asd=10, bm=None, bsd=10, fill_value='random', init='auto',
          neighbors=5, sample=1000, sig=1, tune=1000),
                 num_kwgs=None)



The code above returns the value of \`si.statistics_\` for the \`employment\` key. We observe that our \`si\` \`Imputer\` fit the \`employment\` column using the \`DefaultPredictiveImputer\`. In fact, \`si\` fits all the columns in our toy dataset with the \`DefaultPredictiveImputer\`. This occurs because we did not specify a strategy when creating an instance of the \`Imputer\`. Therefore, the \`Imputer\` set each column's strategy to \`default predictive\` - the default strategy deployed when a user does not set the imputation strategy.

The \`DefaultPredictiveImputer\` **is an example of a series-imputer**. Specifically, it is the series-imputer that maps to any column with the \`default predictive\` strategy. As we mentioned above, each series-imputer is a class itself that implements a specific imputation model following a general set of rules to which all series-imputers must adhere. The \`DefaultPredictiveImputer\` is actually one of the most complex because it has to be flexible enough to fit numeric and categorical columns. In fact, the \`DefaultPredictiveImputer\` delegates its work to other series-imputers depending on the column type, as showin the code below.


\`\`\`python
print_header("The series imputer for categorical columns within DefaultPredictiveImputer")
print(emp_series_imputer.cat_imputer)
print()
print_header("The series imputer for numerical columns within DefaultPredictiveImputer")
print(emp_series_imputer.num_imputer)
\`\`\`

    The series imputer for categorical columns within DefaultPredictiveImputer
    --------------------------------------------------------------------------
    MultinomialLogisticImputer()
    
    The series imputer for numerical columns within DefaultPredictiveImputer
    ------------------------------------------------------------------------
    PMMImputer(am=None, asd=10, bm=None, bsd=10, fill_value='random', init='auto',
          neighbors=5, sample=1000, sig=1, tune=1000)


The \`DefaultPredictiveImputer\` class takes the \`cat_imputer\` and \`num_imputer\` arguments. The \`cat_imputer\` is the series-imputer that the \`DefaultPredictiveImputer\` uses when it comes across a categorical column (i.e. \`object\` datatype in \`pandas\` \`DataFrames\`). The \`num_imputer\` is the series-imputer that a \`DefaultPredictiveImputer\` users when it comes across a numerical column (i.e. \`numeric\` datatype in \`pandas\` \`Dataframes\`). \`cat_imputer\` is set to \`MultinomialLogisticImputer\` by default, while \`num_imputer\` is set to \`PMMImputer\` by default. \`MultinomialLogisticImputer\` is the series-imputer for the \`multinomial logistic\` strategy, while \`PMMImputer\` is the series-imputer for the \`pmm\` strategy. Therefore, the \`default predictive\` strategy is simply an abstraction that chooses between the \`multionmial logistic\` strategy and \`pmm\` strategy depending on the column datatype. Behind these strategies are their respective series-imputers. The \`DefaultPredictiveImputer\` is an abstraction that chooses to implement the \`MultinomialLogisticImputer\` or the \`PMMImputer\` depending on the column datatype.

Before we move to the next section, we'll look at the series-imputer for \`age\`.


\`\`\`python
print_header("The series-imputer for the 'employment' column in the toy dataset")
si.statistics_["age"]
\`\`\`

    The series-imputer for the 'employment' column in the toy dataset
    -----------------------------------------------------------------

    DefaultPredictiveImputer(cat_imputer=MultinomialLogisticImputer(),
                 cat_kwgs=None,
                 num_imputer=PMMImputer(am=48.53533000546936, asd=10,
          bm=array([-1.21117e-05,  8.27985e-03, -5.43102e+00,  1.04216e+01,
            9.20132e+00,  7.98604e+00]),
          bsd=10, fill_value='random', init='auto', neighbors=5, sample=1000,
          sig=1, tune=1000),
                 num_kwgs=None)



The astute reader will have already noticed that the \`DefaultPredictiveImputer\` for \`age\` is a bit different than the one used for \`employment\`, even though each column has the same strategy (\`default predictive\`). The differences are the values of the arguments within the \`PMMImputer\`. This is an implementation detail when the \`PMMImputer\` actually fits a dataset, which we will address in a separate tutorial on what each imputation strategy (or series-imputer) actually does. In our case, what's important to remember is that the \`default predictive\` strategy delegates to either \`pmm\` or \`multinomial logistic\` depending on the column type. Because \`age\` is a numerical column, it's \`default predictive\` strategy is actually \`pmm\`, so the \`PMMImputer\` is evoked to fit the imputation model for \`age\`. With \`employment\`, \`PMMImputer\` is never evoked since the column is categorical. The \`MultinomialLogisticImputer\` is evoked instead, as the \`default predictive\` strategy for \`employment\` delegates to \`multinomial logistic\`.

#### 2.3 The Design Considerations behind Imputers

In Part I, section 2, we introduced the **design goals** of \`Autoimpute\`, noting that we want the package to be easy to use, flexible, and familiar to those with experience using packages in Python's machine learning ecosystem. In this section, we discuss the **design patterns** behind \`Autoimpute\` that help us realize our goals for the package. In this tutorial, we focus on the design considerations for Imputers only, specifically the \`SingleImputer\` and its series-imputer counterparts.

Let's start by reviewing the relationship between **strategies and series-imputers**. The \`SingleImputer\` strategies are easily accessible from the \`strategies\` class attribute (actually inherited from the \`BaseImputer\`). Remember, the \`strategies\` attribute is simply a dictionary that maps strategies to series-imputers. The code below shows a \`DataFrame\` with the available strategies in the \`SingleImputer\` and the corresponding series-imputer that maps to each strategy. The final column contains the **classes that each series-imputer inherit from**. Note that we've removed the \`default strategies\`, because they are merely abstractions that pick between other strategies.


\`\`\`python
inherited = lambda v: list(map(lambda cls_: cls_.__name__, v().__class__.__bases__))
series_imputer_df = pd.DataFrame({
    "strategy": list(si.strategies.keys()), 
    "series-imputer": list(map(lambda v: v.__name__, si.strategies.values())),
    "inherited from": [inherited(v) for v in si.strategies.values()]
})
no_default = np.where(~series_imputer_df.strategy.str.contains("default"))[0]
print_header("Displaying the SingleImputer's strategies and corresponding series-imputers")
series_imputer_df.loc[no_default, ["strategy", "series-imputer", "inherited from"]]
\`\`\`

    Displaying the SingleImputer's strategies and corresponding series-imputers
    ---------------------------------------------------------------------------

`

const partTwoFinish = `

As we've previously stated, each strategy maps to a series-imputer prefixed by the strategy's name. The series-imputer is responsible for implementing the strategy-specific imputation model. Regardless of what the imputation model actually looks like, each series-imputer must implement the imputation model with just \`fit\`, \`impute\`, and \`fit_impute\` methods. This design is enforced by parent class \`ISeriesImputer\`. It is an abstract base class that specifies the way any of its children should behave. For those from Java / C# backgrounds, the \`ISeriesImputer\` assumes the role of **an interface**. The \`ISeriesImputer\` is the contract to which all series-imputers must adhere.

As we notice from the console output above, **all series-imputers inherit from \`ISeriesImputer\`** and therefore implement their imputation model using \`fit\`, \`impute\`, and \`fit_impute\` methods. Enforcing this contract may seem strict, but it makes all series-imputers standardized, and **it allows the \`SingleImputer\` to delegate its work to a series-imputer** without actually worrying about what the series-imputer does under the hood. Because every series-imputer must follow the "imputation contract" the \`ISeriesImptuer\` enforces, the \`SingleImptuer\` then knows every series-imputer will have the same methods. Therefore, when a user calls the \`fit\` method of the \`SeriesImputer\`, the \`SeriesImputer\` simply calls the \`fit\` method of the corresponding series-imputer specified for each column and waits for each series-imputer to respond with an imputation model. 

The delegation design pattern has a number of benefits. First, it isolates dependencies, making it very simple to debug any issue a \`SingleImputer\` faces when fitting a dataset. Because the \`SingleImputer\` delegates its work to independent series-imputers, we can easily identify which series-imputer has a problem if an error is thrown. Next, this design is inherently flexible and easy to extend. Adding another imputer strategy requires a few steps, but those steps are quite clear. A user must create a series-imputer and strategy name, and as long as the series-imputer inherits from the \`ISeriesImputer\` and implements its "imputation contract", then \`SingleImputer\` can use the series-imputer out of the box. Lastly, this delegation pattern is inherently scalable if series-imputers work independently. The \`SingleImputer\` simply waits for series-imputers to return imputation models, so those imputation models can, in theory, fit in parallel.

We're currently working to make Imputers more scalable and extensible by default, and these advanced features will be the subject of later tutorials. For now, it's important to understand the relationship between strategies and series imputers, as well as the delegation pattern the \`SingleImputer\` users to dispatch its work to the proper helper class, or series-imputer, that builds a requested imputation model under a set of rules enforced by the \`ISeriesImputer\`.

### 3. Customizing a SingleImputer through its Arguments
In the last section, we walked through the design of the \`SingleImputer\`, but we did not pay much attention to any of its arguments. In this section, we explore how we can **customize the SingleImputer** by tuning its arguments. As we tune arguments, we look at how the \`statistics_\` attribute of the \`SingleImputer\` changes. We learn that some of the arguments we pass to the \`SingleImputer\` alter the behavior of the \`SingleImputer\` itself, while other arguments modify the specific imputation models created by the series-imputers to which the \`SingleImputer\` delegates its work. The arguments that modify series-imputers give us control over the imputation models within a \`SingleImputer\`. We'll explore these in depth in the remainder of this tutorial. These arguments include:  
* **\`strategy\`**  
* **\`imp_kwgs\`**  
* **\`predictors\`**  

#### The strategy argument
The **strategy** argument is the most important argument within the \`SingleImputer\`. The list below shows all the strategies available to impute a column within a \`DataFrame\`. \`predictive default\` is the default strategy if a user does not specify one. As we observed in section 2, \`predictive default\` chooses the preferred strategy to use depending on a column's data type (\`pmm\` for numerical, \`multinomial logistic\` for categorical). Note that some of these strategies are for categorical data, while others are for numeric data. As we'll see later, the Imputers let the user know whether a strategy will work for a given column when you try to fit the imputation model.


\`\`\`python
print_header("Strategies Available for Imputation")
print(list(SingleImputer().strategies.keys()))
\`\`\`

    Strategies Available for Imputation
    -----------------------------------
    ['default predictive', 'least squares', 'stochastic', 'binary logistic', 'multinomial logistic', 'bayesian least squares', 'bayesian binary logistic', 'pmm', 'lrd', 'default univariate', 'default time', 'mean', 'median', 'mode', 'random', 'norm', 'categorical', 'interpolate', 'locf', 'nocb']


We have a wealth of imputation methods at our disposal, and we continue to make more available. That being said, imputation strategies are restricted to the list of strategies provided above. A user cannot even create an instance of an Imputer if the strategy he or she provides is not supported. Improper strategy specification throws a \`ValueError\`, as shown below. The traceback is removed to keep this tutorial clean, but the error below is clear. The \`strategy\` argument is validated when instantiating the class.


\`\`\`python
# proviging a strategy not yet supported or that doens't exist
print_header("Creating a SingleImputer with an unsupported strategy")
try:
    SingleImputer(strategy="unsupported")
except ValueError as ve:
    print(f"{ve.__class__.__name__}: {ve}")
\`\`\`

    Creating a SingleImputer with an unsupported strategy
    -----------------------------------------------------
    ValueError: Strategy unsupported not a valid imputation method.
     Strategies must be one of ['default predictive', 'least squares', 'stochastic', 'binary logistic', 'multinomial logistic', 'bayesian least squares', 'bayesian binary logistic', 'pmm', 'lrd', 'default univariate', 'default time', 'mean', 'median', 'mode', 'random', 'norm', 'categorical', 'interpolate', 'locf', 'nocb'].


So how can utilize supported strategies? We can set the **strategy** in three ways:  
* As a **string**, which broadcasts the strategy across every column in the DataFrame
* As a **list or tuple**, where the position of strategies in the iterator are applied to the corresponding column
* As a **dictionary**, where the key is the column we want to impute, and the value is the strategy to use

We advise the **dictionary method**, as it is the most explicit and allows the user to impute all or a subset of the columns in a DataFrame. It is also the least prone to unexpected behavior and errors when trying to fit the imputation model. Let's look at some examples below, where we run into problems with the string and iterator method but have better control with the dictionary method.


\`\`\`python
# string strategy broadcasts across all strategies
si_str = SingleImputer(strategy="mean")

# list strategy, where each item is a corresponding strategy
si_list = SingleImputer(strategy=["mean", "binary logistic", "median"])

# dictionary strategy, where we specify column and strategy together
# Note that with the dictionary, we can specify a SUBSET of columns to impute
si_dict = SingleImputer(strategy={"gender":"categorical", "salary": "pmm"})
\`\`\`

Note that we instantiated each \`SingleImputer\` with no issues yet. We provided valid types (string, iterator, or dictionary) for the \`strategy\` argument, and each \`strategy\` we provided is one of the strategies supported. So we are able to at least crete an instance of our class. **But that does not necessarily mean the strategies we've chosen will work with the columns of our \`DataFrame\`**. This is something we cannot validate **until the user fits the \`Imputer\` to a dataset** because the \`Imputer\` itself knows nothing about the dataset's columns or the column types until the \`Imputer\` is fit. We will see how this plays out below when we try to fit sample Imputers to our toy dataset.

First, let's try to fit \`si_str\`, which is a \`Imputer\` but won't work with our toy data.


\`\`\`python
# fitting the string strategy, which yields an error
print_header("Fitting si_str, a SingleImputer that broadcasts strategy='mean'")
try:
    si_str.fit(toy_df)
except TypeError as te:
    print(f"{te.__class__.__name__}: {te}")
\`\`\`

    Fitting si_str, a SingleImputer that broadcasts strategy='mean'
    ---------------------------------------------------------------
    TypeError: mean not appropriate for Series employment of type object.


While a valid imputer, \`si_str\` failed to fit the dataset. We set the strategy as \`mean\`, so the \`si_str\` \`Imputer\` broadcast \`mean\` to all columns in our dataset. \`mean\` worked fine when imputing \`age\`, the first column, because \`age\` is numerical. But an error occurred when we tried to take the \`mean\` of the \`employment\` column. We cannot take the \`mean\` of a categorical column such as \`employment\`, so the \`Imputer\` throws an error. (Note that the same error would have occurred when the \`Imputer\` reached the \`gender\` column.)

Therefore, we must be very careful when setting the strategy with a string since that strategy is broadcast to all columns in the DataFrame. When setting strategy with a string, we must ensure that we want the same strategy for each column, and we must ensure that our DataFrame does not contain any columns that the strategy cannot fit. As we'll see later, we could have specified \`mean\` for every column except \`employment\` and \`gender\` if we had used a dictionary, or we could have specified a different strategy for \`employment\` and \`gender\` had we used a list.

Next, we'll attempt to fit \`si_list\`, an \`Imputer\` that uses a list of strategies.


\`\`\`python
# fitting the string strategy, which yields an error
print_header("Fitting si_list, a SingleImputer with a list of strategies to apply")
try:
    si_list.fit(toy_df)
except ValueError as ve:
    print(f"{ve.__class__.__name__}: {ve}")
\`\`\`

    Fitting si_list, a SingleImputer with a list of strategies to apply
    -------------------------------------------------------------------
    ValueError: Length of columns not equal to number of strategies.
    Length of columns: 5
    Length of strategies: 3


With \`si_list\`, a different problem occurs. If we use a list as the value to the \`strategy\` argument, the list **must contain one strategy per column**. When we created the \`Imputer\`, the list contained 3 valid strategies, so no problem with instantiation. But when we tried to fit the \`Imputer\` to the dataset, the \`Imputer\` noticed the dataset had 5 columns. The \`Imputer\` does not know how to handle the fourth and fifth column, and the \`Imputer\` has not been told explicitly to ignore these columns, so a \`ValueError\` is thrown.

Finally, let's examine \`si_dict\`:


\`\`\`python
print_header("Fitting si_dict, a SingleImputer with a dictionary of strategies to apply")
si_dict.fit(toy_df)
\`\`\`

    Fitting si_dict, a SingleImputer with a dictionary of strategies to apply
    -------------------------------------------------------------------------

    SingleImputer(copy=True, imp_kwgs=None, predictors='all', seed=None,
           strategy={'gender': 'categorical', 'salary': 'pmm'},
           visit='default')



The \`si_dict\` \`Imputer\` successfully fit the toy dataset. For \`gender\`, it used the \`categorical\` method. For \`salary\`, it used \`pmm\`. Because we did not specify any imputation method for \`age\`, \`weight\`, or \`employment\`, these columns are not imputed and are ignored. Not only is the dictionary method more flexible, but it can drastically speed up the time it takes an \`Imputer\` to fit a model if we have hundreds of columns but only need to impute a couple of them.

**So what did the fit method actually do?** As we learned in section 2, Imputers delegate the work for each column to a series-imputer that maps to the specified strategy. In this case, we specified \`pmm\` for \`salary\` and \`categorical\` for \`gender\`, so \`si_dict\` delegated work for \`salary\` to the \`PMMImputer\` and delegated work for \`gender\` to the \`CategoricalImputer\`. 


\`\`\`python
print_header("Accessing statistics after fitting the si_dict Imputer")
si_dict.statistics_
\`\`\`

    Accessing statistics after fitting the si_dict Imputer
    ------------------------------------------------------

    {'gender': CategoricalImputer(),
     'salary': PMMImputer(am=887784.0482856919, asd=10,
           bm=array([  -3475.07484,    -351.17341,  -90835.31078, -197110.63707,
            -128389.04932,   83669.62554]),
           bsd=10, fill_value='random', init='auto', neighbors=5, sample=1000,
           sig=1, tune=1000)}



Note the difference between the statistics in the code above and the statistics in section 2.1. In section 2.1, all columns including \`gender\` and \`salary\` received the \`DefaultPredictiveImputer\`. Here, \`gender\` receives the series-imputer \`CategoricalImputer\`, which maps to the \`categorical\` strategy; \`salary\` receives the series-imputer \`PMMImputer\` which maps to the \`pmm\` strategy; and the remaining columns receive **no series-imputer at all because we explicitly ignored them.** Both \`si\` and \`si_dict\` are instances of the same \`SingleImputer\`, but each looks very different because we've tuned the \`strategy\` argument. Because the \`strategy\` argument maps to a series-imputer that creates each column's imputation model, \`si\` and \`si_dict\` end up with completely different \`statistics_\`. If we actually impute data, each of these Imputers would produce a very different set of imputations. 

#### The img_kwgs argument
Observe that at times, the **series-imputers take arguments of their own**. This occurs because certain strategies may need additional information in order to implement their imputation model. In the example above, the \`categorical\` strategy has no additional parameters necessary to pass to its series-imputer, while the \`pmm\` strategy has 10 additional parameters that control the way the \`PmmImputer\` fits a dataset. While each strategy's respective series-imputer sets default arguments as well, we want to be able to control those arguments to alter how the strategy ultimately works and performs. We can do so using the **imp_kwgs** argument in the \`SingleImputer\`, which by default is set to \`None\`.

Let's review our \`si_dict\` \`Imputer\`. By default, it's value is set to \`None\`.


\`\`\`python
si_dict
\`\`\`

    SingleImputer(copy=True, imp_kwgs=None, predictors='all', seed=None,
           strategy={'gender': 'categorical', 'salary': 'pmm'},
           visit='default')


When \`imp_kwgs\` is \`None\`, all series-imputers for given strategies use their default arguments. We observe those default once we've fit our \`Imputer\` by accessing its statistics.


\`\`\`python
si_dict.statistics_
\`\`\`

    {'gender': CategoricalImputer(),
     'salary': PMMImputer(am=887784.0482856919, asd=10,
           bm=array([  -3475.07484,    -351.17341,  -90835.31078, -197110.63707,
            -128389.04932,   83669.62554]),
           bsd=10, fill_value='random', init='auto', neighbors=5, sample=1000,
           sig=1, tune=1000)}



We specified \`pmm\` (predictive mean matching) as the strategy to use for the \`salary\` column. \`pmm\` is a semi-parametric method that borrows logic from bayesian regression, linear regression, and nearest neighbor search. We'll cover details of imputation algorithms in another tutorial, but for now, let's review some of the arguments the \`PMMImputer\` takes. Specifically, we'll focus on **neighbors** and **fill_value**.  
* **neighbors** is the number of observations \`pmm\` will use to determine an imputation value.  
* If **fill_value** is set to **random**, \`pmm\` randomly selects one of the \`n\` neighbors as the imputation. Random is the default.  
* If the **fill_value** is set to **mean**, \`pmm\` takes the mean of the \`n\` neighbors and uses the mean as the imputation.  

We'll create two new Imputers to demonstrate how we can tweak the behavior of the \`PMMImputer\` through \`imp_kwgs\`.


\`\`\`python
# using the column name
si_dict_col = SingleImputer(
    strategy={"gender":"categorical", "salary": "pmm", "weight": "pmm"},
    imp_kwgs={"salary": {"neighbors": 10, "fill_value": "mean"}}
)

# using the strategy name
si_dict_strat = SingleImputer(
    strategy={"gender":"categorical", "salary": "pmm", "weight": "pmm"},
    imp_kwgs={"pmm": {"neighbors": 10, "fill_value": "mean"}}
)
\`\`\`


\`\`\`python
# fit the si_dict_col imputer
si_dict_col.fit(toy_df)
\`\`\`

    SingleImputer(copy=True,
           imp_kwgs={'salary': {'neighbors': 10, 'fill_value': 'mean'}},
           predictors='all', seed=None,
           strategy={'gender': 'categorical', 'salary': 'pmm', 'weight': 'pmm'},
           visit='default')

\`\`\`python
# fit the si_dict_strat imputer
si_dict_strat.fit(toy_df)
\`\`\`

    SingleImputer(copy=True,
           imp_kwgs={'pmm': {'neighbors': 10, 'fill_value': 'mean'}},
           predictors='all', seed=None,
           strategy={'gender': 'categorical', 'salary': 'pmm', 'weight': 'pmm'},
           visit='default')

We fit two new imputers to the our toy dataset. The first \`Imputer\`, \`si_dict_col\`, sets \`pmm\` as the strategy for both \`weight\` and \`salary\`. Additionally, it sets \`imp_kwgs\` to fine-tune **the pmm algorithm for the salary column only**. Our second imputer, \`si_dict_strat\`, sets the same strategies, but it sets \`imp_kwgs\` to fine-tune **any column that uses the pmm algorithm**.

As a result, the \`si_dict_col\` \`Imputer\` uses a customized version of \`pmm\` for salary but the default version of \`pmm\` for weight. We can see the differences by accessing the Imputer's statistics, as shown below. The \`weight\` column has the default number of \`neighbors\` (5) and the default \`fill_value\` (random). The \`salary\` column, on the other hand, uses 10 \`neighbors\`, and its \`fill_value\` is set to \`mean\`.


\`\`\`python
print_header("PMMImputer for weight")
print(si_dict_col.statistics_["weight"])
print_header("PMMImputer for salary")
print(si_dict_col.statistics_["salary"])
print_header("Number of neighbors used for weight vs. salary")
print(
    {"number of neighbors for salary": si_dict_col.statistics_["salary"].neighbors, 
     "number of neighbors for weight": si_dict_col.statistics_["weight"].neighbors}
)
\`\`\`

    PMMImputer for weight
    ---------------------
    PMMImputer(am=203.908957825882, asd=10,
          bm=array([ 9.18857e-02, -1.35828e-05, -1.92486e+00,  2.90271e+00,
            2.08853e+00,  8.95211e+00]),
          bsd=10, fill_value='random', init='auto', neighbors=5, sample=1000,
          sig=1, tune=1000)
    PMMImputer for salary
    ---------------------
    PMMImputer(am=887784.0482856919, asd=10,
          bm=array([  -3475.07484,    -351.17341,  -90835.31078, -197110.63707,
           -128389.04932,   83669.62554]),
          bsd=10, fill_value='mean', init='auto', neighbors=10, sample=1000,
          sig=1, tune=1000)
    Number of neighbors used for weight vs. salary
    ----------------------------------------------
    {'number of neighbors for salary': 10, 'number of neighbors for weight': 5}


The \`si_dict_strat\` \`Imputer\` applies \`imp_kwgs\` to **any column that has pmm as its strategy**. Therefore, the customized \`PMMImputer\` applies to both the \`salary\` and the \`weight\` column, as shown below. The number of neighbors is the same for both columns, as is the \`fill_value\`.


\`\`\`python
print_header("PMMImputer for weight")
print(si_dict_strat.statistics_["weight"])
print_header("PMMImputer for salary")
print(si_dict_strat.statistics_["salary"])
print_header("Number of neighbors used for weight vs. salary")
print(
    {"number of neighbors for salary": si_dict_strat.statistics_["salary"].neighbors, 
     "number of neighbors for weight": si_dict_strat.statistics_["weight"].neighbors}
)
\`\`\`

    PMMImputer for weight
    ---------------------
    PMMImputer(am=203.908957825882, asd=10,
          bm=array([ 9.18857e-02, -1.35828e-05, -1.92486e+00,  2.90271e+00,
            2.08853e+00,  8.95211e+00]),
          bsd=10, fill_value='mean', init='auto', neighbors=10, sample=1000,
          sig=1, tune=1000)
    PMMImputer for salary
    ---------------------
    PMMImputer(am=887784.0482856919, asd=10,
          bm=array([  -3475.07484,    -351.17341,  -90835.31078, -197110.63707,
           -128389.04932,   83669.62554]),
          bsd=10, fill_value='mean', init='auto', neighbors=10, sample=1000,
          sig=1, tune=1000)
    Number of neighbors used for weight vs. salary
    ----------------------------------------------
    {'number of neighbors for salary': 10, 'number of neighbors for weight': 10}


Therefore, we can customize the series-imputer for any given imputation strategy **by column** or **by strategy itself**. While we demonstrated this behavior for \`pmm\`, the same logic applies to any imputation strategy that takes additional arguments. Below is an example using \`interpolate\` as an imputation strategy. We'll specify \`imp_kwgs\` by column.


\`\`\`python
# interpolate with imp_kwgs using the column name
si_interp = SingleImputer(
    strategy={"salary": "interpolate", "weight": "interpolate"},
    imp_kwgs={"salary": {"fill_strategy": "linear"}, "weight": {"fill_strategy": "quadratic"}}
)

# fit the imputer
si_interp.fit(toy_df)
\`\`\`

    SingleImputer(copy=True,
           imp_kwgs={'salary': {'fill_strategy': 'linear'}, 'weight': {'fill_strategy': 'quadratic'}},
           predictors='all', seed=None,
           strategy={'salary': 'interpolate', 'weight': 'interpolate'},
           visit='default')




\`\`\`python
si_interp.statistics_
\`\`\`

    {'salary': InterpolateImputer(end=None, fill_strategy='linear', order=None, start=None),
     'weight': InterpolateImputer(end=None, fill_strategy='quadratic', order=None,
               start=None)}



In this case, we've created an \`Imputer\` that uses **linear interpolation** to impute \`salary\` and **quadratic interpolation** to impute \`weight\`. This is just another example of how we can use \`imp_kwgs\` along with a column or strategy to fine-tune the imputation algorithm itself when we create an instance of an \`Imputer\`.

The code samples above demonstrate proper usage of the \`imp_kwgs\` argument. To put it succinctly, **\`imp_kwgs\` extends the \`strategy\` argument**. They provide additional control over the imputation model itself should a user want to deviate from a series-imputer's default arguments.

#### The predictors argument
So far, we've seen that the \`strategy\` argument determines which type of imputation model(s) we want to use, while the \`imp_kwgs\` argument provides additional parameters to specified strategies to control the behavior of their respective series-imputer. These arguments focus on **tuning the imputation model itself**. We also have options to specify **what the imputation model depends on** if the imputation model is multivariate predictive. In other words, we can control what **predictors** a series-imputer uses to fit an imputation model. Unsuprisingly, we control this behavior through the \`SeriesImputer\` **\`predictor\`** argument.


\`\`\`python
si_all = SingleImputer()
print_header("Show the SingleImputer as well as its predictor argument")
print(si_all)
print(f"si_all predictors: {si.predictors}")
\`\`\`

    Show the SingleImputer as well as its predictor argument
    --------------------------------------------------------
    SingleImputer(copy=True, imp_kwgs=None, predictors='all', seed=None,
           strategy='default predictive', visit='default')
    si_all predictors: all


As with the \`strategy\` and \`imp_kwgs\` arguments, we have flexibility surrounding how we specify the \`predictors\` argument. By default, \`SingleImputers\` set \`predictors\` to \`all\`. This default means that **all columns are used** to build an imputation model for strategies that depend on predictors. We could also use a list or a dictionary to specify the predictors we want to use. We demonstrate both below.

#### When are imp_kwgs ignored?
While \`imp_kwgs\` extend the \`strategy\` argument, there isn't always something to extend. In some cases, the \`imp_kwgs\` argument is ignored even if a user explicitly defines it. There are two primary cases where this behavior occurs:  

1. **strategy's series-imputer has no class instance attributes**  
2. **When using a dictionary to define strategies, the user does not provide a strategy for a column**  

**Mean imputation** is a good example of the first case above. Mean imputation simply takes the mean of observed values in an array. Therefore, Autoimpute's \`MeanImputer\` does not need any arguments to instantiate it. Therefore, if a user defines arbitrary arguments within the \`imp_kwgs\` for a column with \`mean\` as its strategy, the \`imp_kwgs\` are ignored. This behavior is intended, and an **error is not thrown**. Autoimpute avoids an error in this case because some strategies take arbitrary keyword arguments (a.k.a. $kwargs$) to their series-imputer. Because Python ignores unused $kwargs$, so does Autoimpute. A good example of this is the \`MultinomialLogisticImputer\` seen in examples above. While the class does not appear to have any arguments, it actually takes $kwargs$. Under the hood, the $kwargs$ are passed to an instance of \`sklearn\` \`LogisticRegression\`, which has a number of possible keyword arguments to control how the regression behaves. We'll cover the details of each series-imputer in subsequent tutorials. For now, it's important to understand when and why \`imp_kwgs\` are ignored. In general, remember that \`imp_kwgs\` are evaluated **if and only if they logically extend the strategy in question**.

The second case is a bit more straightforward. If a end user defines strategies in a dictionary and does not provide a strategy for a column, the column is ignored when fitting a dataset. When a column is ignored, it does not matter what its \`imp_kwgs\` are, as the \`Imputer\` is not building a model for that column.


\`\`\`python
print_header("Creating a SingleImputer with a list of predictors for all columns")
preds_list = ["weight", "age", "salary"]
si_list_pred = SingleImputer(predictors=preds_list)
print(si_list_pred)
print()
print_header("Creating a SingleImputer with a dictionary of predictors by column")
preds_dict = {"weight": ["age", "gender"], "salary": ["education", "age"]}
si_dict_pred = SingleImputer(predictors=preds_dict)
print(si_dict_pred)
\`\`\`

    Creating a SingleImputer with a list of predictors for all columns
    ------------------------------------------------------------------
    SingleImputer(copy=True, imp_kwgs=None,
           predictors=['weight', 'age', 'salary'], seed=None,
           strategy='default predictive', visit='default')
    
    Creating a SingleImputer with a dictionary of predictors by column
    ------------------------------------------------------------------
    SingleImputer(copy=True, imp_kwgs=None,
           predictors={'weight': ['age', 'gender'], 'salary': ['education', 'age']},
           seed=None, strategy='default predictive', visit='default')


If we use a list or iterator to specify predictors, **imputation models use the columns within the list as the set of predictors**. The \`SingleImputer\` will take care of removing a column as a predictor for that column's imputation strategy. For example, when it's time to impute \`salary\` in the example above, the \`SingleImputer\` will use \`weight\` and \`age\`. Similarly, the \`SingleImputer\` will use \`weight\` and \`salary\` when it's time to impute \`age\`.

We can also use a dictionary to specify predictors. As with the \`strategy\` argument, we specify the predictors column-wise. Unlike \`strategy\`, **the predictors argument is implicitly set to \`all\` for any columns we do not provide predictors**. This implicit behavior stems from the fact that **predictors support the strategy argument**. If a user specifies a multivariate strategy for a column but does not specify \`predictors\`, we cannot ignore the predictors argument, or else the strategy will not work.

#### When are predictors ignored?
In certain cases, the \`SingleImputer\` simply ignores the \`predictors\` argument. This behavior occurs when \`predictors\` simply are not relevant. As mentioned above, \`predictors\` support a \`strategy\`. So if the \`strategy\` does not need \`predictors\`, then the \`predictors\` are ingored. Let's review two cases where this may occur:  

1. **Univariate strategies**  
2. **When using a dictionary to define strategies, the user does not provide a strategy for a column**  

The first case applies to univariate strategies. Similarly to \`imp_kwgs\`, **predictors are evaluated if and only if they logically support a strategy**. Univariate strategies don't need predictors ever, so the predictors are always ignored. Using the same example as \`imp_kwgs\`, take the \`MeanImputer\`. The \`MeanImputer\` uses observed values in an array to determine the imputation model. Because it never needs predictors to determine the \`mean\` of an array, the \`MeanImputer\` ignores predictors in all cases. They are not needed to support mean imputation, so they are politely ignored.

The second case is the same as we saw with \`imp_kwgs\`. Columns with no strategy have no imputation model, so obviously they do not need predictors.

This concludes the second part of this series. In this tutorial, we examined how the \`SingleImputer\` works under the hood. We reviewed the design patterns that make \`Autoimpute Imputers\` meet the high level design goals discussed in part I of this series. We then explored some of the arguments a \`SingleImputer\` takes. Specifically, we addressed the arguments that impact the behavior of respective series-imputers (i.e. the arguments that determine the structure of the imputation models). In the next turorial, we extend these concepts to the \`MultipleImputer\`. We observe how the \`MultipleImputer\` is simply multiple \`SingleImputers\` at work under the hood.


`

class ImputerII extends Component {
    render() {
      return (
        <div className="imputer-II">
        <ReactMarkdown source={header} escapeHtml={false} renderers={{code: CodeBlock}} />
        <table border="1" class="dataframe">
          <thead>
            <tr>
              <th>age</th>
              <th>employment</th>
              <th>gender</th>
              <th>salary</th>
              <th>weight</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>74</td>
              <td>Unemployed</td>
              <td>Female</td>
              <td>484881</td>
              <td>233.9</td>
            </tr>
            <tr>
              <td>26</td>
              <td>Employed</td>
              <td>Male</td>
              <td>874459</td>
              <td>110.1</td>
            </tr>
            <tr>
              <td>66</td>
              <td>Self-Employed</td>
              <td>Female</td>
              <td>800823</td>
              <td>231.1</td>
            </tr>
            <tr>
              <td>44</td>
              <td>Part Time</td>
              <td>Male</td>
              <td>606560</td>
              <td>144.7</td>
            </tr>
            <tr>
              <td>57</td>
              <td>Self-Employed</td>
              <td>Male</td>
              <td>269862</td>
              <td>220.0</td>
            </tr>
          </tbody>
        </table>
        <ReactMarkdown source={partOneTwo} escapeHtml={false} renderers={{code: CodeBlock}} />
        <table border="1" class="dataframe">
          <thead>
            <tr>
              <th>strategy</th>
              <th>series-imputer</th>
              <th>inherited from</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>least squares</td>
              <td>LeastSquaresImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>stochastic</td>
              <td>StochasticImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>binary logistic</td>
              <td>BinaryLogisticImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>multinomial logistic</td>
              <td>MultinomialLogisticImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>bayesian least squares</td>
              <td>BayesianLeastSquaresImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>bayesian binary logistic</td>
              <td>BayesianBinaryLogisticImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>pmm</td>
              <td>PMMImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>lrd</td>
              <td>LRDImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>mean</td>
              <td>MeanImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>median</td>
              <td>MedianImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>mode</td>
              <td>ModeImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>random</td>
              <td>RandomImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>norm</td>
              <td>NormImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>categorical</td>
              <td>CategoricalImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>interpolate</td>
              <td>InterpolateImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>locf</td>
              <td>LOCFImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
            <tr>
              <td>nocb</td>
              <td>NOCBImputer</td>
              <td>[ISeriesImputer]</td>
            </tr>
          </tbody>
        </table>
        <ReactMarkdown source={partTwoFinish} escapeHtml={false} renderers={{code: CodeBlock}} />
        </div>
        );
    }
  }
   
  export default ImputerII;