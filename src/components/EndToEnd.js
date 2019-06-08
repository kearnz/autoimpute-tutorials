import React, { Component } from "react";
import ReactMarkdown from "react-markdown";
import CodeBlock from "./CodeBlock";

// HEADER TO START THE TUTORIAL
const inputHeader01 = `
## An End-to-End Analysis of Missing Data using Autoimpute

* This tutorial demonstrates how an analyst can utilize the \`Autoimpute\` package to handle missing data from exploration through supervised learning.  
* It presents **two examples, side by side**. The process for each example is the same, but the underlying datasets (and their missingness) differ.  
* The example on the left explores data with **MCAR** missingness, while the example on the right examines data with **MAR** missingness.  
`

/* START OF MCAR BELOW 
---------------------
*/

const mcarHeader = `
## MCAR
---
* 500 observations for two features, **predictor x** and **response y**  
* Correlation between the variables is **0.7**
* Predictor **x** is fully observed, while response **y** is missing **40%** of the observations
* The underlying missingness mechanism is **MCAR**  
* Imputation methods explored: **mean**, **least squares**, **PMM**    
`

const mcarDataPrep = `
### Imports and Data Preparation
---
\`\`\`python
'''Handling imports for analysis'''
%matplotlib inline

# general modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# autoimpute imports - utilities & visuals
from autoimpute.utils import md_pattern, proportions
from autoimpute.visuals import plot_md_locations, plot_md_percent
from autoimpute.visuals import plot_imp_dists, plot_imp_boxplots
from autoimpute.visuals import plot_imp_swarm, plot_imp_strip
from autoimpute.visuals import plot_imp_scatter

# autoimpute imports - imputations & analysis
from autoimpute.imputations import MultipleImputer
from autoimpute.analysis import MiLinearRegression

# reading the full dataset and mcar dataset
full = pd.read_csv("full.csv")
mcar = pd.read_csv("mcar.csv")
\`\`\`
`

const mcarPercent = `
### Percent Missing by Feature
---
\`\`\`python
'''MCAR percent plot'''
plot_md_percent(mcar)
\`\`\`
`

const mcarLocation = `
### Location of Missingness by Feature
---
\`\`\`python
'''MCAR location plot'''
plot_md_locations(mcar)
\`\`\`
`

const mcarMean = `
### Mean Imputation
---
\`\`\`python
'''MCAR mean imputation'''

# create the mean imputer
mi_mean_mcar = MultipleImputer(
    strategy="mean", n=5, return_list=True, seed=101
)

# print the mean imputer to console
print(mi_mean_mcar)

# perform mean imputation procedure
imp_mean_mcar = mi_mean_mcar.fit_transform(mcar)
\`\`\`
`

const mcarMeanDistBox = `
### Distribution Plots after Mean Imputation
---
\`\`\`python
'''MCAR distribution plots after mean imputation'''

# distribution plot for mean imputation
plot_imp_dists(
    d=imp_mean_mcar,
    mi=mi_mean_mcar, 
    col="y",
    title="Distributions after Mean Imputation: MCAR",
    separate_observed=False,
    hist_observed=True,
    hist_imputed=False
)

# box plot for mean imputation
plot_imp_boxplots(
    d=imp_mean_mcar,
    mi=mi_mean_mcar,
    col="y",
    title="Boxplots after Mean Imputation: MCAR"
)

# strip plot for mean imputation
plot_imp_strip(
    d=imp_mean_mcar,
    mi=mi_mean_mcar,
    col="y",
    title="Imputed vs Observed Dists after Mean Imputation: MCAR"
)
\`\`\`
`
const mcarLs = `
### Distribution Plots after Least Squares Imputation
---
\`\`\`python
'''MCAR distribution plots after least squares imputation'''

# create the least squares imputer
mi_ls_mcar = MultipleImputer(
    strategy="least squares", n=5, return_list=True, seed=101
)

# perform least squares imputation procedure
imp_ls_mcar = mi_ls_mcar.fit_transform(mcar)

# distribution plot for least squares imputation
plot_imp_dists(
    d=imp_ls_mcar,
    mi=mi_ls_mcar, 
    col="y",
    title="Distributions after Least Squares Imputation: MCAR",
    separate_observed=False,
    hist_observed=True,
    hist_imputed=False
)

# box plot for least squares imputation
plot_imp_boxplots(
    d=imp_ls_mcar,
    mi=mi_ls_mcar,
    col="y",
    title="Boxplots after Least Squares Imputation: MCAR"
)

# strip plot for least squares imputation
plot_imp_strip(
    d=imp_ls_mcar,
    mi=mi_ls_mcar,
    col="y",
    title="Imputed vs Observed Dists after Least Squares Imputation: MCAR"
)
\`\`\`
`

const mcarPmm = `
### Distribution Plots after PMM Imputation
---
\`\`\`python
'''MCAR distribution plots after PMM imputation'''

# create the PMM imputer
mi_pmm_mcar = MultipleImputer(
    strategy="pmm", n=5, return_list=True, seed=101
)

# perform PMM imputation procedure
imp_pmm_mcar = mi_pmm_mcar.fit_transform(mcar)

# distribution plot for PMM imputation
plot_imp_dists(
    d=imp_pmm_mcar,
    mi=mi_pmm_mcar, 
    col="y",
    title="Distributions after PMM Imputation: MCAR",
    separate_observed=False,
    hist_observed=True,
    hist_imputed=False
)

# box plot for PMM imputation
plot_imp_boxplots(
    d=imp_pmm_mcar,
    mi=mi_pmm_mcar,
    col="y",
    title="Boxplots after PMM Imputation: MCAR"
)

# swarm plot for PMM imputation
plot_imp_swarm(
    d=imp_pmm_mcar,
    mi=mi_pmm_mcar,
    col="y",
    title="Imputed vs Observed Dists after PMM Imputation: MCAR"
)
\`\`\`
`

const mcarRegression = `
### Linear Regression on Multiply Imputed Data
---
\`\`\`python
'''Regression after Multiple Imputation on MCAR'''

# NOTE: Full, Listwise delete, and bias code not included

# create the regression using custom imputers
lm_mean_mcar = MiLinearRegression(mi=mi_mean_mcar)
lm_ls_mcar = MiLinearRegression(mi=mi_ls_mcar)
lm_pmm_mcar = MiLinearRegression(mi=mi_pmm_mcar)
models_mcar = [lm_mean_mcar, lm_ls_mcar, lm_pmm_mcar]

# a bit of manipulation to create one dataframe
get_coeff = lambda lm_, x: lm_.summary().loc["x"].to_frame()
res_mcar = pd.concat([get_coeff(lm, "x") for lm in models_mcar], axis=1)
res_mcar.columns = ["mean", "least squares", "pmm"]
res_mcar = res_mcar.T[["coefs", "std", "vw", "vb", "vt"]]

# show the results
res_mcar
\`\`\`
`

const mcarScatter = `
### Scatterplots after Imputation Methods
---
\`\`\`python
'''Visualizing effect of imputation on regression for scatter'''

# scatterplot for mean
plot_imp_scatter(
    d=mcar, x="x", y="y", strategy="mean", color="y",
    title="Scatter after Mean Imputation: MCAR"
)

# scatterplot for least squares
plot_imp_scatter(
    d=mcar, x="x", y="y", strategy="least squares", color="y",
    title="Scatter after Least Squares Imputation: MCAR"
)

# scatterplot for pmm
plot_imp_scatter(
    d=mcar, x="x", y="y", strategy="pmm", color="y",
    title="Scatter after PMM Imputation: MCAR"
)

\`\`\`
`


/* START OF MAR BELOW 
---------------------
*/

const marHeader = `
## MAR
---
* 500 observations for two features, **predictor x** and **response y**  
* Correlation between the variables is **0.7**
* Predictor **x** is missing **40%** of the observations, while response **y** is fully observed  
* The underlying missingness mechanism is **MAR**  
* Imputation methods explored: **mean**, **least squares**, **PMM**  
`

const marDataPrep = `
### Imports and Data Preparation
---
\`\`\`python
'''Handling imports for analysis'''
%matplotlib inline

# general modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# autoimpute imports - utilities & visuals
from autoimpute.utils import md_pattern, proportions
from autoimpute.visuals import plot_md_locations, plot_md_percent
from autoimpute.visuals import plot_imp_dists, plot_imp_boxplots
from autoimpute.visuals import plot_imp_swarm, plot_imp_strip
from autoimpute.visuals import plot_imp_scatter

# autoimpute imports - imputations & analysis
from autoimpute.imputations import MultipleImputer
from autoimpute.analysis import MiLinearRegression

# reading the full dataset and mar dataset
full = pd.read_csv("full.csv")
mar = pd.read_csv("mar.csv")
\`\`\`
`

const marPercent = `
### Percent Missing by Feature
---
\`\`\`python
'''MAR percent plot'''
plot_md_percent(mar)
\`\`\`
`

const marLocation = `
### Location of Missingness by Feature
---
\`\`\`python
'''MAR location plot'''
plot_md_locations(mar)
\`\`\`
`

const marMean = `
### Mean Imputation
---
\`\`\`python
'''MAR mean imputation'''

# create the mean imputer
mi_mean_mar = MultipleImputer(
    strategy="mean", n=5, return_list=True, seed=101
)

# print the mean imputer to console
print(mi_mean_mar)

# perform mean imputation procedure
imp_mean_mar = mi_mean_mar.fit_transform(mar)
\`\`\`
`

const marMeanDistBox = `
### Distribution Plots after Mean Imputation
---
\`\`\`python
'''MAR distribution plots after mean imputation'''

# distribution plot for mean imputation
plot_imp_dists(
    d=imp_mean_mar,
    mi=mi_mean_mar, 
    col="x",
    title="Distributions after Mean Imputation: MAR",
    separate_observed=False,
    hist_observed=True,
    hist_imputed=False
)

# box plot for mean imputation
plot_imp_boxplots(
    d=imp_mean_mar,
    mi=mi_mean_mar,
    col="x",
    title="Boxplots after Mean Imputation: MAR"
)

# strip plot for mean imputation
plot_imp_strip(
    d=imp_mean_mar,
    mi=mi_mean_mar,
    col="x",
    title="Imputed vs Observed Dists after Mean Imputation: MAR"
)
\`\`\`
`

const marLs = `
### Distribution Plots after Least Squares Imputation
---
\`\`\`python
'''MAR distribution plots after least squares imputation'''

# create the least squares imputer
mi_ls_mar = MultipleImputer(
    strategy="least squares", n=5, return_list=True, seed=101
)

# perform least squares imputation procedure
imp_ls_mar = mi_ls_mar.fit_transform(mar)

# distribution plot for least squares imputation
plot_imp_dists(
    d=imp_ls_mar,
    mi=mi_ls_mar, 
    col="x",
    title="Distributions after Least Squares Imputation: MAR",
    separate_observed=False,
    hist_observed=True,
    hist_imputed=False
)

# box plot for least squares imputation
plot_imp_boxplots(
    d=imp_ls_mar,
    mi=mi_ls_mar,
    col="x",
    title="Boxplots after Least Squares Imputation: MAR"
)

# strip plot for least squares imputation
plot_imp_strip(
    d=imp_ls_mar,
    mi=mi_ls_mar,
    col="x",
    title="Imputed vs Observed Dists after Least Squares Imputation: MAR"
)
\`\`\`
`

const marPmm = `
### Distribution Plots after PMM Imputation
---
\`\`\`python
'''MAR distribution plots after PMM imputation'''

# create the PMM imputer
mi_pmm_mar = MultipleImputer(
    strategy="pmm", n=5, return_list=True, seed=101
)

# perform PMM imputation procedure
imp_pmm_mar = mi_pmm_mar.fit_transform(mar)

# distribution plot for PMM imputation
plot_imp_dists(
    d=imp_pmm_mar,
    mi=mi_pmm_mar, 
    col="x",
    title="Distributions after PMM Imputation: MAR",
    separate_observed=False,
    hist_observed=True,
    hist_imputed=False
)

# box plot for PMM imputation
plot_imp_boxplots(
    d=imp_pmm_mar,
    mi=mi_pmm_mar,
    col="x",
    title="Boxplots after PMM Imputation: MAR"
)

# swarm plot for PMM imputation
plot_imp_swarm(
    d=imp_pmm_mar,
    mi=mi_pmm_mar,
    col="x",
    title="Imputed vs Observed Dists after PMM Imputation: MAR"
)
\`\`\`
`

const marRegression = `
### Linear Regression on Multiply Imputed Data
---
\`\`\`python
'''Regression after Multiple Imputation on MAR'''

# NOTE: Full, Listwise delete, and bias code not included

# create the regression using custom imputers
lm_mean_mar = MiLinearRegression(mi=mi_mean_mar)
lm_ls_mar = MiLinearRegression(mi=mi_ls_mar)
lm_pmm_mar = MiLinearRegression(mi=mi_pmm_mar)
models_mar = [lm_mean_mar, lm_ls_mar, lm_pmm_mar]

# a bit of manipulation to create one dataframe
get_coeff = lambda lm_, x: lm_.summary().loc["x"].to_frame()
res_mar = pd.concat([get_coeff(lm, "x") for lm in models_mar], axis=1)
res_mar.columns = ["mean", "least squares", "pmm"]
res_mar = res_mar.T[["coefs", "std", "vw", "vb", "vt"]]

# show the results
res_mar
\`\`\`
`

const marScatter = `
### Scatterplots after Imputation Methods
---
\`\`\`python
'''Visualizing effect of imputation on regression for scatter'''

# scatterplot for mean
plot_imp_scatter(
    d=mar, x="x", y="y", strategy="mean", color="x",
    title="Scatter after Mean Imputation: MAR"
)

# scatterplot for least squares
plot_imp_scatter(
    d=mar, x="x", y="y", strategy="least squares", color="x",
    title="Scatter after Least Squares Imputation: MAR"
)

# scatterplot for pmm
plot_imp_scatter(
    d=mar, x="x", y="y", strategy="pmm", color="x",
    title="Scatter after PMM Imputation: MAR"
)

\`\`\`
`


class EndToEnd extends Component {
    render() {
      return (
        <div className="end-to-end">
          <div className="ete-header-01">
            <ReactMarkdown source={inputHeader01} escapeHtml={false} renderers={{code: CodeBlock}} />
          </div>
          <div className="ete-mcar">
            <ReactMarkdown source={mcarHeader} escapeHtml={false} renderers={{code: CodeBlock}} />
            <ReactMarkdown source={mcarDataPrep} escapeHtml={false} renderers={{code: CodeBlock}} />
            <ReactMarkdown source={mcarPercent} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="mcarPercent" className="ete-percent" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-plot-md-percent.png"></img>
            <ReactMarkdown source={mcarLocation} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="mcarLocation" className="ete-locations" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-plot-md-locations.png"></img>
            <ReactMarkdown source={mcarMean} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="mcarMeanImputer" className="ete-mean-imputer" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mean-imputer.png"></img>
            <ReactMarkdown source={mcarMeanDistBox} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="mcarMeanImputerDist" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-mean-dist.png"></img>
            <img alt="mcarMeanImputerBox" className="ete-box" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-mean-box.png"></img>
            <img alt="mcarMeanImputerStrip" className="ete-strip" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-mean-strip.png"></img>
            <ReactMarkdown source={mcarLs} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="mcarLsImputerDist" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-ls-dist.png"></img>
            <img alt="mcarLsImputerBox" className="ete-box" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-ls-box.png"></img>
            <img alt="mcarLsImputerStrip" className="ete-strip" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-ls-strip.png"></img>
            <ReactMarkdown source={mcarPmm} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="mcarPmmOutput" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/pmm-bayes-mcar.png"></img>
            <img alt="mcarPmmImputerDist" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-pmm-dist.png"></img>
            <img alt="mcarPmmImputerBox" className="ete-box" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-pmm-box.png"></img>
            <img alt="mcarPmmImputerSwarm" className="ete-swarm" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-pmm-swarm.png"></img>
            <ReactMarkdown source={mcarRegression} escapeHtml={false} renderers={{code: CodeBlock}} />
            <table border="1" className="dataframe">  
                <thead>    
                    <tr>      
                        <th>method</th>      
                        <th>coefs</th>      
                        <th>std</th>      
                        <th>vw</th>     
                        <th>vb</th>      
                        <th>vt</th>
                        <th>bias</th>    
                    </tr> 
                </thead>
                <tbody>    
                    <tr>      
                        <td><b>full</b></td>      
                        <td>0.70000</td>      
                        <td>0.03200</td>      
                        <td>0.00102</td>      
                        <td>0.00000</td>      
                        <td>0.00102</td>
                        <td>0%</td>    
                    </tr>    
                    <tr>      
                        <td><b>delete</b></td>      
                        <td>0.73915</td>      
                        <td>0.04682</td>      
                        <td>0.00219</td>      
                        <td>0.00000</td>      
                        <td>0.00219</td>
                        <td className="pos-bias">5.6%</td>    
                    </tr>    
                    <tr>      
                        <td><b>mean</b></td>      
                        <td>0.39330</td>      
                        <td>0.03056</td>      
                        <td>0.00093</td>      
                        <td>0.00000</td>      
                        <td>0.00093</td>
                        <td className="neg-bias">-43.8%</td>    
                    </tr>    
                    <tr>      
                        <td><b>ls</b></td>      
                        <td>0.73915</td>      
                        <td>0.02570</td>      
                        <td>0.00066</td>      
                        <td>0.00000</td>      
                        <td>0.00066</td>
                        <td className="pos-bias">5.6%</td>    
                        </tr>    
                    <tr>      
                        <td><b>pmm</b></td>      
                        <td>0.67692</td>      
                        <td>0.05404</td>      
                        <td>0.00128</td>      
                        <td>0.00136</td>      
                        <td>0.00292</td>
                        <td className="neg-bias">-3.3%</td>        
                    </tr>  
                    </tbody>
                </table>
            <ReactMarkdown source={mcarScatter} escapeHtml={false} renderers={{code: CodeBlock}} /> 
            <img alt="mcarMeanScatter" className="ete-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-mean-scatter.png"></img>
            <img alt="mcarLsScatter" className="ete-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-ls-scatter.png"></img>
            <img alt="mcarPmmScatter" className="ete-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mcar-pmm-scatter.png"></img>
                       
          </div>
          <div className="ete-mar">
            <ReactMarkdown source={marHeader} escapeHtml={false} renderers={{code: CodeBlock}} />
            <ReactMarkdown source={marDataPrep} escapeHtml={false} renderers={{code: CodeBlock}} />
            <ReactMarkdown source={marPercent} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="marPercent" className="ete-percent"  src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-plot-md-percent.png"></img>
            <ReactMarkdown source={marLocation} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="marLocation" className="ete-locations" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-plot-md-locations.png"></img>
            <ReactMarkdown source={marMean} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="marMeanImputer" className="ete-mean-imputer" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mean-imputer.png"></img>
            <ReactMarkdown source={marMeanDistBox} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="marMeanImputerDist" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-mean-dist.png"></img>
            <img alt="marMeanImputerBox" className="ete-box" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-mean-box.png"></img>
            <img alt="marMeanImputerStrip" className="ete-strip" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-mean-strip.png"></img>
            <ReactMarkdown source={marLs} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="marLsImputerDist" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-ls-dist.png"></img>
            <img alt="marLsImputerBox" className="ete-box" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-ls-box.png"></img>
            <img alt="marLsImputerStrip" className="ete-strip" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-ls-strip.png"></img>
            <ReactMarkdown source={marPmm} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="marPmmOutput" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/pmm-bayes-mar.png"></img>
            <img alt="marPmmImputerDist" className="ete-dist" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-pmm-dist.png"></img>
            <img alt="marPmmImputerBox" className="ete-box" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-pmm-box.png"></img>
            <img alt="marPmmImputerSwarm" className="ete-swarm" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-pmm-swarm.png"></img>
            <ReactMarkdown source={marRegression} escapeHtml={false} renderers={{code: CodeBlock}} />
            <table border="1" className="dataframe">
                <thead>
                    <tr>
                        <th>method</th>
                        <th>coefs</th>
                        <th>std</th>
                        <th>vw</th>
                        <th>vb</th>      
                        <th>vt</th>
                        <th>bias</th>    
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><b>full</b></td>      
                        <td>0.70000</td>      
                        <td>0.03200</td>      
                        <td>0.00102</td>      
                        <td>0.00000</td>      
                        <td>0.00102</td>
                        <td>0%</td>    
                    </tr>    
                    <tr>      
                        <td><b>delete</b></td>      
                        <td>0.66180</td>      
                        <td>0.03865</td>      
                        <td>0.00149</td>      
                        <td>0.00000</td>      
                        <td>0.00149</td>
                        <td className="neg-bias">-5.5%</td>    
                    </tr>    
                    <tr>      
                        <td><b>mean</b></td>      
                        <td>0.66180</td>      
                        <td>0.04896</td>      
                        <td>0.00240</td>      
                        <td>0.00000</td>      
                        <td>0.00240</td>
                        <td className="neg-bias">-5.5%</td>     
                    </tr>    
                    <tr>      
                        <td><b>ls</b></td>      
                        <td>0.86053</td>      
                        <td>0.02889</td>      
                        <td>0.00083</td>      
                        <td>0.00000</td>      
                        <td>0.00083</td> 
                        <td className="pos-bias">22.9%</td>    
                    </tr>    
                    <tr>      
                        <td><b>pmm</b></td>      
                        <td>0.69690</td>     
                        <td>0.03707</td>      
                        <td>0.00096</td>      
                        <td>0.00034</td>      
                        <td>0.00137</td>
                        <td className="neg-bias">-0.44%</td>     
                    </tr>  
                </tbody>
            </table>
            <ReactMarkdown source={marScatter} escapeHtml={false} renderers={{code: CodeBlock}} />
            <img alt="marMeanScatter" className="ete-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-mean-scatter.png"></img>
            <img alt="marLsScatter" className="ete-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-ls-scatter.png"></img>
            <img alt="marPmmScatter" className="ete-scatter" src="https://kearnz.github.io/autoimpute-tutorials/img/ete/ete-mar-pmm-scatter.png"></img>

          </div>
        </div>
      );
    }
  }

export default EndToEnd;