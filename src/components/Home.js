import React, { Component } from "react";
import ReactMarkdown from "react-markdown";
import CodeBlock from "./CodeBlock";

const welcome = `
## Welcome to Autoimpute!
---
[![PyPI version](https://badge.fury.io/py/autoimpute.svg)](https://badge.fury.io/py/autoimpute) [![Build Status](https://travis-ci.com/kearnz/autoimpute.svg?branch=master)](https://travis-ci.com/kearnz/autoimpute) [![Documentation Status](https://readthedocs.org/projects/autoimpute/badge/?version=latest)](https://autoimpute.readthedocs.io/en/latest/?badge=latest) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/) [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

[Autoimpute](https://pypi.org/project/autoimpute/) is a Python package for **handling missing data**. Install it from PyPI with:

\`\`\`python
pip install autoimpute
\`\`\`
`

const getStarted = `

### Getting Started
---
Autoimpute is designed to be user friendly and flexible.

Imputations can be as simple as:
\`\`\`python

from autoimpute.imputations import MiceImputer
imp = MiceImputer()
imp.fit_transform(data)
\`\`\`

Analysis of multiply imputed data is easy too:

\`\`\`python

from autoimpute.analysis import MiLinearRegression
imp_lm = MiLinearRegression()
imp_lm.fit(X_train, y_train)
imp_lm.summary()
predictions = imp_lm.predict(X_test)
`

const backgroundText = `

### Main Features
---
* Utility functions to examine patterns in missing data
* Missingness classifier and automatic missing data test set generator
* Numerous imputation methods for continuous, categorical, and time-series data
* Single, Multiple, and MICE imputer classes to apply imputation methods
* Custom visualization support for utility functions and imputation methods
* Analysis methods and pooled parameter inference using multiply imputed datasets
* Adherence to \`sklearn\` API design for imputation and analysis classes
* Direct integration with \`pandas\`, \`statsmodels\`, \`pymc3\`, and more
`

class Home extends Component {
  render() {
    return (
        <div className="home-page">
            <div className="home-page-left">
                <ReactMarkdown source={welcome} escapeHtml={false} renderers={{code: CodeBlock}} />
                <ReactMarkdown source={getStarted} escapeHtml={false} renderers={{code: CodeBlock}} />
            </div>
            <div className="hompe-page-right">
                <img alt="autoimpute-logo" className="autoimpute-logo" src="https://kearnz.github.io/autoimpute-tutorials/img/home/autoimpute-logo-transparent.png"></img>
                <ReactMarkdown source={backgroundText} escapeHtml={false} renderers={{code: CodeBlock}} />
            </div>
        </div>
    );
  }
}

export default Home;
