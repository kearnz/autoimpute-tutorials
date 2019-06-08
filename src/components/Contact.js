import React, { Component } from "react";
import ReactMarkdown from "react-markdown"; 
import CodeBlock from "./CodeBlock";

const getInTouch = `
## Get in touch with us about Autoimpute!
---
The info below outlines the best method to reach the authors, depending on your inquiry or message.

#### Issues with Autoimpute or Bugs in the Package
* If you find an issue with Autoimpute or a bug in the source code, please raise an issue or bug on our github.  
* For bug reports, follow the bug report template. For issues, follow the issue template.  
* More information here: https://github.com/kearnz/autoimpute/issues

#### New Features for Autoimpute
* If you have new feature requests or would like to collaborate on the package, please issue a pull request.  
* There is no pull request template yet, but please follow standards from other packages in the python ecosystem.  
* More information here: https://github.com/kearnz/autoimpute/pulls

#### General Inquiries, Questions, and Feedback
* For everything else outside of bugs, issues, and pull requests, you can get in touch with the authors directly.

|Source   |Joseph Kearney           |Shahid Barkat              |
|:--------|:------------------------|:--------------------------|
|U Chicago| jkearney@uchicago.edu   |shahidb@uchicago.edu       |
|Personal |josephkearney14@gmail.com|shahidbarkat@gmail.com     |
|Github   |https://github.com/kearnz|https://github.com/shabarka|

`

class Contact extends Component {
  render() {
    return (
      <div className="contact">
        <ReactMarkdown source={getInTouch} escapeHtml={false} renderers={{code: CodeBlock}} />
      </div>
    );
  }
}
 
export default Contact;