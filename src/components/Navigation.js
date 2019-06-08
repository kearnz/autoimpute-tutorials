import React, { Component } from 'react';
import $ from 'jquery';
import Contact from "./Contact";
import Home from "./Home";
import Comparing from "./Comparing";
import ImputerI from "./ImputerI";
import ImputerII from "./ImputerII";
import ImputerIII from "./ImputerIII";
import EndToEnd from "./EndToEnd";


const NavItem = props => {
  const pageURI = window.location.pathname+window.location.search
  const liClassName = (props.path === pageURI) ? "nav-item active" : "nav-item";
  const aClassName = props.disabled ? "nav-link disabled" : "nav-link"
  return (
    <li className={liClassName}>
      <a onClick={() => props.onClick()} href={props.path} className={aClassName}>
        {props.name}
        {(props.path === pageURI) ? (<span className="sr-only">(current)</span>) : ''}
      </a>
    </li>
  );
}

class NavDropdown extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isToggleOn: false
    };
  }
  showDropdown(e) {
    e.preventDefault();
    this.setState(prevState => ({
      isToggleOn: !prevState.isToggleOn
    }));
  }
  render() {
    const classDropdownMenu = 'dropdown-menu' + (this.state.isToggleOn ? ' show' : '')
    return (
      <li className="nav-item dropdown">
        <a className="nav-link dropdown-toggle" id="navbarDropdown" role="button" data-toggle="dropdown"
          aria-haspopup="true" aria-expanded="false"
          onClick={(e) => {this.showDropdown(e)}}>
          {this.props.name}
        </a>
        <div className={classDropdownMenu} aria-labelledby="navbarDropdown">
          {this.props.children}
        </div>
      </li>
    )
  }
}


class Navigation extends React.Component {
  constructor(props) {
        // Required step: always call the parent class' constructor
        super(props);
    
        // Set the state directly. Use props if necessary.
        this.state = {
          activeKey: 1,
          df: "/autoimpute-tutorials/"
        }

        //this.handleClick = this.handleClick.bind(this);
  }
  handleClick(key) {
    //event.preventDefault();
    this.setState({activeKey: key});
    
    // NOT A GOOD SOLUTION - temporary hack to remove dropdown
    // but requires a double click on tutorial button next time
    // because state not altered, class just changed to trigger event
    // should not be using jquery for this moving forward
    if($('.dropdown-menu').hasClass('show')){
      $( ".dropdown-menu" ).trigger( "click" );
        $('.dropdown-menu').removeClass('show');
    }
  }
  render() {
    return (
     <div className="main-page">
      <nav className="navbar navbar-expand-lg">
        <a className="navbar-brand" href={this.state.df}>Autoimpute</a>
        <div className="collapse navbar-collapse" id="navbarSupportedContent">
          <ul className="navbar-nav mr-auto">
            <NavItem name="Home" onClick={this.handleClick.bind(this, 1)} />
            <NavItem name="Contact" onClick={this.handleClick.bind(this, 2)} />
            <NavDropdown name="Tutorials">
                <a className="dropdown-item" href={this.props.path} onClick={this.handleClick.bind(this, 3.2)}>Imputers: Part I</a>
                <a className="dropdown-item" href={this.props.path} onClick={this.handleClick.bind(this, 3.3)}>Imputers: Part II</a>
                <a className="dropdown-item" href={this.state.path} onClick={this.handleClick.bind(this, 3.4)}>Imputers: Part III</a>
                <a className="dropdown-item" href={this.state.path} onClick={this.handleClick.bind(this, 3.5)}>Comparing Imputation Methods</a>
                <a className="dropdown-item" href={this.state.path} onClick={this.handleClick.bind(this, 3.6)}>End-to-End Analysis</a>
            </NavDropdown>
          </ul>
        </div>
       </nav>
       <div className="content">
         {this.state.activeKey === 1 ? <Home/> : null}
         {this.state.activeKey === 2 ? <Contact/> : null}
         {this.state.activeKey === 3.2 ? <ImputerI/> : null}
         {this.state.activeKey === 3.3 ? <ImputerII/> : null}
         {this.state.activeKey === 3.4 ? <ImputerIII/> : null}
         {this.state.activeKey === 3.5 ? <Comparing/> : null}
         {this.state.activeKey === 3.6 ? <EndToEnd/> : null}
       </div>
      </div>
      
    )
  }
}

export default Navigation;