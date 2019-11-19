import React from 'react';

import './input-box.css';

export default class InputBox extends React.Component
{
    constructor(props)
    {
        super(props);

        // - Bindings
        this.handleChange = this.handleChange.bind(this);
        this.handleSend = this.handleSend.bind(this);

        this.state =
        {
            value: ""
        }
    }

    render = () =>
    (
        <input
            type="text"
            className="input-container"
            placeholder="Ask us something by writing or using the microphone button"
            onKeyUp={this.handleSend}
            onChange={this.handleChange}
            value={this.state.value}
        />
    )

    handleChange(event)
    {
        this.setState({value: event.target.value});
    }

    handleSend(event)
    {
        if (event.keyCode === 13)
        {
            if (this.props.onSend)
                this.props.onSend(this.state.value);

            this.setState({value: ""});
        }
    }
}
