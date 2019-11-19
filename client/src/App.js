import React from 'react';

import './App.css';

import { ReactComponent as Police} from './police.svg';

import MessageView from './chat/message-view';
import InputBox from './chat/input-box';
import RecorderInterface from './chat/recorder';

export default class App extends React.Component
{
    constructor(props)
    {
        super(props);

        // - Bindings
        this.sendMessage = this.sendMessage.bind(this);
        this.sendMessageAudio = this.sendMessageAudio.bind(this);
        this.vote = this.vote.bind(this);

        // - State initialization
        this.state =
        {
            session: 1,
            writing: false,
            messages:
            [
            ]
        }

        // - Default API call settings
        this.APISettings =
        {
            mode: 'cors',
            cache: 'no-cache',
            credentials: 'same-origin'
        }
    }

    componentDidMount()
    {
        let params = { ...this.APISettings, method: 'PUT' }

        // - API Call to start a new conversation session
        fetch('http://127.0.0.1:5000/api/chat', params)
            .then(res => res.json())
            .then((data) =>
            {
                // - We save the session id in the application's state
                this.setState({ session: data.id })
            })
            .catch(console.log)
    }

    render = () =>
    (
        <div className="app">

            <header>
                <div>POLICY</div>
                <div><Police /></div>
                <div>AGENT</div>
            </header>

            <div className="body">
                <div style={{width: "100%", height: "80%"}}>
                    <MessageView messages={this.state.messages} onVote={this.vote} writing={this.state.writing} />
                </div>

                <div style={{width: "80%"}}><InputBox onSend={this.sendMessage} /></div>

                <div><RecorderInterface onSend={this.sendMessageAudio} /></div>
            </div>

        </div>
    )

    vote(id, type)
    {
        var messages = this.state.messages.map((x) => !x.payload.policy ? x : { ...x, payload: { policy: x.payload.policy.map((y) =>
        {
            if (y.id === id)
            {
                if (y.vote == type) y.vote = null;
                else y.vote = type;

                let params = { ...this.APISettings, method: 'PUT', headers: { 'session': this.state.session } }

                // - This API Call handles an up/down vote for a paragraph match
                fetch('http://127.0.0.1:5000/api/vote/' + y.id + '/' + (y.vote ? type : "none"), params);
            }

            return y;
        })}});

        this.setState({ messages: messages });
    }

    sendMessageAudio(data)
    {
        this.sendMessage(data, true);
    }

    sendMessage(data, audio=false)
    {
        let sendData =
        {
            message: data,
            audio: audio
        }

        if (!audio)
            // - If we sending a text message we can immediately show it in the chat
            this.setState({ writing: true, messages: [...this.state.messages, { message: data, user: 'guest'}]});
        else
            // - Else we have to parse the audio > we show only the loading
            this.setState({ writing: true });

        let params =
        {
            ...this.APISettings,
            method: 'POST',
            headers: { 'session': this.state.session, 'Content-Type': 'application/json' },
            body: JSON.stringify(sendData)
        }

        // - We send the message data to the server
        fetch('http://127.0.0.1:5000/api/chat', params)
            .then(res => res.json())
            .then((data) =>
            {
                let params = { ...this.APISettings, method: 'GET', headers: { 'session': this.state.session } }

                // - After the request is completed we get the list of messages for the
                // - purpose of refreshing the chat
                fetch('http://127.0.0.1:5000/api/chat', params)
                    .then(res => res.json())
                    .then((data) =>
                    {
                        // - We add the vote property in all messages
                        var messages = data.messages.map((x) => !x.payload.policy ?
                        x : { ...x, payload:{ policy: x.payload.policy.map((y) =>
                        ({ ...y, vote: null }))}});

                        // - Update the interface and the application state
                        this.setState({ messages: messages, writing: false});

                        // - If we sent an audio message we should play the response
                        if (messages[messages.length - 1].audio !== "")
                        {
                            var voice = new Audio();
                            voice.src = "data:audio/ogg;base64," + messages[messages.length - 1].audio;
                            voice.play();
                        }
                    })
                    .catch(console.log)
            })
            .catch(console.log)
    }
}
