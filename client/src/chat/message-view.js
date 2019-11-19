import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronUp, faChevronDown } from '@fortawesome/free-solid-svg-icons';
import ReactTooltip from 'react-tooltip';

import './message-view.css';

export default class MessageView extends React.Component
{
    constructor(props)
    {
        super(props);

        this.upvoteTag = this.upvoteTag.bind(this);
        this.downvoteTag = this.downvoteTag.bind(this);
    }

    renderMessage = ({user, message, payload}, i) =>
    (
        <div className={user + " message"} key={i}>
            <span>{message}{payload && payload.date ? (<p>Collected on {payload.date}</p>) : ""}</span>
            {   payload && payload.policy ?
                payload.policy.map((x) =>
                    (<div>
                        <div className="vote">
                            <ReactTooltip effect="solid" className="tooltip"/>
                            <div data-tip="Upvote this match"><FontAwesomeIcon icon={faChevronUp} data-color="green" data-selected={x.vote === "upvote"} onClick={this.upvoteTag(x.id)}/></div>
                            <div>{x.positive_feedback - x.negative_feedback + (x.vote === "upvote" ? 1 : (x.vote === "downvote" ? -1 : 0))}</div>
                            <div data-tip="Downvote this match"><FontAwesomeIcon icon={faChevronDown} data-color="red" data-selected={x.vote === "downvote"} onClick={this.downvoteTag(x.id)} /></div>
                        </div>
                        <span className="content">{x.text}</span>
                        <div className="confidence">confidence {(x.confidence * 100).toFixed(2)}%</div>
                    </div>))
                : ""
            }
        </div>
    )

    writingMessage = () =>
    (
        <div className="agent message loading">
            <img src="waiting.gif" height="50" /> <span>Thinking...</span>
        </div>
    )

    render = () =>
    (
        <div className="container" ref={(e) => this.scroll = e}>
            {this.props.messages.map((m, i) => this.renderMessage(m, i))}
            {this.props.writing ? this.writingMessage() : ""}
        </div>
    )

    upvoteTag(id)
    {
        var that = this;
        return function()
        {
            if (that.props.onVote)
                that.props.onVote(id, 'upvote')
        }
    }

    downvoteTag(id)
    {
        var that = this;
        return function()
        {
            if (that.props.onVote)
                that.props.onVote(id, 'downvote')
        }
    }

    componentDidUpdate(prevProps)
    {
        this.scroll.scrollTop = this.scroll.scrollHeight;
    }
}
