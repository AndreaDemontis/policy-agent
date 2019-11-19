import React from 'react';
import Recorder from 'opus-recorder';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone } from '@fortawesome/free-solid-svg-icons';


import "./recorder.css";

export default class RecorderInterface extends React.Component
{
    constructor(props)
    {
        super(props);

        this.state =
        {
            recording: false
        }

        this.micRecorder = null;


        this.startStopRecording = this.startStopRecording.bind(this);
    }

    render = () =>
    (
        <div className="recorder-container">
            <div className="line"></div>
            <div className={"button " + (this.state.recording ? "active" : "")} onClick={this.startStopRecording}>
                <FontAwesomeIcon icon={faMicrophone} />
            </div>
            <div className="line"></div>
        </div>
    )

    startStopRecording()
    {
        if (!this.state.recording)
        {
            this.setState({ recording: true });

            // - Utility function that converts blob data to a base64 url
            function BlobToDataURL(blob)
            {
                return new Promise((resolve, reject)=>
                {
                    const reader = new FileReader();
                    reader.addEventListener("loadend", e => resolve(String(reader.result)));
                    reader.readAsDataURL(blob);
                });
            }

            // - Handles OPUS audio recording from the microphone
            this.micRecorder = new Recorder(
            {
                encoderApplication: 2049,
                encoderSampleRate: 16000,
                originalSampleRateOverride: 16000,
                encoderPath: "./encoderWorker.min.js"
            });

            this.micRecorder.ondataavailable = async typedArray =>
            {
                // - We transform the data in a base64 string
                const audioData = new Blob([typedArray], {type: "audio/ogg"});
                const audioData_dataURL = await BlobToDataURL(audioData);
                const audioData_str = audioData_dataURL.replace(/^data:.+?base64,/, "");

                if (this.props.onSend)
                    this.props.onSend(audioData_str);
            };

            this.micRecorder.start();
        }
        else
        {
            this.micRecorder.stop();

            this.setState({ recording: false });
        }


    }
}
