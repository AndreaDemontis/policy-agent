from flask_restplus import Namespace, Resource
from flask import request
import base64
# -------------------------------------------------
from server import app
from server.session import Session

# - API Namespace ---------------------------------
ns = Namespace('Chat', description="Chat apis")
# - - - DIALOGFLOW --------------------------------
import dialogflow_v2 as DialogFlow
import google.protobuf  as pf
# -------------------------------------------------

@ns.route('', endpoint="Users")
class Chat(Resource):


    def get(self):

        session = request.headers.get('session')
        session = next((x for x in app.sessions if x.get_id() == int(session)), None)

        return {"messages": session.get_messages()}, 200

    def put(self):

        # - Create a new session
        session = Session()

        # - Add the new session
        app.sessions.append(session)

        return {"id": session.get_id()}, 200

    def post(self):

        msg = request.get_json()["message"]
        audio = request.get_json()["audio"]

        session = request.headers.get('session')
        session = next((x for x in app.sessions if x.get_id() == int(session)), None)
        session_path = app.dialog.session_path("policy-agent-stytig", session.get_id())

        if not audio:
            text_input = DialogFlow.types.TextInput(text=msg, language_code='en-US')
            query_input = DialogFlow.types.QueryInput(text=text_input)

            # - Send the request to dialogflow which will give us the intent response
            response = app.dialog.detect_intent(session_path, query_input=query_input)

            if response.query_result.fulfillment_text:
                webhook_payload = response.query_result.webhook_payload
                payload = pf.json_format.MessageToDict(webhook_payload, including_default_value_fields=False)

                # - Add the messages in the session message list
                session.push_message(msg, "guest", {})
                session.push_message(response.query_result.fulfillment_text, "agent", payload)

        else:
            msg = base64.decodestring(msg)

            # - Setting up audio parameters (Can be parametrized by client/server side)
            audio_input = DialogFlow.types.InputAudioConfig(language_code='en-US', sample_rate_hertz=16000, audio_encoding=DialogFlow.enums.AudioEncoding.AUDIO_ENCODING_OGG_OPUS)
            query_input = DialogFlow.types.QueryInput(audio_config=audio_input)

            # - Send the audio request to dialogflow which will give us the intent response
            response = app.dialog.detect_intent(session_path, query_input=query_input, input_audio=msg)

            if response.query_result.fulfillment_text:
                webhook_payload = response.query_result.webhook_payload
                payload = pf.json_format.MessageToDict(webhook_payload, including_default_value_fields=False)

                # - Add the messages in the session message list
                session.push_message(response.query_result.query_text, "guest", {})
                session.push_message(response.query_result.fulfillment_text, "agent", payload, base64.encodestring(response.output_audio))


        return 200

    def delete(self):

        session = request.headers.get('session')
