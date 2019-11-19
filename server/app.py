# - - - FLASK ---------------------------------
from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
# - - - DIALOGFLOW --------------------------------
import dialogflow_v2 as DialogFlow
# - - - UTILITIES ---------------------------------
import yaml, os, logging
# -------------------------------------------------

class App(object):

    def __init__(self, path):
        super(App, self).__init__()

        # - Create a new flask app
        self.flask = Flask(__name__, static_folder='../client/build')
        self.path = path

    def configure(self, config):

        # - Set flask configuration
        self.flask.config.update(config)

        # - Enable cors if necessary
        if self.flask.config["server"]["cors"]:
            paths = config["server"]["cors"]["paths"]
            cors = CORS(self.flask, resources=paths)

        # - Initialize dialog flow apis credentials
        key = config["server"]["dialogflow"]["apikey"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key

        # - Set database credentials
        connectUri = config["server"]["database"]["connectionUri"]
        self.flask.config['SQLALCHEMY_DATABASE_URI'] = connectUri
        self.db = SQLAlchemy(self.flask)

    def run(self):
        from .api import register

        # - Initialize database models
        from server.model.test import User

        # - Reflect changes on the real database
        self.db.create_all()

        # - Initialize dialog flow apis with credentials
        self.dialog = DialogFlow.SessionsClient()

        # - Prepare server sessions
        self.sessions = []

        # - Api initialization
        register(self)

        # Serve React App
        @self.flask.route('/', defaults={'path': ''})
        @self.flask.route('/<path:path>')
        def serve(path):
            if path != "" and os.path.exists(self.flask.static_folder + '/' + path):
                return send_from_directory(self.flask.static_folder, path)
            else:
                return send_from_directory(self.flask.static_folder, 'index.html')


        host = self.flask.config["server"]["ip"]
        ssl = 'adhoc' if self.flask.config["server"]["ssl"] else None
        port = self.flask.config["server"]["port"]
        debug = self.flask.config["debug"]
        threaded = self.flask.config["server"]["threaded"]

        # - Start application
        self.flask.run(ssl_context=ssl, host=host, use_reloader=debug, port=port, threaded=threaded)
