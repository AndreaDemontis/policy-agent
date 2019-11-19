from flask_restplus import Api, Namespace, Resource
from flask import Blueprint
# -------------------------------------------------

from .chat import ns as chat_namespace
from .policy import ns as policy_namespace
from .vote import ns as vote_namespace

def register(app):

    # - - - Api framework
    blueprint = Blueprint('api', __name__)
    app.api = Api(blueprint, title="Policy agent", strict_slashes=True)

    # - Remove default api namespace
    app.api.namespaces.pop(0)

    # - Api namespaces
    app.api.add_namespace(chat_namespace, path="/chat")
    app.api.add_namespace(policy_namespace, path="/policy")
    app.api.add_namespace(vote_namespace, path="/vote")

    # - Api documentation json schema
    @app.api.hide
    @app.api.route('/schema')
    class Schema(Resource):

        def get(self):
            return app.api.__schema__


    app.flask.register_blueprint(blueprint, url_prefix='/api')
