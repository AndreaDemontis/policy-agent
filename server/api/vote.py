from flask_restplus import Namespace, Resource
from flask import request
# -------------------------------------------------
from server import app
from server.session import Session

from server.model.tag_relation import Tag_Relation as DB_Tag_Relation

# - API Namespace ---------------------------------
ns = Namespace('Vote', description="Vote tag apis")

db = app.db

@ns.route('/<string:id>/<string:type>', endpoint="Vote")
class Vote(Resource):

    def put(self, id, type):

        session = request.headers.get('session')
        session = next((x for x in app.sessions if x.get_id() == int(session)), None)

        mod = session.vote(id, type)

        # 1 Upvote command with no previous votes
        # 2 Upvote command with previous downvote
        # 0 Upvote command with previous upvote or Downvote command with previous dowvote
        # -1 Downvote command with no previoous votes
        # -2 Downvote command with previous upvote

        if mod != 0:
            if mod>0:
                db.session.query(DB_Tag_Relation).filter(DB_Tag_Relation.id == id).update({DB_Tag_Relation.positive_feedback: DB_Tag_Relation.positive_feedback +1}, synchronize_session=False)
                if mod==2:
                    db.session.query(DB_Tag_Relation).filter(DB_Tag_Relation.id == id).update({DB_Tag_Relation.negative_feedback: 0 if DB_Tag_Relation.negative_feedback - 1 < 0 else DB_Tag_Relation.negative_feedback - 1}, synchronize_session=False)
            elif mod<0:
                db.session.query(DB_Tag_Relation).filter(DB_Tag_Relation.id == id).update({DB_Tag_Relation.negative_feedback: DB_Tag_Relation.negative_feedback + 1}, synchronize_session=False)
                if mod == -2:
                    db.session.query(DB_Tag_Relation).filter(DB_Tag_Relation.id == id).update({DB_Tag_Relation.positive_feedback: 0 if DB_Tag_Relation.positive_feedback - 1 < 0 else DB_Tag_Relation.positive_feedback - 1}, synchronize_session=False)

            db.session.commit()
