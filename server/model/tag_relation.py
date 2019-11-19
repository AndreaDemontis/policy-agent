from server import app
from sqlalchemy import UniqueConstraint

db = app.db


class Tag_Relation(db.Model):
    __tablename__ = 'tag_relation'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    paragraph = db.Column(db.Integer, db.ForeignKey('paragraph.id'))
    tag = db.Column(db.VARCHAR(256))
    positive_feedback = db.Column(db.Integer, default=0)
    negative_feedback = db.Column(db.Integer, default=0)
    confidence_level = db.Column(db.Float, nullable=False)

    UniqueConstraint('paragraph', 'tag', name='uix_1')

