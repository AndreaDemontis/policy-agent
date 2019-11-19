from server import app

db = app.db


class Paragraph(db.Model):
    __tablename__ = 'paragraph'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.UnicodeText, nullable=False)
    service = db.Column(db.VARCHAR(256), nullable=False)
    source_url = db.Column(db.VARCHAR(256))
    policy_date = db.Column(db.DateTime())


