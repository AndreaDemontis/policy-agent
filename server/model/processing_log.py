from server import app

db = app.db


class Processing_Log(db.Model):
    __tablename__ = 'processing_log'

    service = db.Column(db.VARCHAR(256), nullable=False, primary_key=True)


