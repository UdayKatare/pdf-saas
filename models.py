from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class UploadRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    filename = db.Column(db.String(256), nullable=False)
    filesize = db.Column(db.Integer, nullable=False)
    summary_done = db.Column(db.Boolean, default=False)
    question_count = db.Column(db.Integer, default=0)
