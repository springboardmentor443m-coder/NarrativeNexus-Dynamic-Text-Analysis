import os

class Config:
    """Application configuration settings"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000


