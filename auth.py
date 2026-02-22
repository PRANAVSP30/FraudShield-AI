import bcrypt

USERNAME = "admin"
PASSWORD_HASH = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())

def authenticate(username, password):
    if username == USERNAME:
        return bcrypt.checkpw(password.encode(), PASSWORD_HASH)
    return False