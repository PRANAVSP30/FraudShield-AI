import bcrypt

# Create hashed password once
def get_hashed_password():
    password = "admin123"
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# Store hashed password
HASHED_PASSWORD = get_hashed_password()

def authenticate(username, password):
    if username == "admin":
        return bcrypt.checkpw(password.encode(), HASHED_PASSWORD)
    return False