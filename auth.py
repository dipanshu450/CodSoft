import streamlit as st
import database as db
import re
from datetime import datetime, timedelta

# Regex patterns for validation
USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]{3,20}$')
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PASSWORD_PATTERN = re.compile(r'^.{8,}$')  # At least 8 characters

def init_session_state():
    """Initialize session state variables for authentication."""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'login_message' not in st.session_state:
        st.session_state.login_message = None
    if 'register_message' not in st.session_state:
        st.session_state.register_message = None


def register_user(username, email, password, confirm_password):
    """
    Register a new user.
    
    Args:
        username: Desired username
        email: User's email address
        password: Desired password
        confirm_password: Password confirmation
        
    Returns:
        message: Success or error message
        success: Boolean indicating success or failure
    """
    # Validate inputs
    if not username or not email or not password or not confirm_password:
        return "All fields are required", False
        
    if not USERNAME_PATTERN.match(username):
        return "Username must be 3-20 characters and contain only letters, numbers, and underscores", False
        
    if not EMAIL_PATTERN.match(email):
        return "Please enter a valid email address", False
        
    if not PASSWORD_PATTERN.match(password):
        return "Password must be at least 8 characters long", False
        
    if password != confirm_password:
        return "Passwords do not match", False
    
    # Attempt to create user
    try:
        user_id = db.create_user(username, email, password)
        if user_id:
            return "Registration successful! You can now log in.", True
        else:
            return "Username or email already exists", False
    except Exception as e:
        return f"Error during registration: {str(e)}", False


def login_user(username, password):
    """
    Log in a user.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        message: Success or error message
        success: Boolean indicating success or failure
    """
    if not username or not password:
        return "Username and password are required", False
    
    try:
        user = db.authenticate_user(username, password)
        if user:
            st.session_state.user = user
            st.session_state.authenticated = True
            return "Login successful!", True
        else:
            return "Invalid username or password", False
    except Exception as e:
        return f"Error during login: {str(e)}", False


def logout_user():
    """Log out the current user."""
    st.session_state.user = None
    st.session_state.authenticated = False
    return "You have been logged out.", True


def is_authenticated():
    """Check if the user is authenticated."""
    return st.session_state.authenticated and st.session_state.user is not None


def get_current_user():
    """Get the current user from session state."""
    return st.session_state.user


def display_login_form():
    """Display the login form and handle submissions."""
    with st.form("login_form"):
        st.subheader("Log In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("Log In")
        
        # Show message if exists
        if st.session_state.login_message:
            st.info(st.session_state.login_message[0])
            # Clear message after showing once
            if st.session_state.login_message[1]:  # If success
                st.session_state.login_message = None
    
    # Handle form submission
    if submit_button:
        message, success = login_user(username, password)
        st.session_state.login_message = (message, success)
        if success:
            st.rerun()  # Refresh page to show authenticated content


def display_registration_form():
    """Display the registration form and handle submissions."""
    with st.form("registration_form"):
        st.subheader("Create an Account")
        username = st.text_input("Username (3-20 characters, letters, numbers, underscores)")
        email = st.text_input("Email Address")
        password = st.text_input("Password (minimum 8 characters)", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("Register")
        
        # Show message if exists
        if st.session_state.register_message:
            st.info(st.session_state.register_message[0])
            # Clear message after showing once
            if st.session_state.register_message[1]:  # If success
                st.session_state.register_message = None
    
    # Handle form submission
    if submit_button:
        message, success = register_user(username, email, password, confirm_password)
        st.session_state.register_message = (message, success)
        if success:
            # Add a login message to prompt the user to log in
            st.session_state.login_message = ("Registration successful! Please log in with your new account.", True)
            st.rerun()  # Refresh page to show success and login form


def display_auth_page():
    """Display the authentication page with login and registration forms."""
    st.title("🔐 Authentication")
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        display_login_form()
    
    with register_tab:
        display_registration_form()
    
    st.markdown("---")
    st.markdown("⚠️ Your account allows you to save your captioned images and manage privacy settings.")


def display_user_profile():
    """Display the user profile page."""
    user = get_current_user()
    
    if not user:
        st.error("User not found. Please log in again.")
        logout_user()
        st.rerun()
        return
    
    st.title("👤 User Profile")
    
    st.write(f"**Username:** {user['username']}")
    st.write(f"**Email:** {user['email']}")
    st.write(f"**Account Created:** {user['created_at'].strftime('%Y-%m-%d')}")
    
    # Logout button
    if st.button("Log Out"):
        logout_user()
        st.rerun()
    
    # Change password form
    with st.expander("Change Password"):
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password (minimum 8 characters)", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            submit_button = st.form_submit_button("Change Password")
            
            if submit_button:
                # Validate inputs
                if not current_password or not new_password or not confirm_password:
                    st.error("All fields are required")
                elif not PASSWORD_PATTERN.match(new_password):
                    st.error("New password must be at least 8 characters long")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                else:
                    # Verify current password
                    user_check = db.authenticate_user(user['username'], current_password)
                    if not user_check:
                        st.error("Current password is incorrect")
                    else:
                        # Update password
                        success = db.update_user_password(user['id'], new_password)
                        if success:
                            st.success("Password changed successfully!")
                        else:
                            st.error("Failed to change password")


def auth_required(func):
    """
    Decorator to require authentication for a page or function.
    Use this to protect pages that require login.
    """
    def wrapper(*args, **kwargs):
        if is_authenticated():
            return func(*args, **kwargs)
        else:
            st.warning("⚠️ You need to log in to access this feature.")
            display_login_form()
            return None
    return wrapper