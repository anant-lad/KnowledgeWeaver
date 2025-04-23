"""
Authentication UI components for the KnowledgeWeaver application.
"""
import streamlit as st
from utils.auth_utils import (
    authenticate_user, create_user_in_neo4j, login_user, logout_user,
    get_current_user
)
from utils.ui_components import show_status

def render_login_form(neo4j_url: str, neo4j_user: str, neo4j_password: str) -> bool:
    """
    Render login form

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        True if login successful, False otherwise
    """
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if not username or not password:
                show_status("Please enter both username and password", "error")
                return False

            result = authenticate_user(neo4j_url, neo4j_user, neo4j_password, username, password)

            if result["status"] == "success":
                login_user(result)
                # Close the popup
                st.session_state.show_auth_popup = False
                show_status(f"Welcome back, {username}!", "success")
                st.rerun()
                return True
            else:
                show_status(result["message"], "error")
                return False

    return False

def render_signup_form(neo4j_url: str, neo4j_user: str, neo4j_password: str) -> bool:
    """
    Render signup form

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        True if signup successful, False otherwise
    """
    with st.form("signup_form", clear_on_submit=False):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Sign Up")

        if submit_button:
            if not username or not email or not password or not confirm_password:
                show_status("Please fill in all fields", "error")
                return False

            if password != confirm_password:
                show_status("Passwords do not match", "error")
                return False

            result = create_user_in_neo4j(neo4j_url, neo4j_user, neo4j_password, username, email, password)

            if result["status"] == "success":
                # Switch to login mode after successful signup
                st.session_state.auth_popup_mode = "login"
                show_status("Account created successfully! Please log in.", "success")
                st.rerun()
                return True
            else:
                show_status(result["message"], "error")
                return False

    return False

def render_auth_page(neo4j_url: str, neo4j_user: str, neo4j_password: str) -> bool:
    """
    Render authentication page with login and signup forms

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        True if authenticated or anonymous access is allowed
    """
    # Check if already authenticated
    if st.session_state.authenticated:
        return True

    # Allow anonymous access - return True to show the app
    # Just show a warning about data not being saved
    st.warning("You are using the application in anonymous mode. Your data will not be saved permanently. Login to save your data.")

    # Check if we should show the auth popup
    if st.session_state.get("show_auth_popup", False):
        # Display the popup prominently
        render_auth_popup(neo4j_url, neo4j_user, neo4j_password)

    # Set a flag to indicate anonymous mode
    if 'anonymous_mode' not in st.session_state:
        st.session_state.anonymous_mode = True

    # Always return True to allow access to the app in anonymous mode
    return True

def render_auth_popup(neo4j_url: str, neo4j_user: str, neo4j_password: str):
    """
    Render authentication popup with login/signup forms

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
    """
    # Get the current auth popup mode
    auth_mode = st.session_state.get("auth_popup_mode", "login")

    # Create a popup using Streamlit's built-in columns and containers
    # First, create a full-width container
    popup_container = st.container()

    with popup_container:
        # Add custom CSS for styling the popup
        st.markdown("""
        <style>
        .auth-popup {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            margin: 20px auto;
            max-width: 500px;
            border: 1px solid #e0e0e0;
        }
        .popup-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid #f0f0f0;
            padding-bottom: 10px;
        }
        .close-button {
            cursor: pointer;
            font-size: 24px;
            color: #666;
        }
        .close-button:hover {
            color: #ff4b4b;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create a centered column for the popup
        _, popup_col, _ = st.columns([1, 3, 1])

        with popup_col:
            # Create the popup container with styling
            st.markdown('<div class="auth-popup">', unsafe_allow_html=True)

            # Popup header with title and close button
            st.markdown(f'''
            <div class="popup-header">
                <h3>🔑 {"Login" if auth_mode == "login" else "Sign Up"}</h3>
                <div class="close-button" onclick="closePopup()">✕</div>
            </div>
            <script>
            function closePopup() {{
                // This is just for visual effect, the actual closing happens with the button below
                document.querySelector('.auth-popup').style.display = 'none';
            }}
            </script>
            ''', unsafe_allow_html=True)

            # Hidden close button that will be triggered by JavaScript
            if st.button("Close", key="hidden_close_btn", help="Close the popup"):
                st.session_state.show_auth_popup = False
                st.rerun()

            # Show either login or signup form based on mode
            if auth_mode == "login":
                render_login_form(neo4j_url, neo4j_user, neo4j_password)
                st.markdown("Don't have an account?")
                if st.button("✏️ Sign Up", key="popup_signup_btn"):
                    st.session_state.auth_popup_mode = "signup"
                    st.rerun()
            else:  # signup
                render_signup_form(neo4j_url, neo4j_user, neo4j_password)
                st.markdown("Already have an account?")
                if st.button("🔑 Login", key="popup_login_btn"):
                    st.session_state.auth_popup_mode = "login"
                    st.rerun()

            # Close the popup div
            st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar_auth() -> bool:
    """
    Render authentication buttons in the sidebar

    Returns:
        True if authenticated, False otherwise
    """
    # Create a container in the sidebar for the auth buttons
    with st.sidebar:
        # Show login/signup buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔑 Login", key="sidebar_login_btn"):
                st.session_state.show_auth_popup = True
                st.session_state.auth_popup_mode = "login"
                st.rerun()
        with col2:
            if st.button("✏️ Sign Up", key="sidebar_signup_btn"):
                st.session_state.show_auth_popup = True
                st.session_state.auth_popup_mode = "signup"
                st.rerun()

    return False

def render_user_menu() -> None:
    """
    Render user menu for logged in users
    """
    user = get_current_user()
    if not user:
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 👤 {user['username']}")

    # Admin panel is only accessible by direct URL - no UI indication

    # Logout button
    if st.sidebar.button("🚪 Logout"):
        logout_user()
        st.rerun()

def render_auth_header() -> None:
    """
    Render authentication header with login/signup or user info
    """
    # Get current user
    user = get_current_user()

    # Create container for auth header
    auth_container = st.container()

    with auth_container:
        if user:
            # User is logged in
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### Welcome, {user['username']}!")
            with col2:
                if st.button("Logout", key="header_logout"):
                    logout_user()
                    st.rerun()
        else:
            # User is not logged in
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Login", key="header_login"):
                    st.session_state.show_auth_popup = True
                    st.session_state.auth_popup_mode = "login"
                    st.rerun()
            with col2:
                if st.button("Sign Up", key="header_signup"):
                    st.session_state.show_auth_popup = True
                    st.session_state.auth_popup_mode = "signup"
                    st.rerun()

def render_login_signup_buttons() -> None:
    """
    Render login and signup buttons for navigation bar
    """
    # Get current user
    user = get_current_user()

    # Create a container with CSS for positioning in the top right corner
    st.markdown("""
    <style>
    .auth-buttons-container {
        position: fixed;
        top: 0.5rem;
        right: 2rem;
        display: flex;
        gap: 0.5rem;
        z-index: 1000;
    }
    .auth-button {
        padding: 0.3rem 0.8rem;
        border-radius: 0.3rem;
        color: white;
        font-size: 0.9rem;
        text-align: center;
        text-decoration: none;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .login-button {
        background-color: #2196F3;
    }
    .signup-button {
        background-color: #4CAF50;
    }
    .user-button {
        background-color: #9C27B0;
    }
    .logout-button {
        background-color: #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

    if user:
        # User is logged in - show username and logout
        st.markdown(f"""
        <div class="auth-buttons-container">
            <div class="auth-button user-button">👤 {user['username']}</div>
            <a href="/" onclick="logoutUser()" style='text-decoration: none;'>
                <div class="auth-button logout-button">🚪 Logout</div>
            </a>
        </div>
        <script>
        function logoutUser() {{
            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'logout'}}, '*');
        }}
        </script>
        """, unsafe_allow_html=True)
    else:
        # User is not logged in - show login and signup buttons
        st.markdown(f"""
        <div class="auth-buttons-container">
            <a href="javascript:void(0)" onclick="showLoginPopup()" style='text-decoration: none;'>
                <div class="auth-button login-button">🔑 Login</div>
            </a>
            <a href="javascript:void(0)" onclick="showSignupPopup()" style='text-decoration: none;'>
                <div class="auth-button signup-button">✏️ Sign Up</div>
            </a>
        </div>
        <script>
        function showLoginPopup() {{
            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'show_login_popup'}}, '*');
        }}
        function showSignupPopup() {{
            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'show_signup_popup'}}, '*');
        }}
        </script>
        """, unsafe_allow_html=True)
