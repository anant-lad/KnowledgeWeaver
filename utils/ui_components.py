"""
UI components for the KnowledgeWeaver application.
"""
import streamlit as st

def show_status(message, status_type="info"):
    """Display a styled status message
    
    Args:
        message: The message to display
        status_type: One of 'success', 'info', 'warning', or 'error'
    """
    if status_type == "success":
        st.markdown(f"<div class='status-box success-box'>✅ {message}</div>", unsafe_allow_html=True)
    elif status_type == "warning":
        st.markdown(f"<div class='status-box warning-box'>⚠️ {message}</div>", unsafe_allow_html=True)
    elif status_type == "error":
        st.markdown(f"<div class='status-box error-box'>❌ {message}</div>", unsafe_allow_html=True)
    else:  # info
        st.markdown(f"<div class='status-box info-box'>ℹ️ {message}</div>", unsafe_allow_html=True)

def load_css():
    """Load custom CSS for the application"""
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton button {
            width: 100%;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .success-box {
            background-color: #d1e7dd;
            color: #0f5132;
        }
        .warning-box {
            background-color: #fff3cd;
            color: #856404;
        }
        .error-box {
            background-color: #f8d7da;
            color: #842029;
        }
        .info-box {
            background-color: #cff4fc;
            color: #055160;
        }
        .image-container {
            display: flex;
            justify-content: center;
            margin: 1rem 0;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
        }
        .analysis-result {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
        .auth-form {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .user-info {
            background-color: #e9ecef;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .admin-panel {
            background-color: #f8d7da;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            color: #842029;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def create_navigation_links():
    """Create navigation links for the application"""
    st.markdown("""
    <div style='display: flex; gap: 1rem; margin-bottom: 1rem;'>
        <a href='/' style='text-decoration: none;'>
            <div style='background-color: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;'>
                📁 Document Management
            </div>
        </a>
        <a href='/image_analysis' style='text-decoration: none;'>
            <div style='background-color: #2196F3; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;'>
                📷 Image Analysis
            </div>
        </a>
        <a href='/document_comparison' style='text-decoration: none;'>
            <div style='background-color: #9C27B0; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-align: center;'>
                📊 Document Comparison
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

def display_user_info(user):
    """Display user information
    
    Args:
        user: User data dictionary
    """
    st.markdown(f"""
    <div class='user-info'>
        <h3>👤 {user['username']}</h3>
        <p><strong>Email:</strong> {user['email']}</p>
        <p><strong>Role:</strong> {user['role']}</p>
    </div>
    """, unsafe_allow_html=True)

def create_admin_link():
    """Create a link to the admin panel"""
    st.markdown("""
    <a href='/admin_panel' style='text-decoration: none;'>
        <div class='admin-panel'>
            🔧 Admin Panel
        </div>
    </a>
    """, unsafe_allow_html=True)
