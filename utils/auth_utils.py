"""
Authentication utilities for the KnowledgeWeaver application.
"""
import os
import json
import uuid
import hashlib
import datetime
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

# User roles
ROLE_USER = "user"
ROLE_ADMIN = "admin"

def hash_password(password: str) -> str:
    """
    Hash a password for storing

    Args:
        password: The password to hash

    Returns:
        Hashed password
    """
    # In a production environment, use a proper password hashing library like bcrypt
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password: str, provided_password: str) -> bool:
    """
    Verify a stored password against a provided password

    Args:
        stored_password: The stored hashed password
        provided_password: The password to check

    Returns:
        True if the password matches, False otherwise
    """
    return stored_password == hash_password(provided_password)

def create_user_in_neo4j(neo4j_url: str, neo4j_user: str, neo4j_password: str,
                        username: str, email: str, password: str, role: str = ROLE_USER) -> Dict:
    """
    Create a new user in Neo4j

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        username: New user's username
        email: New user's email
        password: New user's password
        role: User's role (default: user)

    Returns:
        Dictionary with status and user ID if successful
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Check if user already exists
        with driver.session() as session:
            result = session.run(
                "MATCH (u:User) WHERE u.username = $username OR u.email = $email RETURN u",
                username=username, email=email
            )

            if result.single():
                return {"status": "error", "message": "Username or email already exists"}

            # Create user
            user_id = str(uuid.uuid4())
            hashed_password = hash_password(password)
            created_at = datetime.datetime.now().isoformat()

            session.run(
                """
                CREATE (u:User {
                    id: $id,
                    username: $username,
                    email: $email,
                    password: $password,
                    role: $role,
                    created_at: $created_at,
                    active: true
                })
                """,
                id=user_id,
                username=username,
                email=email,
                password=hashed_password,
                role=role,
                created_at=created_at
            )

            return {"status": "success", "user_id": user_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'driver' in locals():
            driver.close()

def authenticate_user(neo4j_url: str, neo4j_user: str, neo4j_password: str,
                     username: str, password: str) -> Dict:
    """
    Authenticate a user

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        username: User's username
        password: User's password

    Returns:
        Dictionary with user information if successful, error otherwise
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Find user
        with driver.session() as session:
            result = session.run(
                "MATCH (u:User) WHERE u.username = $username RETURN u",
                username=username
            )

            user_record = result.single()
            if not user_record:
                return {"status": "error", "message": "Invalid username or password"}

            user = user_record["u"]

            # Verify password
            if not verify_password(user["password"], password):
                return {"status": "error", "message": "Invalid username or password"}

            # Check if user is active
            if not user.get("active", True):
                return {"status": "error", "message": "Account is inactive"}

            # Return user information
            return {
                "status": "success",
                "user_id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "created_at": user["created_at"]
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'driver' in locals():
            driver.close()

def get_user_by_id(neo4j_url: str, neo4j_user: str, neo4j_password: str, user_id: str) -> Dict:
    """
    Get user information by ID

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        user_id: User's ID

    Returns:
        Dictionary with user information if successful, error otherwise
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Find user
        with driver.session() as session:
            result = session.run(
                "MATCH (u:User) WHERE u.id = $user_id RETURN u",
                user_id=user_id
            )

            user_record = result.single()
            if not user_record:
                return {"status": "error", "message": "User not found"}

            user = user_record["u"]

            # Return user information (excluding password)
            return {
                "status": "success",
                "user_id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "created_at": user["created_at"],
                "active": user.get("active", True)
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'driver' in locals():
            driver.close()

def get_all_users(neo4j_url: str, neo4j_user: str, neo4j_password: str) -> Dict:
    """
    Get all users (admin function)

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Dictionary with list of users if successful, error otherwise
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Get all users
        with driver.session() as session:
            result = session.run(
                "MATCH (u:User) RETURN u ORDER BY u.created_at DESC"
            )

            users = []
            for record in result:
                user = record["u"]
                # Exclude password from user data
                users.append({
                    "user_id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"],
                    "created_at": user["created_at"],
                    "active": user.get("active", True)
                })

            return {"status": "success", "users": users}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'driver' in locals():
            driver.close()

def update_user(neo4j_url: str, neo4j_user: str, neo4j_password: str,
               user_id: str, update_data: Dict) -> Dict:
    """
    Update user information

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        user_id: User's ID
        update_data: Dictionary with fields to update

    Returns:
        Dictionary with status
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Update user
        with driver.session() as session:
            # Build update query dynamically based on provided fields
            update_statements = []
            params = {"user_id": user_id}

            for key, value in update_data.items():
                # Don't allow updating user_id
                if key == "user_id":
                    continue

                # Hash password if it's being updated
                if key == "password":
                    value = hash_password(value)

                update_statements.append(f"u.{key} = ${key}")
                params[key] = value

            if not update_statements:
                return {"status": "error", "message": "No fields to update"}

            update_query = f"""
            MATCH (u:User) WHERE u.id = $user_id
            SET {', '.join(update_statements)}
            RETURN u
            """

            result = session.run(update_query, **params)

            if not result.single():
                return {"status": "error", "message": "User not found"}

            return {"status": "success", "message": "User updated successfully"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'driver' in locals():
            driver.close()

def delete_user(neo4j_url: str, neo4j_user: str, neo4j_password: str, user_id: str) -> Dict:
    """
    Delete a user

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        user_id: User's ID

    Returns:
        Dictionary with status
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Delete user
        with driver.session() as session:
            result = session.run(
                "MATCH (u:User) WHERE u.id = $user_id DETACH DELETE u RETURN count(*) as deleted",
                user_id=user_id
            )

            deleted = result.single()["deleted"]
            if deleted == 0:
                return {"status": "error", "message": "User not found"}

            return {"status": "success", "message": "User deleted successfully"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'driver' in locals():
            driver.close()

def is_admin(user_data: Dict) -> bool:
    """
    Check if a user is an admin

    Args:
        user_data: User data dictionary

    Returns:
        True if user is an admin, False otherwise
    """
    return user_data.get("role") == ROLE_ADMIN

def initialize_auth_state():
    """
    Initialize authentication state in session
    """
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

def login_user(user_data: Dict):
    """
    Log in a user by setting session state

    Args:
        user_data: User data dictionary
    """
    st.session_state.user = user_data
    st.session_state.authenticated = True
    st.session_state.user_id = user_data["user_id"]

def logout_user():
    """
    Log out a user by clearing session state
    """
    st.session_state.user = None
    st.session_state.authenticated = False
    if 'user_id' in st.session_state:
        del st.session_state.user_id

def get_current_user() -> Optional[Dict]:
    """
    Get the currently logged in user

    Returns:
        User data dictionary or None if not logged in
    """
    return st.session_state.user if st.session_state.authenticated else None

def require_login():
    """
    Require login to access a page

    If user is not logged in, redirect to home page
    """
    if not st.session_state.authenticated:
        st.warning("Please log in to access this page")
        st.stop()

def require_admin():
    """
    Require admin role to access a page

    If user is not an admin, redirect to home page
    """
    if not st.session_state.authenticated or not is_admin(st.session_state.user):
        st.warning("You don't have permission to access this page")
        st.stop()

def create_initial_admin(neo4j_url: str, neo4j_user: str, neo4j_password: str) -> Dict:
    """
    Create initial admin user if no users exist

    Args:
        neo4j_url: Neo4j database URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Dictionary with status
    """
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

        # Check if any users exist
        with driver.session() as session:
            result = session.run("MATCH (u:User) RETURN count(u) as count")
            count = result.single()["count"]

            if count > 0:
                return {"status": "info", "message": "Users already exist"}

            # Create admin user with specified credentials
            admin_username = "admin"
            admin_email = "admin@knowledgeweaver.com"
            admin_password = "admin@123"  # Specified admin password

            return create_user_in_neo4j(
                neo4j_url, neo4j_user, neo4j_password,
                admin_username, admin_email, admin_password, ROLE_ADMIN
            )

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if 'driver' in locals():
            driver.close()
