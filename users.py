import streamlit as st
import bcrypt

MAX_LOGIN_ATTEMPTS = 5

def check_password(username, password):
    """
    Check the password by dynamically fetching the stored hash from secrets.
    """
    env_key = f"{username.upper()}_HASH"  # Fetch the corresponding user's hash key from secrets.
    stored_hash = st.secrets.get(env_key)
    
    if not stored_hash:
        return False  # Return False if no hash is found in secrets.
    return bcrypt.checkpw(password.encode("utf-8"), stored_hash["value"].encode("utf-8"))

def login():
    if "login_attempts" not in st.session_state:
        st.session_state["login_attempts"] = 0

    if st.session_state["login_attempts"] >= MAX_LOGIN_ATTEMPTS:
        st.error("🚫 Too many failed login attempts. Try again later.")
        st.stop()

    st.markdown(
        """
        <style>
        .login-title {
            text-align: center;
            font-size: 28px;
            color: #ffffff;
            margin-bottom: 1.5rem;
        }
        .stTextInput > div > div > input {
            background-color: #2e2e2e;
            color: #fff;
            border: 1px solid #444;
            border-radius: 8px;
        }
        .stTextInput label {
            color: #ccc;
        }
        .stButton > button {
            background-color: #4a90e2;
            color: white;
            border-radius: 10px;
            font-weight: 600;
            width: 100%;
            height: 3rem;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div class="login-title">🔐 Login User</div>', unsafe_allow_html=True
        )
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_password(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["login_attempts"] = 0
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.session_state["login_attempts"] += 1
                remaining = MAX_LOGIN_ATTEMPTS - st.session_state["login_attempts"]
                st.error(f"❌ Invalid credentials. {remaining} attempts left.")

def logout():
    username = st.session_state.get("username", "User")
    st.sidebar.markdown("---")
    st.sidebar.write(f"🔓 Logged in as: `{username}`")

    if st.sidebar.button("Logout"):
        for key in [
            "authenticated",
            "username",
            "final_df",
            "original_df",
            "removed_questions",
            "removed_patterns",
        ]:
            st.session_state.pop(key, None)
        st.success("👋 Logged out successfully.")
        st.rerun()
