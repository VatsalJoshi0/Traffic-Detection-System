import streamlit as st
from services import check_login

def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center;'>🚦 Traffic Sentinel Login</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Secure Access for Traffic Enforcement</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="admin")
            
            submitted = st.form_submit_button("Login System")
            
            if submitted:
                if check_login(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
