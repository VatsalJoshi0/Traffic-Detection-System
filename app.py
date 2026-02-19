import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Traffic Sentinel AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules after page config
from modules import login, file_upload, dashboard, support

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

def main():
    if not st.session_state['authenticated']:
        login.render_login()
    else:
        # Pinned Sidebar Layout
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100, caption="Admin Profile")
            st.title(f"Hi, {st.session_state['username']}")
            st.markdown("---")
            
            menu = ["Realtime Dashboard", "File Upload", "Support"]
            choice = st.radio("Navigate", menu)
            
            st.markdown("---")
            if st.button("Logout"):
                st.session_state['authenticated'] = False
                st.session_state['username'] = None
                st.rerun()

        # Main Content Routing
        if choice == "Realtime Dashboard":
            dashboard.render_dashboard()
        elif choice == "File Upload":
            file_upload.render_upload()
        elif choice == "Support":
            support.render_support()

if __name__ == "__main__":
    main()
