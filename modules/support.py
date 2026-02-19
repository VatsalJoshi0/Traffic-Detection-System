import streamlit as st

def render_support():
    st.header("🛠️ Support & Help Center")
    
    tab1, tab2 = st.tabs(["Contact Support", "FAQ"])
    
    with tab1:
        st.subheader("Report an Issue")
        with st.form("support_form"):
            issue_type = st.selectbox("Issue Type", ["Camera Offline", "Login Issue", "False Detection", "Other"])
            description = st.text_area("Description", placeholder="Describe the problem in detail...")
            
            submitted = st.form_submit_button("Submit Ticket")
            if submitted:
                st.success("Ticket #4029 created successfully. IT Support will contact you shortly.")
                
    with tab2:
        st.subheader("Frequently Asked Questions")
        with st.expander("How do I add a new camera?"):
            st.write("Contact the admin to register a new RTSP stream in the backend configuration.")
            
        with st.expander("Is the data saved automatically?"):
            st.write("Yes, all detected violations are automatically logged to the SQLite database.")
