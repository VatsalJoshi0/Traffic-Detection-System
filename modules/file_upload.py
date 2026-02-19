import streamlit as st
from services import upload_file_to_backend

def render_upload():
    st.header("📂 Upload Video for Analysis")
    st.markdown("Upload recorded traffic footage for offline processing.")
    
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(f"Filename: {uploaded_file.name}")
            st.info(f"Size: {uploaded_file.size / 1024 / 1024:.2f} MB")
            
        with col2:
            if st.button("Start Analysis", type="primary"):
                with st.spinner("Uploading and initiating AI models..."):
                    result = upload_file_to_backend(uploaded_file)
                    st.success(f"Success! {result['message']}")
                    st.balloons()
