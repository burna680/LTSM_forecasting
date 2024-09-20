import streamlit as st
from src.pages.Page import Page
from src.constants import *

class ContactPage(Page):
    def render(self):
        st.subheader("You can find me on:")
        margin_r,body,margin_l = st.columns([0.4, 3, 0.4])

        with body:
            with st.container():
                col1, col2 = st.columns([0.1, 3])
                with col1:
                    st.write(linkedin_logo, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"#####  [Linkedin]({linkedin_link})")
            with st.container():
                col1, col2 = st.columns([0.1, 3])
                with col1:
                    st.write(github_logo, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"#####  [Github]({github_link})")
            with st.container():
                col1, col2 = st.columns([0.1, 3])
                with col1:
                    st.write(personal_logo, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"#####  [My Website]({personal_website})")