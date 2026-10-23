import streamlit as st

st.set_page_config(page_title="AwareML Meta-Recommender V3", layout="wide")
from awareml.ui_v2.meta_recommender_v3 import render_meta_recommender_v3

render_meta_recommender_v3()
