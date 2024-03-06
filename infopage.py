import streamlit as st
import streamlit.components.v1 as components
def run():
    iframe_src = "https://cosspp.maps.arcgis.com/apps/webappviewer/index.html?id=7fa44a0704c04506b169f7abf077f810"
    components.iframe(iframe_src,height=800, width = 100%)

st.set_page_config(layout="wide")
st.title("iPlantGreenS\u00b2 : Integrated Planning toolkit for Green Infrastructure Siting and Selection")

if __name__ == "__main__":
    run()
