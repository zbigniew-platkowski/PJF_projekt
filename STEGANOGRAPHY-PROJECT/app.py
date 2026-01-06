# Main file of the program.

import streamlit as st
from gui import render_gui

def main() -> None:
    """
    Main function. Configures Streamlit and renders the GUI.

    :return: None
    """

    # Configure Streamlit application layout
    st.set_page_config(
        page_title="Steganography",
        layout="wide",
    )

    # Render the graphical user interface
    render_gui()

main()