import json
import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from functions import displayPDF
from rapatriement_parser import extract_text_from_pdf, parse_rapatriement, to_pretty_json


if get_script_run_ctx() is None:
    print("ERROR: this application must be run with Streamlit.")
    print("Use: streamlit run app.py")
    sys.exit(0)


st.set_page_config(page_title="Rapatriement PDF to JSON", layout="wide")

st.title("Rapatriement PDF to JSON")

pdf_file = st.file_uploader("PDF", type=["pdf"])

if not pdf_file:
    st.stop()

pdf_bytes = pdf_file.read()

with st.expander("Document", expanded=True):
    displayPDF(pdf_bytes)

try:
    text, extraction_mode = extract_text_from_pdf(pdf_bytes)
except Exception as error:
    st.error(str(error))
    st.stop()

parsed = parse_rapatriement(text)

st.caption(f"Extraction mode: {extraction_mode}")

left, right = st.columns([1, 1])

with left:
    st.subheader("Fields")
    date_dos = st.text_input("dateDosRap", value=str(parsed["dateDosRap"]))
    mnt_rap = st.number_input("mntRap", value=float(parsed["mntRap"]), format="%.3f")
    type_piece = st.number_input("typePieceBenef", value=int(parsed["typePieceBenef"]), step=1)
    no_piece = st.text_input("noPieceBenef", value=str(parsed["noPieceBenef"]))

final_data = {
    "dateDosRap": date_dos,
    "mntRap": float(mnt_rap),
    "typePieceBenef": int(type_piece),
    "noPieceBenef": no_piece,
}

json_text = to_pretty_json(final_data)

with right:
    st.subheader("JSON")
    edited_json = st.text_area("Output", value=json_text, height=360)
    try:
        download_json = json.dumps(json.loads(edited_json), ensure_ascii=False, indent=2)
        st.download_button(
            "Download JSON",
            data=download_json,
            file_name="Rapatriment.json",
            mime="application/json",
        )
    except json.JSONDecodeError as error:
        st.error(f"Invalid JSON: {error}")

with st.expander("Extracted text"):
    st.text(text)
