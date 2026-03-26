import sys
import os
import shutil
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import pdf2image
from PIL import Image
import pytesseract
from pytesseract import Output, TesseractError
from functions import convert_pdf_to_txt_pages, convert_pdf_to_txt_file, save_pages, displayPDF, images_to_txt

# Auto-detect tesseract path for the running process.
# If user has it in PATH this picks it up; otherwise tries default Windows install path.
_tesseract_path = shutil.which("tesseract")
if not _tesseract_path:
    _tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not os.path.exists(_tesseract_path):
        _tesseract_path = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

if _tesseract_path and os.path.exists(_tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = _tesseract_path

# Ensure app is launched via Streamlit runner (not `python app.py`) to avoid missing ScriptRunContext warnings.
if get_script_run_ctx() is None:
    print("ERROR: this application must be run with Streamlit.")
    print("Use: streamlit run app.py")
    sys.exit(0)

st.set_page_config(page_title="PDF to Text")


html_temp = """
            <div style="background-color:{};padding:1px">
            
            </div>
            """

# st.markdown("""
#     ## :outbox_tray: Text data extractor: PDF to Text
    
# """)
# st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
st.markdown("""
    ## Text data extractor: PDF to Text
    
""")
languages = {
    'English': 'eng',
    'French': 'fra',
    'Arabic': 'ara',
    'Spanish': 'spa',
}

with st.sidebar:
    st.title(":outbox_tray: PDF to Text")
    textOutput = st.selectbox(
        "How do you want your output text?",
        ('One text file (.txt)', 'Text file per page (ZIP)'))
    ocr_box = st.checkbox('Enable OCR (scanned document)')
    
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work?
    Simply load your PDF and convert it to single-page or multi-page text.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
    st.markdown("""
    Made by [@nainia_ayoub](https://twitter.com/nainia_ayoub) 
    """)
    st.markdown(
        """
        <a href="https://www.buymeacoffee.com/nainiayoub" target="_blank">
        <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174">
        </a>
        """,
        unsafe_allow_html=True,
    )
    

pdf_file = st.file_uploader("Load your PDF", type=['pdf', 'png', 'jpg'])
hide="""
<style>
footer{
	visibility: hidden;
    	position: relative;
}
.viewerBadge_container__1QSob{
  	visibility: hidden;
}
#MainMenu{
	visibility: hidden;
}
<style>
"""
st.markdown(hide, unsafe_allow_html=True)
if pdf_file:
    path = pdf_file.read()
    file_extension = pdf_file.name.split(".")[-1]
    
    if file_extension == "pdf":
        # display document
        with st.expander("Display document"):
            displayPDF(path)
        if ocr_box:
            option = st.selectbox('Select the document language', list(languages.keys()))
        # pdf to text
        if textOutput == 'One text file (.txt)':
            if ocr_box:
                texts, nbPages = images_to_txt(path, languages[option])
                totalPages = "Pages: "+str(nbPages)+" in total"
                text_data_f = "\n\n".join(texts)
            else:
                text_data_f, nbPages = convert_pdf_to_txt_file(pdf_file)
                totalPages = "Pages: "+str(nbPages)+" in total"

            st.info(totalPages)
            st.download_button("Download txt file", text_data_f)
        else:
            if ocr_box:
                text_data, nbPages = images_to_txt(path, languages[option])
                totalPages = "Pages: "+str(nbPages)+" in total"
            else:
                text_data, nbPages = convert_pdf_to_txt_pages(pdf_file)
                totalPages = "Pages: "+str(nbPages)+" in total"
            st.info(totalPages)
            zipPath = save_pages(text_data)
            # download text data   
            with open(zipPath, "rb") as fp:
                btn = st.download_button(
                    label="Download ZIP (txt)",
                    data=fp,
                    file_name="pdf_to_txt.zip",
                    mime="application/zip"
                )
    else:
        option = st.selectbox("What's the language of the text in the image?", list(languages.keys()))
        pil_image = Image.open(pdf_file)
        try:
            # Ensure tesseract path is explicit for this process
            if not pytesseract.pytesseract.tesseract_cmd:
                # try a fallback path as Well.
                if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
                    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                elif os.path.exists(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

            text = pytesseract.image_to_string(pil_image, lang=languages[option])
        except (pytesseract.TesseractNotFoundError, FileNotFoundError) as error:
            st.error(f"Tesseract not found: {error}. Install Tesseract OCR, add it to PATH, then reload the app.")
            st.stop()
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Display Image"):
                st.image(pdf_file)
        with col2:
            with st.expander("Display Text"):
                st.info(text)
        st.download_button("Download txt file", text)

    st.markdown('''
    <a target="_blank" style="color: black" href="https://twitter.com/intent/tweet?text=You%20can%20extract%20text%20from%20your%20PDF%20using%20this%20PDF%20to%20Text%20streamlit%20app%20by%20@nainia_ayoub!%0A%0Ahttps://nainiayoub-pdf-text-data-extractor-app-p6hy0z.streamlit.app/">
        <button class="btn">
            Spread the word!
        </button>
    </a>
    <style>
    .btn{
        display: inline-flex;
        -moz-box-align: center;
        align-items: center;
        -moz-box-pack: center;
        justify-content: center;
        font-weight: 400;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        margin: 0px;
        line-height: 1.6;
        color: rgb(49, 51, 63);
        background-color: #fff;
        width: auto;
        user-select: none;
        border: 1px solid rgba(49, 51, 63, 0.2);
        }
    .btn:hover{
        color: #00acee;
        background-color: #fff;
        border: 1px solid #00acee;
    }
    </style>
    ''',
    unsafe_allow_html=True
    )
    
