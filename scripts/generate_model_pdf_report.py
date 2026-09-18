from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "rapport_modele_ocr_qwen_rapatriement.pdf"


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F3A5F"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#41546B"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#1F3A5F"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#2F5D8C"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#202A36"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#152238"),
            backColor=colors.HexColor("#F4F7FA"),
            borderColor=colors.HexColor("#D8E1EA"),
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#5D6B7A"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            leftIndent=10,
            firstLineIndent=-6,
            textColor=colors.HexColor("#202A36"),
            spaceAfter=3,
        )
    )
    return styles


def p(text, style):
    return Paragraph(text, style)


def bullet(text, styles):
    return p("- " + text, styles["BulletText"])


def code(text, styles):
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return p(escaped.replace("\n", "<br/>"), styles["CodeBlock"])


def section(story, title, styles):
    story.append(p(title, styles["SectionTitle"]))


def make_table(rows, widths=None):
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C7D3DF")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(2 * cm, 1.15 * cm, "Rapport PFE - Module OCR Qwen Rapatriement")
    canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_report():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Rapport modele OCR Qwen Rapatriement",
        author="Projet PFE",
    )

    story = []
    story.append(Spacer(1, 4 * cm))
    story.append(p("Rapport technique", styles["CoverSubtitle"]))
    story.append(p("Module OCR intelligent base sur Qwen", styles["CoverTitle"]))
    story.append(p("Extraction automatique des donnees de documents de rapatriement", styles["CoverSubtitle"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(
        make_table(
            [
                ["Element", "Description"],
                ["Backend API", "FastAPI - endpoint POST /api/ocr/parse"],
                ["Modele IA", "Qwen2.5-VL-7B via LM Studio"],
                ["Interface de test", "Gradio"],
                ["Sortie", "JSON normalise pour pre-remplissage React"],
                ["Evaluation actuelle", "100% sur 1 document test valide"],
            ],
            [5 * cm, 9 * cm],
        )
    )
    story.append(PageBreak())

    section(story, "1. Presentation du module", styles)
    story.append(
        p(
            "Le module OCR intelligent automatise l'extraction de donnees depuis un document financier de rapatriement. "
            "L'objectif est de remplacer la saisie manuelle par une extraction fiable, normalisee et directement exploitable "
            "par une application web.",
            styles["BodyJustify"],
        )
    )
    story.append(code('{\n  "dateDosRap": "2026-06-08",\n  "mntRap": 250.0,\n  "typePieceBenef": 1,\n  "noPieceBenef": "0004545Y"\n}', styles))

    section(story, "2. Objectifs fonctionnels", styles)
    for item in [
        "Charger un document PDF de rapatriement.",
        "Extraire la date de l'operation.",
        "Extraire le montant du rapatriement.",
        "Identifier le type de piece du beneficiaire.",
        "Extraire le numero de piece du beneficiaire.",
        "Retourner un JSON standardise pour une application React.",
        "Pre-remplir automatiquement le formulaire metier.",
    ]:
        story.append(bullet(item, styles))

    section(story, "3. Technologies utilisees", styles)
    story.append(
        make_table(
            [
                ["Technologie", "Role dans le projet"],
                ["Python 3.11", "Langage principal du backend OCR."],
                ["FastAPI", "Creation de l'API REST consommable par React."],
                ["Uvicorn", "Serveur ASGI pour executer l'API FastAPI."],
                ["PyMuPDF", "Extraction du texte depuis les fichiers PDF."],
                ["Qwen2.5-VL-7B", "Modele LLM local utilise pour aider l'interpretation des documents."],
                ["LM Studio", "Execution locale du modele et exposition d'une API compatible OpenAI."],
                ["Gradio", "Interface de test locale, non utilisee comme API d'integration."],
                ["React", "Application cliente qui upload le PDF et pre-remplit le formulaire."],
            ],
            [4.2 * cm, 10.2 * cm],
        )
    )

    section(story, "4. Architecture generale", styles)
    story.append(
        code(
            "Utilisateur\n"
            "  -> Application React\n"
            "  -> POST /api/ocr/parse (multipart/form-data)\n"
            "  -> Backend FastAPI\n"
            "  -> PyMuPDF + regles metier\n"
            "  -> LM Studio / Qwen local si necessaire\n"
            "  -> JSON normalise\n"
            "  -> Formulaire pre-rempli",
            styles,
        )
    )
    story.append(
        p(
            "L'API officielle d'integration est developpee avec FastAPI. Gradio reste une interface de test locale "
            "permettant de valider rapidement le comportement du module.",
            styles["BodyJustify"],
        )
    )

    section(story, "5. Role du modele IA", styles)
    story.append(
        p(
            "Le modele Qwen execute localement assiste l'analyse des documents lorsque le layout varie ou lorsque le texte "
            "extrait demande une interpretation. L'approche combine des regles deterministes et l'apport d'un LLM local. "
            "Cette combinaison limite les erreurs et conserve les donnees sensibles sur la machine locale.",
            styles["BodyJustify"],
        )
    )
    story.append(code("LM_STUDIO_HOST=http://127.0.0.1:1234\nQWEN_MODEL=qwen2.5-vl-7b\nLM_STUDIO_TIMEOUT=900", styles))

    section(story, "6. Inputs et outputs", styles)
    story.append(
        make_table(
            [
                ["Element", "Detail"],
                ["Input", "Fichier PDF, PNG, JPG ou JPEG envoye avec le champ form-data file."],
                ["Endpoint", "POST http://localhost:8000/api/ocr/parse"],
                ["Output", "JSON contenant dateDosRap, mntRap, typePieceBenef et noPieceBenef."],
                ["Format date", "YYYY-MM-DD"],
                ["Type piece", "C -> 1, S -> 4"],
            ],
            [4.2 * cm, 10.2 * cm],
        )
    )

    section(story, "7. Regles metier d'extraction", styles)
    for item in [
        "dateDosRap correspond a la date de l'operation trouvee dans le document.",
        "mntRap correspond au montant associe a DROITS A TRANSFERT CUMULES (6), ou au plus petit montant pertinent.",
        "typePieceBenef vaut 1 si le document contient le code C.",
        "typePieceBenef vaut 4 si le document contient le code S.",
        "noPieceBenef correspond au code beneficiaire compose de 7 chiffres suivis d'une lettre.",
    ]:
        story.append(bullet(item, styles))

    section(story, "8. API REST exposee", styles)
    story.append(code("GET  http://localhost:8000/api/health\nGET  http://localhost:8000/api/ocr/parse/schema\nPOST http://localhost:8000/api/ocr/parse", styles))
    story.append(p("Exemple de requete Postman:", styles["SubTitle"]))
    story.append(code("Method: POST\nURL: http://localhost:8000/api/ocr/parse\nBody: form-data\nKey: file\nType: File\nValue: Rapatriment_AVA.pdf", styles))

    section(story, "9. Integration React", styles)
    story.append(
        code(
            "const formData = new FormData();\n"
            "formData.append(\"file\", file);\n\n"
            "const response = await fetch(\"http://localhost:8000/api/ocr/parse\", {\n"
            "  method: \"POST\",\n"
            "  body: formData,\n"
            "});\n\n"
            "const data = await response.json();",
            styles,
        )
    )
    story.append(
        p(
            "Le JSON retourne est ensuite mappe vers les champs du formulaire: typePieceBenef, noPieceBenef, mntRap et dateDosRap.",
            styles["BodyJustify"],
        )
    )

    section(story, "10. Evaluation et accuracy", styles)
    story.append(
        p(
            "L'accuracy est calculee en comparant le JSON extrait avec des valeurs attendues validees manuellement. "
            "La metrique principale est l'accuracy par champ.",
            styles["BodyJustify"],
        )
    )
    story.append(code("Accuracy = nombre de champs correctement extraits / nombre total de champs attendus", styles))
    story.append(
        make_table(
            [
                ["Document", "Champs corrects", "Total champs", "Accuracy"],
                ["Rapatriment_AVA.pdf", "4", "4", "100%"],
            ],
            [5 * cm, 3.2 * cm, 3.2 * cm, 3 * cm],
        )
    )
    story.append(
        p(
            "Cette valeur represente l'evaluation actuelle sur un document test. Pour une evaluation scientifique plus solide, "
            "il faut constituer un dataset contenant plusieurs documents annotes.",
            styles["BodyJustify"],
        )
    )

    section(story, "11. Securite et confidentialite", styles)
    for item in [
        "Les fichiers temporaires sont supprimes apres traitement.",
        "La taille maximale des fichiers est limitee par configuration.",
        "Les cles API restent cote backend et ne sont jamais exposees dans React.",
        "LM Studio ne doit pas etre expose directement a Internet.",
        "Une authentification peut etre ajoutee si l'API est exposee hors reseau local.",
    ]:
        story.append(bullet(item, styles))

    section(story, "12. Limites et perspectives", styles)
    story.append(p("Limites identifiees:", styles["SubTitle"]))
    for item in [
        "PDF scanne ou image de mauvaise qualite.",
        "Documents fortement inclines, flous ou incomplets.",
        "Layouts tres differents du format attendu.",
        "Modele Qwen non charge dans LM Studio.",
    ]:
        story.append(bullet(item, styles))
    story.append(p("Ameliorations possibles:", styles["SubTitle"]))
    for item in [
        "Ajouter un score de confiance par champ.",
        "Tester le modele sur un dataset plus large.",
        "Ajouter une validation humaine avant soumission.",
        "Gerer plusieurs operations dans un seul document.",
        "Ajouter une authentification JWT pour securiser l'API.",
    ]:
        story.append(bullet(item, styles))

    section(story, "13. Conclusion", styles)
    story.append(
        p(
            "Le module OCR intelligent base sur Qwen permet d'automatiser l'extraction d'informations depuis des documents "
            "financiers de rapatriement. Il combine extraction PDF, regles metier et assistance LLM locale. L'API FastAPI "
            "POST /api/ocr/parse offre une integration simple avec React pour pre-remplir automatiquement le formulaire metier.",
            styles["BodyJustify"],
        )
    )

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    build_report()
    print(OUTPUT_PATH)
