import fitz
import pdfplumber
import base64
import io

def extract_pdf_contents(pdf_bytes: bytes):
    """
    Extract text, tables, images from a PDF (bytes input).
    Returns a structured dictionary.
    """
    result = {"pages": []}

    # open PDF from bytes in memory
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    plumber = pdfplumber.open(io.BytesIO(pdf_bytes))

    for page_index, page in enumerate(doc):
        page_data = {
            "page_number": page_index + 1,
            "text": page.get_text("text"),   # raw text
            "tables": [],
            "images": []
        }

        # extract tables using pdfplumber
        plumber_page = plumber.pages[page_index]
        tables = plumber_page.extract_tables()
        for table in tables:
            page_data["tables"].append(table)

        # extract images
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)

            # convert image to base64 for safe storage/transfer
            img_b64 = base64.b64encode(base_image["image"]).decode()

            page_data["images"].append({
                "extension": base_image["ext"],
                "data": img_b64
            })

        result["pages"].append(page_data)

   
    doc.close()
    plumber.close()
    return result
