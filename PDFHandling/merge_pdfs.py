import os
import glob

try:
    from pypdf import PdfWriter
    USE_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfMerger
        USE_PYPDF = False
    except ImportError:
        print("Error: Required library not found.")
        print("Please install pypdf by running: pip install pypdf")
        exit(1)

def merge_pdfs(target_dir, output_filename="merged_documents.pdf"):
    """
    Combines all PDF files in the given directory into a single PDF.
    """
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    # Find all PDF files in the directory and sort them alphabetically
    search_pattern = os.path.join(target_dir, "*.pdf")
    pdf_files = glob.glob(search_pattern)
    pdf_files.sort()
    
    # Exclude the output file itself if it already exists in the directory
    output_path = os.path.join(target_dir, output_filename)
    if output_path in pdf_files:
        pdf_files.remove(output_path)

    if not pdf_files:
        print(f"No PDF files found in {target_dir}")
        return
        
    print(f"Found {len(pdf_files)} PDF files to merge:")
    
    if USE_PYPDF:
        from pypdf import PdfReader
        merger = PdfWriter()
        for pdf in pdf_files:
            print(f" - Adding {os.path.basename(pdf)}")
            try:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    cw = float(page.mediabox.width)
                    ch = float(page.mediabox.height)
                    if cw > 0 and ch > 0:
                        # Distinguish portrait vs landscape roughly
                        if cw > ch:
                            target_w, target_h = 842.0, 595.0
                        else:
                            target_w, target_h = 595.0, 842.0
                            
                        # Scale factor that fits standard dimensions
                        scale_factor = min(target_w / cw, target_h / ch)
                        
                        # Scale if the page is too small or overly large
                        if scale_factor < 0.9 or scale_factor > 1.1:
                            page.scale_by(scale_factor)
                            
                    merger.add_page(page)
            except Exception as e:
                print(f"Error reading {pdf}: {e}")
                
        merger.write(output_path)
    else:
        merger = PdfMerger()
        for pdf in pdf_files:
            print(f" - Adding {os.path.basename(pdf)}")
            merger.append(pdf)
        merger.write(output_path)
        
    merger.close()
    print(f"\nSuccessfully created merged PDF at:\n{output_path}")

if __name__ == "__main__":
    DIRECTORY = r"C:\Users\somsh\Downloads\canadadocs"
    OUTPUT_FILE = "combined_canadadocs.pdf"
    
    merge_pdfs(DIRECTORY, OUTPUT_FILE)
