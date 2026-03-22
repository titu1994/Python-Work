import argparse
import os
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is not installed.", file=sys.stderr)
    print("Please install it using: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

def compress_pdf(input_path, output_path, quality=50, dpi=150):
    """
    Compresses a PDF file using PyMuPDF (fitz).
    It works by flattening the entire document—rasterizing every page into
    a single high-quality optimized JPEG. This guarantees the file format is 
    valid and is explicitly meant for Scanned Documents / Passports.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        return False

    print(f"Compressing (Flattening): {input_path}")
    
    try:
        # Open the original PDF document
        doc = fitz.open(input_path)
        out_doc = fitz.open()

        print(f"Flattening PDF to images (JPEG Quality: {quality}, DPI: {dpi})...")
        for page in doc:
            # Render page to an image at the specified DPI which is good for documents
            pix = page.get_pixmap(dpi=dpi)
            
            # Extract standard optimized JPEG bytes using PyMuPDF directly
            img_bytes = pix.tobytes("jpeg", jpg_quality=quality)
            
            # Create a brand new page with exact same dimensions
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            
            # Insert the newly uniform compressed image onto the blank page
            new_page.insert_image(page.rect, stream=img_bytes)
            
        # Save securely with top level PDF compression on the new objects
        out_doc.save(output_path, garbage=4, deflate=True, clean=True)
        
        out_doc.close()
        doc.close()
        
        print(f"Success! Compressed PDF saved to: {output_path}")
        
        # Compare file sizes
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Original file size: {original_size:.2f} MB")
        print(f"Compressed file size: {compressed_size:.2f} MB")
        
        if original_size > 0:
            print(f"Space saved: {original_size - compressed_size:.2f} MB "
                  f"({(1 - compressed_size/original_size)*100:.1f}%)")
        return True
        
    except Exception as e:
        print(f"An error occurred during compression: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Reliably compress a scanned PDF using PyMuPDF flattening on Windows.")
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", help="Path to the output (compressed) PDF file", default=None)
    parser.add_argument("-q", "--quality", type=int, default=50, help="JPEG compression quality (1-100), default: 50")
    parser.add_argument("--dpi", type=int, default=150, help="DPI at which to rasterize pages, default: 150")

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    if output_file is None:
        file_root, ext = os.path.splitext(input_file)
        output_file = f"{file_root}_compressed{ext}"

    compress_pdf(
        input_path=input_file,
        output_path=output_file,
        quality=args.quality,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()
