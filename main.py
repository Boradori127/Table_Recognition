import argparse
from pipeline import unitable

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--gpu", action='store_true')
    args.add_argument("--image_path", default='./examples/English_Table/PMC514528_004_00.png', required=False, help="path to input image directory")
    args.add_argument("--korean", action='store_true', help="Use this argument when processing Korean tables.")
    args.add_argument("--ocr", default=1, choices=[1, 2, 3], required=False, type=int, help="For Korean tables : Use OCR 1 for EasyOCR (default), 2 for PaddleOCR, or 3 for NaverClovaOCR.")
    args.add_argument("--api_url", required=False, help="Must be specified when using NaverClovaOCR.")
    args.add_argument("--secret_key", required=False, help="Must be specified when using NaverClovaOCR.")
    args.add_argument("--model_dir", default='./weights', required=False, help="Path to the model weight directory")
    args.add_argument("--vocab_dir", default='./vocab', required=False, help="Path to the vocab directory")
    args.add_argument("--d_model", default=768, required=False)
    args.add_argument("--patch_size", default=16, required=False)
    args.add_argument("--nhead", default=12, required=False)
    args.add_argument("--dropout", default=0.2, required=False)
    args.add_argument("--structure_max_seq_len", default=784, required=False)
    args.add_argument("--structure_max_decode_len", default=512, required=False)
    args.add_argument("--bbox_max_seq_len", default=1024, required=False)
    args.add_argument("--bbox_max_decode_len", default=1024, required=False)
    args.add_argument("--content_max_seq_len", default=200, required=False)
    args.add_argument("--content_max_decode_len", default=200, required=False)
    args, _ = args.parse_known_args()
    return unitable(args)
    
if __name__ == "__main__":
    pred_html, pred_bbox, pred_code = main()
    print('\n==== Predict Table Structure ==== \n', pred_html)
    print('\n==== Predict Bounding Box ==== \n', pred_bbox)
    print('\n==== Predict All Talbe HTML code ==== \n', pred_code)