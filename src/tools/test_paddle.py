
from src.ocr.paddle_engine import LocalPaddleEngine
import sys

def test_paddle():
    print("Testing PaddleOCR Init...")
    engine = LocalPaddleEngine()
    if engine.ocr:
        print("PaddleOCR Initialized Successfully.")
    else:
        print("PaddleOCR Init Failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_paddle()
