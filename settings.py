from pathlib import Path
import sys

FILE = Path(__file__).resolve()

ROOT = FILE.parent

if ROOT not in sys.path:
    sys.path.append(str(ROOT))

ROOT = ROOT.relative_to(Path.cwd())

IMAGE = 'Image'

SOURCES_LIST = [IMAGE]

IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'tooth-decay-image.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'tooth-decay-image.jpg'

MODEL_DIR = ROOT / 'weights'

DETECTION_MODEL = MODEL_DIR / 'best.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'sam_b.pt'