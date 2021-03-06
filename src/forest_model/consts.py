from pathlib import Path

_this_file = Path(__file__).resolve()

DIR_REPO = _this_file.parent.parent.parent.resolve()
DATA_PATH = (DIR_REPO / "data").resolve()
DIR_SCRIPTS = (DIR_REPO / "scripts").resolve()
DIR_SRC = (DIR_REPO / "src").resolve()
DIR_MODEL = (DIR_REPO / "models").resolve()
DIR_FIXTURES = (DIR_REPO / "tests" / "fixtures").resolve()
