# Module imports
import nexport
import sys
import os

# File imports
from nexport.pytorch import exporting as npte
from nexport.pytorch import importing as npti
from nexport.tensorflow import exporting as ntfe
from nexport.tensorflow import importing as ntfi

# External function visibility
__all__ = ['detect_framework', 'export']


# Module functions

def detect_framework(imported: object = sys.modules.keys()) -> str:
    """
    Detect deep learning framework
    in use from imported modules.
    """
    frameworks = {
        "torch": "pytorch",
        "tensorflow": "tensorflow"
    }
    print(frameworks)

    if framework := "torch" in imported:
        if framework := "tensorflow" in imported:
            print("both frameworks imported.")
            return "both"
        print("torch detected!")
        return frameworks['torch']
    elif framework := "tensorflow" in imported:
        print("tensorflow detected!")
        return frameworks['tensorflow']
    else:
        print("no framework imported...")
        return "unknown"


def export(model: object, filetype: str, filename: str = "model", indent: int = 4, verbose: int = 1, include_metadata: bool = False, model_name: str = "My Model", model_author: str = os.getlogin()):
    match nexport.__framework__:
        case "pytorch":
            match filetype:
                case "txt":
                    npte.export_to_file(model=model, filename=filename)
                case "json":
                    npte.export_to_json(model=model, filename=filename, indent=indent, verbose=verbose, include_metadata=include_metadata, model_name=model_name, model_author=model_author)
                case "csv", "xml":
                    raise NotImplementedError(f"This feature ({filetype} for {nexport.__framework__}) has not yet been implemented.")
                case other:
                    raise RuntimeError(f"This filetype ({other}) is unrecognized and will not be supported in the near future.")
        case "tensorflow":
            match filetype:
                case "txt", "json", "csv", "xml":
                    raise NotImplementedError(f"This feature ({filetype} for {nexport.__framework__}) has not yet been implemented.")
                case other:
                    raise RuntimeError(f"This filetype ({other}) is unrecognized and will not be supported in the near future.")