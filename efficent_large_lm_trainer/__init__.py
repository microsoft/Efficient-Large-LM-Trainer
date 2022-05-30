import importlib
import os

# automatically import any Python files in the current directory
cur_dir = os.path.join(os.path.dirname(__file__), "criterions")
for file in os.listdir(cur_dir):
    path = os.path.join(cur_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        mod_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(__name__ + ".criterions." + mod_name)
