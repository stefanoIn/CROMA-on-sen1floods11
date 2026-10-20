"""Execute the authors notebook for all four checkpoints on v1.0 and v1.1."""
import os
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
SOURCE = Path("experiments/sen1floods11_reproduction/authors/Test_Modelsv0v0.ipynb")
suite_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + uuid4().hex[:8]
suite_dir = Path("runs/sen1floods11_reproduction") / ("comparison_" + suite_id)
suite_dir.mkdir(parents=True, exist_ok=False)
manifest = {"suite_id": suite_id, "notebook": SOURCE.as_posix(), "runs": []}
print("SUITE", suite_dir, flush=True)
for version in ("v1.0", "v1.1"):
    for model in ("hand_labeled", "s1_weak", "s2_weak", "permanent_water"):
        os.environ.update(FCNN_MODEL=model, FCNN_DATASET=version, FCNN_RUN_ID=suite_id)
        book = nbformat.read(SOURCE, as_version=4)
        executed = suite_dir / f"{model}_{version}.ipynb"
        print("START", model, version, flush=True)
        try:
            NotebookClient(book, timeout=1200, kernel_name="thesis-d-cuda", resources={"metadata": {"path": str(ROOT)}}).execute()
        finally:
            nbformat.write(book, executed)
        matches = list((Path("runs/sen1floods11_reproduction") / ("authors_" + model)).glob(f"*_{version}_{suite_id}"))
        assert len(matches) == 1
        folder = matches[0]
        assert len(list((folder / "predictions/non_bol").glob("*.tif"))) == 90
        assert len(list((folder / "predictions/bol").glob("*.tif"))) == 15
        manifest["runs"].append({"model": model, "test_dataset": version, "directory": folder.as_posix(), "executed_notebook": executed.as_posix()})
        (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("DONE", model, version, "105 predictions", flush=True)
print("COMPLETE", suite_dir, flush=True)
