import tempfile, pickle
from sacred.run import Run
from renard.pipeline.core import PipelineState


def archive_pipeline_state_(_run: Run, state: PipelineState, name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        pickle_path = f"{tmpdir}/{name}.pickle"
        with open(pickle_path, "wb") as f:
            pickle.dump(state, f)
        _run.add_artifact(pickle_path)
