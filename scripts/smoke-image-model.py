"""Fail the image build if the bundled multilingual model cannot infer offline."""

import os
from pathlib import Path

from server.laya_adapter import LayaAdapter


model_dir = Path(os.environ["LAYA_MODEL_DIR"])
assert os.environ["LAYA_MODEL_PROFILE"] == "multilingual"
assert (model_dir / "multilingual" / "model.safetensors").is_file()
for unavailable in ("english", "typed-decisions"):
    assert not (model_dir / unavailable).exists()

predictor = LayaAdapter(model_dir, os.environ.get("LAYA_DEVICE", "cuda"), 1, "multilingual")
question = {"flag": {"type": "noul", "instructions": "Does this customer request a refund?"}}
for state, max_len in (("Please refund the duplicate charge.", None), ("请退还重复扣除的费用。", 8192)):
    result = predictor.predict(state, question, max_len=max_len)
    assert result["routing"]["model"] == "multilingual", result
    assert "flag" in result["answers"], result
# Confirm that the actual model parameters reside on the requested device.
agent = predictor.router.load("multilingual")
actual_device = str(next(agent.model.parameters()).device)
assert actual_device.split(":")[0] == predictor.device.split(":")[0], actual_device
if ":" in predictor.device:
    assert actual_device == predictor.device, actual_device
print(f"Bundled multilingual model passed offline inference on {predictor.device} with default and extended budgets")
