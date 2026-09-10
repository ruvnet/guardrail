import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import PolicyRequest, evaluate_local
body=sys.stdin.buffer.read(32769)
if len(body)>32768: raise ValueError('Input limit')
request=PolicyRequest.model_validate_json(body)
print(json.dumps(evaluate_local(request),allow_nan=False))
