"""Serve the accompanying draft checkpoint with a separately supplied target model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--target', required=True, help='Local Qwen3.8-27B target model directory')
p.add_argument('--sglang-source', required=True, help='python directory of the pinned SGLang source')
p.add_argument('--checkpoint', default=str(ROOT / 'checkpoint'))
p.add_argument('--gpu', required=True, type=int, help='One physical GPU index')
p.add_argument('--port', type=int, default=30000)
p.add_argument('--reader-backend', choices=['reference', 'triton'], default='triton')
p.add_argument('--flashinfer-cubins', help='Matching FlashInfer cubin directory, if separately installed')
p.add_argument('--check-only', action='store_true')
a = p.parse_args()
checkpoint = Path(a.checkpoint).resolve()
manifest = json.loads((checkpoint / 'SHA256.json').read_text())
for name, expected in manifest.items():
    with (checkpoint / name).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, name
engine = Path(a.sglang_source).resolve()
assert (engine / 'sglang/srt/speculative/dspark_components/dspark_draft.py').is_file()
engine_manifest = '\n'.join(
    f'{path.relative_to(engine).as_posix()}\0{hashlib.sha256(path.read_bytes()).hexdigest()}'
    for path in sorted(engine.rglob('*.py'))
)
expected_engine = json.loads((ROOT / 'environment.json').read_text())['engine_python_sha256']
assert hashlib.sha256(engine_manifest.encode()).hexdigest() == expected_engine, 'Inference engine source differs from the pinned revision'
assert Path(a.target).is_dir()
if a.check_only:
    print('Checkpoint and inference engine hashes verified.')
    raise SystemExit(0)
env = {k: v for k, v in os.environ.items() if not k.startswith(('READY_', 'PYTHONPATH', 'SGLANG_'))}
env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), READY_CHECKPOINT=str(checkpoint), OMP_NUM_THREADS='8',
           READY_READER_BACKEND=a.reader_backend,
           SGLANG_RAGGED_VERIFY_MODE='static',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           PYTHONPATH=os.pathsep.join(map(str, [ROOT/'ready_inference/hooks', ROOT, engine])))
if a.flashinfer_cubins:
    env['FLASHINFER_CUBIN_DIR'] = str(Path(a.flashinfer_cubins).resolve())
cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(Path(a.target).resolve()),
       '--speculative-algorithm', 'DSPARK', '--speculative-draft-model-path', str(checkpoint),
       '--speculative-draft-model-quantization', 'unquant', '--speculative-dspark-block-size', '7',
       '--speculative-num-steps', '1', '--speculative-eagle-topk', '1',
       '--attention-backend', 'triton', '--speculative-draft-attention-backend', 'triton',
       '--kv-cache-dtype', 'bfloat16', '--mamba-ssm-dtype', 'float32',
       '--mamba-radix-cache-strategy', 'extra_buffer', '--mamba-full-memory-ratio', '11.93',
       '--mem-fraction-static', '0.6', '--context-length', '16384', '--max-running-requests', '1',
       '--max-total-tokens', '16384', '--skip-server-warmup', '--cuda-graph-bs', '1', '2', '4',
       '--disable-piecewise-cuda-graph', '--host', '127.0.0.1', '--port', str(a.port),
       '--nccl-port', str(a.port+1000), '--dtype', 'bfloat16', '--random-seed', '20261014']
os.execvpe(sys.executable, cmd, env)
