import json
import math
import os
import random
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import PixArtSigmaPipeline


def configure_determinism():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


configure_determinism()

M0 = 9.1
BASELINE_SCORE = 17.6
CANDIDATE_TIMEOUT = 900
GENERATION_TIMEOUT = 5400


def reward_for(metric):
    improvement = max(0.0, float(metric) - M0)
    reference = BASELINE_SCORE - M0
    u = improvement / reference
    return u / (1.0 + u)


def sanitized_env(cuda_visible="0"):
    keep = {
        "HOME": "/home/agent",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH": "",
        "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": cuda_visible,
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "all"),
        "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "compute,utility"),
    }
    return keep


def run_process(command, timeout, env):
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd="/tmp", start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        raise TimeoutError(f"candidate timed out; stderr={stderr[-4000:]}")
    if proc.returncode != 0:
        raise RuntimeError(f"candidate exit={proc.returncode}; stderr={stderr[-4000:]}")
    return stdout, stderr


def reap_agent_processes():
    """Kill every process owned by the submission user in this verifier."""
    for _ in range(3):
        found = False
        for item in Path("/proc").iterdir():
            if not item.name.isdigit():
                continue
            try:
                status = (item / "status").read_text()
                uid_line = next(line for line in status.splitlines()
                                if line.startswith("Uid:"))
                uids = [int(value) for value in uid_line.split()[1:]]
                if 1001 not in uids[:2]:
                    continue
                os.kill(int(item.name), signal.SIGKILL)
                found = True
            except (FileNotFoundError, ProcessLookupError, PermissionError,
                    StopIteration, ValueError):
                continue
        if not found:
            break
        time.sleep(0.1)


def stage_candidate(entry):
    parent = Path(tempfile.mkdtemp(prefix="quant-grade-", dir="/tmp"))
    parent.chmod(0o755)
    staged_entry = parent / "quantize.py"
    staged_runner = parent / "runner.py"
    shutil.copyfile(entry, staged_entry)
    shutil.copyfile("/tests/candidate_runner.py", staged_runner)
    for path in (staged_entry, staged_runner):
        path.chmod(0o444)
    output = parent / "candidate-images"
    output.mkdir(mode=0o700)
    os.chown(output, 1001, 1001)
    return parent, staged_entry, staged_runner, output


def generate_images(prompts, output, quantized_dir=None):
    script = r'''
import json, math, random, sys, types
from pathlib import Path
import numpy as np
import torch
from diffusers import PixArtSigmaPipeline
from safetensors.torch import load_file

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False

class QuantizedLinear(torch.nn.Module):
 def __init__(self, state, prefix, info):
  super().__init__(); self.in_features=info["in_features"]; self.out_features=info["out_features"]
  for field in ("qweight","scales","l1","l2","smooth","bias"):
   key=prefix+"."+field
   if key in state: self.register_buffer(field,state[key].cuda())
 def forward(self,x):
  shape=x.shape; xs=x/self.smooth
  flat=xs.reshape(-1,self.in_features); parts=[]
  for start in range(0,self.in_features,64):
   chunk=flat[:,start:start+64]; scale=chunk.float().abs().amax(dim=1,keepdim=True)/7
   scale=torch.where(scale==0,torch.ones_like(scale),scale)
   parts.append((torch.round(chunk.float()/scale).clamp(-7,7)*scale).to(x.dtype))
  qx=torch.cat(parts,dim=1)
  wparts=[]
  for group,start in enumerate(range(0,self.in_features,64)):
   wparts.append(self.qweight[:,start:start+64].to(torch.float16)*self.scales[:,group:group+1])
  weight=torch.cat(wparts,dim=1)
  y=qx@weight.T+(flat@self.l1)@self.l2
  if hasattr(self,"bias"): y=y+self.bias
  return y.reshape(*shape[:-1],self.out_features)

def replace(root,path,module):
 parent=root
 bits=path.split(".")
 for bit in bits[:-1]: parent=getattr(parent,bit)
 setattr(parent,bits[-1],module)

p=PixArtSigmaPipeline.from_pretrained("/opt/pixart-sigma",torch_dtype=torch.float16,local_files_only=True)
if sys.argv[3] != "-":
 qdir=Path(sys.argv[3]); state=load_file(qdir/"quantized.safetensors",device="cpu")
 manifest=json.loads((qdir/"manifest.json").read_text())
 for info in manifest["layers"]: replace(p.transformer,info["name"],QuantizedLinear(state,info["name"],info))
p.to("cuda"); p.set_progress_bar_config(disable=True)
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=False)
for i,row in enumerate(rows):
 g=torch.Generator(device="cuda").manual_seed(int(row["seed"]))
 x=p(row["prompt"],num_inference_steps=20,guidance_scale=4.5,height=1024,width=1024,generator=g,output_type="pt").images[0]
 if x.shape!=(3,1024,1024) or not torch.isfinite(x).all(): raise ValueError("bad image")
 torch.save(x.detach().cpu().to(torch.float16).clamp(0,1).contiguous(),out/f"{i:04d}.pt")
'''
    qarg = "-" if quantized_dir is None else str(quantized_dir)
    run_process(["/usr/bin/python3", "-c", script, str(prompts), str(output), qarg], GENERATION_TIMEOUT, {**sanitized_env(), "HOME": "/root"})


def verify_quantized_artifact(path):
    from safetensors.torch import load_file
    manifest = json.loads((path / "manifest.json").read_text())
    state = load_file(path / "quantized.safetensors", device="cpu")
    if not isinstance(manifest.get("layers"), list) or not manifest["layers"] or manifest.get("rank") != 32:
        raise ValueError("invalid manifest")
    expected = set()
    for info in manifest["layers"]:
        name, m, n, has_bias = info["name"], int(info["in_features"]), int(info["out_features"]), bool(info["has_bias"])
        groups = math.ceil(m / 64)
        fields = {"qweight", "scales", "l1", "l2", "smooth"} | ({"bias"} if has_bias else set())
        expected |= {f"{name}.{field}" for field in fields}
        q, scales, l1, l2, smooth = (state[f"{name}.{field}"] for field in ("qweight", "scales", "l1", "l2", "smooth"))
        if q.dtype != torch.int8 or q.shape != (n, m) or (q.numel() and (int(q.min()) < -7 or int(q.max()) > 7)):
            raise ValueError("invalid qweight")
        r = l1.shape[1] if l1.ndim == 2 else -1
        specs = [(scales, torch.float16, (n, groups), True), (l1, torch.float16, (m, r), False), (l2, torch.float16, (r, n), False), (smooth, torch.float16, (m,), True)]
        if not 0 <= r <= 32:
            raise ValueError("invalid rank")
        if has_bias: specs.append((state[f"{name}.bias"], torch.float16, (n,), False))
        for tensor, dtype, shape, positive in specs:
            if tensor.dtype != dtype or tensor.shape != shape or not torch.isfinite(tensor).all() or (positive and not (tensor > 0).all()):
                raise ValueError("invalid quantized tensor")
    if set(state) != expected:
        raise ValueError("unexpected tensor state")
    expected_count = 0
    pipe = PixArtSigmaPipeline.from_pretrained("/opt/pixart-sigma", torch_dtype=torch.float16, local_files_only=True)
    original = {name: (module.in_features, module.out_features, module.bias is not None) for name, module in pipe.transformer.named_modules() if isinstance(module, torch.nn.Linear)}
    listed = {x["name"]: (x["in_features"], x["out_features"], x["has_bias"]) for x in manifest["layers"]}
    if listed != original:
        raise ValueError("manifest does not replace every original linear")
    del pipe


def safe_tensor(path):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (3, 1024, 1024):
        raise ValueError("invalid image tensor")
    tensor = tensor.to(torch.float32)
    if not torch.isfinite(tensor).all() or float(tensor.min()) < 0 or float(tensor.max()) > 1:
        raise ValueError("image outside [0,1]")
    return tensor


def score(entry, prompts):
    started = time.monotonic()
    parent = reference_dir = generated_dir = validated_parent = None
    try:
        parent, staged_entry, runner, candidate_dir = stage_candidate(entry)
        reference_dir = Path(tempfile.mkdtemp(prefix="reference-images-", dir="/tmp")) / "images"
        command = [
            "/usr/sbin/runuser", "-u", "agent", "--", "/usr/bin/setsid", "/usr/bin/python3",
            str(runner), "--entry", str(staged_entry), "--output", str(candidate_dir),
            "--rank", "32",
        ]
        try:
            _, stderr = run_process(command, CANDIDATE_TIMEOUT, sanitized_env())
        finally:
            reap_agent_processes()
        print(f"candidate stderr tail: {stderr[-2000:]}")
        verify_quantized_artifact(candidate_dir)

        validated_parent = Path(tempfile.mkdtemp(prefix="validated-quant-", dir="/tmp"))
        validated_parent.chmod(0o700)
        validated_dir = validated_parent / "artifact"
        validated_dir.mkdir(mode=0o700)
        for name in ("manifest.json", "quantized.safetensors"):
            target = validated_dir / name
            shutil.copyfile(candidate_dir / name, target)
            target.chmod(0o400)
        verify_quantized_artifact(validated_dir)
        generated_dir = Path(tempfile.mkdtemp(prefix="candidate-generated-", dir="/tmp")) / "images"
        generate_images(prompts, reference_dir)
        generate_images(prompts, generated_dir, validated_dir)
        count = sum(1 for line in prompts.read_text().splitlines() if line.strip())
        values = []
        for index in range(count):
            candidate = safe_tensor(generated_dir / f"{index:04d}.pt")
            reference = safe_tensor(reference_dir / f"{index:04d}.pt")
            mse = float(torch.mean((candidate - reference) ** 2, dtype=torch.float64))
            values.append(0.0 if not math.isfinite(mse) or mse <= 0 else 10.0 * math.log10(1.0 / mse))
        metric = sum(values) / len(values)
        return {
            "valid": True, "metric_name": "mean_psnr_db", "metric": metric,
            "reward": reward_for(metric), "examples": len(values),
            "elapsed_sec": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "valid": False, "metric_name": "mean_psnr_db", "metric": None,
            "reward": 0.0, "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.monotonic() - started,
        }
    finally:
        if parent is not None:
            shutil.rmtree(parent, ignore_errors=True)
        if reference_dir is not None:
            shutil.rmtree(reference_dir.parent, ignore_errors=True)
        if generated_dir is not None:
            shutil.rmtree(generated_dir.parent, ignore_errors=True)
        if validated_parent is not None:
            shutil.rmtree(validated_parent, ignore_errors=True)
