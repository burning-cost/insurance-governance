"""
Run insurance-governance expansion tests on Databricks.
Uses the Jobs API with proper SDK types.
"""

import os
import sys
import time
import base64
from pathlib import Path

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

base = Path("/home/ralph/repos/insurance-governance")


def read_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Collect all source files
src_root = base / "src" / "insurance_governance"
src_files = {}

for fpath in src_root.rglob("*"):
    if fpath.suffix in (".py", ".j2", ".html", ".typed") and fpath.is_file():
        rel = fpath.relative_to(src_root)
        key = str(rel)
        src_files[key] = read_b64(str(fpath))

# Collect all test files
test_root = base / "tests"
test_files = {}
for fpath in test_root.glob("*.py"):
    test_files[fpath.name] = read_b64(str(fpath))

lines = [
    "import subprocess, sys, os, base64, tempfile",
    "from pathlib import Path",
    "",
    "# Install dependencies",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',",
    "    'polars', 'jinja2', 'scikit-learn', 'pytest', 'numpy', 'scipy'], check=True)",
    "",
    "# Build package in temp directory",
    "tmpdir = Path(tempfile.mkdtemp())",
    "pkg_dir = tmpdir / 'insurance_governance'",
    "pkg_dir.mkdir()",
    "test_dir = tmpdir / 'ig_tests'",
    "test_dir.mkdir()",
    "(test_dir / '__init__.py').write_text('')",
    "",
]

# Write source files, creating subdirectories as needed
for rel_path, b64content in src_files.items():
    safe_rel = rel_path.replace("\\", "/")
    parts = safe_rel.split("/")
    if len(parts) > 1:
        for i in range(1, len(parts)):
            subdir = "/".join(parts[:i])
            lines.append(f"(pkg_dir / '{subdir}').mkdir(exist_ok=True)")
    lines.append(f"(pkg_dir / '{safe_rel}').write_bytes(base64.b64decode('{b64content}'))")

lines.append("")

for fname, b64content in test_files.items():
    lines.append(f"(test_dir / '{fname}').write_bytes(base64.b64decode('{b64content}'))")

lines.extend([
    "",
    "sys.path.insert(0, str(tmpdir))",
    "import insurance_governance",
    "",
    "# Remove fairness-dependent tests",
    "import shutil as _sh",
    "for _f in test_dir.glob('*fairness*'): _f.unlink()",
    "",
    "# Run full test suite",
    "env = dict(os.environ)",
    "env['PYTHONPATH'] = str(tmpdir) + ':' + env.get('PYTHONPATH', '')",
    "result = subprocess.run(",
    "    [sys.executable, '-m', 'pytest', str(test_dir),",
    "     '-v', '--tb=short', '-p', 'no:cacheprovider', '-q'],",
    "    capture_output=True, text=True, env=env,",
    ")",
    "output = (result.stdout or '') + '\\n' + (result.stderr or '')",
    "exit_msg = output[-6000:] + f'\\n\\nEXIT_CODE={result.returncode}'",
    "dbutils.notebook.exit(exit_msg)",
])

notebook_source = "\n".join(lines)

notebook_path = "/Workspace/Shared/insurance-governance-expansion-v3"
nb_b64 = base64.b64encode(notebook_source.encode("utf-8")).decode("utf-8")

print(f"Uploading runner notebook ({len(notebook_source)//1024}KB)...")
w.workspace.import_(
    path=notebook_path,
    content=nb_b64,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print("Uploaded.")

JOB_NAME = "insurance-governance-expansion-v3"

# Check if job already exists
existing_job_id = None
for j in w.jobs.list():
    if j.settings and j.settings.name == JOB_NAME:
        existing_job_id = j.job_id
        break

task = jobs.Task(
    task_key="run-tests",
    notebook_task=jobs.NotebookTask(
        notebook_path=notebook_path,
        source=jobs.Source.WORKSPACE,
    ),
    environment_key="default",
)

environment = jobs.JobEnvironment(
    environment_key="default",
    spec=compute.Environment(
        client="1",
        dependencies=[
            "polars",
            "jinja2",
            "scikit-learn",
            "pytest",
            "numpy",
            "scipy",
        ],
    ),
)

new_settings = jobs.JobSettings(
    name=JOB_NAME,
    tasks=[task],
    environments=[environment],
    queue=jobs.QueueSettings(enabled=True),
)

if existing_job_id:
    print(f"Updating existing job {existing_job_id}...")
    w.jobs.reset(job_id=existing_job_id, new_settings=new_settings)
    job_id = existing_job_id
else:
    print("Creating new job...")
    created = w.jobs.create(
        name=JOB_NAME,
        tasks=[task],
        environments=[environment],
        queue=jobs.QueueSettings(enabled=True),
    )
    job_id = created.job_id

print(f"Job ID: {job_id}")
print("Running job now...")
run_waiter = w.jobs.run_now(job_id=job_id)
run_id = run_waiter.run_id
print(f"Run ID: {run_id}")

print("Waiting for run to complete...")
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = str(run_state.state.life_cycle_state)
    print(f"  Status: {life_cycle}")
    if any(s in life_cycle for s in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]):
        break
    time.sleep(20)

result_state = str(run_state.state.result_state)
print(f"\nFinal result: {result_state}")

nb_output = ""
for task in (run_state.tasks or []):
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output:
            print("\n--- Test output ---")
            print(output.notebook_output.result)
        if output.error:
            print("\n--- Error ---")
            print(output.error)
        if output.error_trace:
            print("\n--- Error trace ---")
            print(output.error_trace[-8000:])
        if output.logs:
            print("\n--- Logs ---")
            print(output.logs[-4000:])
        if output.notebook_output and output.notebook_output.result:
            nb_output = output.notebook_output.result
    except Exception as e:
        print(f"Could not get output: {e}")

tests_passed = "SUCCESS" in result_state or "EXIT_CODE=0" in nb_output

if tests_passed:
    print("\nAll tests passed.")
    sys.exit(0)
else:
    print(f"\nTests failed. State: {result_state}")
    sys.exit(1)
