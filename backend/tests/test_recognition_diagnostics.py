"""Diagnostic CLI must not make paid requests or write source-controlled outputs by accident."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/inspect_recognition.py'


def test_hybrid_comparison_requires_explicit_cloud_consent():
    result = subprocess.run([sys.executable,'-B',str(SCRIPT),'--mode','both'],
                            cwd=ROOT,capture_output=True,text=True,timeout=10)
    assert result.returncode == 2
    assert '--allow-cloud' in result.stderr


def test_diagnostics_reject_outputs_outside_ignored_generated_directory(tmp_path):
    result = subprocess.run([sys.executable,'-B',str(SCRIPT),'--output',str(tmp_path)],
                            cwd=ROOT,capture_output=True,text=True,timeout=10)
    assert result.returncode == 2
    assert 'Git-ignored data/generated' in result.stderr
    assert not list(tmp_path.iterdir())
