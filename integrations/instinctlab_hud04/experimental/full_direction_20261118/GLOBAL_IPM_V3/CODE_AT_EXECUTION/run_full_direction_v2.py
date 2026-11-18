"""Fresh versioned execution after PRE-SOLVER spawn import failure only.

This never replays the initial intent and never relaxes its numerical budget.
"""
import os,json
from pathlib import Path

ROOT=Path(__file__).parent
old=json.loads((ROOT/'FULL_DIRECTION_RECEIPT.json').read_text())
if old['status']!='stopped' or old['numerical_descent_generated'] is not False:raise ValueError('Initial execution did not stop before solve')
log=(ROOT/'FULL_EXECUTION.log').read_text()
if "ModuleNotFoundError: No module named 'bounded_runtime'" not in log:raise ValueError('Expected pre-worker spawn failure not found')
if any((ROOT/name).exists() for name in ('LP_PROGRESS.jsonl','FULL_DIRECTION_REPORT.json','STEP1_FULL_LENGTH_JOINT_DIAGNOSTIC.npz')):raise ValueError('Initial execution has numerical evidence; review manually')
os.environ['HUD04_FULL_DIRECTION_EXECUTION']='spawn_path_fixed_v2'
from run_full_direction import main,worker

if __name__=='__main__':main()