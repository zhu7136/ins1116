"""CPU tests only. No Isaac, CUDA initialization, platform task or reference."""
import argparse
import ast
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import contact_instrument as instrument
import contact_instrument_worker as worker

ROOT=Path(__file__).resolve().parent


def balanced():
    g=instrument.GRAVITY
    return [[[3*g,2*g,g],[0.,2*g,2*g]],[[3*g,2*g,g],[0.,2*g,2*g]]]


class BalanceTests(unittest.TestCase):
    def test_expected_masses(self):self.assertEqual(instrument.check_balance(balanced()),0.)
    def test_net_only_is_not_pair_identity(self):
        rows=balanced();rows[0][0][0]=instrument.GRAVITY
        with self.assertRaises(RuntimeError):instrument.check_balance(rows)
    def test_missing_nonterrain(self):
        rows=balanced();rows[0][0][1]=0.
        with self.assertRaises(RuntimeError):instrument.check_balance(rows)
    def test_wrong_dt_factor(self):
        rows=[[[v*instrument.DT for v in body] for body in env] for env in balanced()]
        with self.assertRaises(RuntimeError):instrument.check_balance(rows)
    def test_nonfinite(self):
        for value in (float('nan'),float('inf'),-float('inf')):
            rows=balanced();rows[0][0][0]=value
            with self.assertRaises(ValueError):instrument.check_balance(rows)
    def test_wrong_environments(self):
        with self.assertRaises(ValueError):instrument.check_balance(balanced()[:1])
    def test_wrong_body_shape(self):
        rows=balanced();rows[0][1].append(0.)
        with self.assertRaises(ValueError):instrument.check_balance(rows)
    def test_explicit_unvalidated_scope(self):
        data=instrument.plan()
        self.assertEqual(data['physics_steps_max'],260)
        for name in ('gpu_started','reference_loaded','reference_exported',
                     'accepted_reference','training_ready','real_physics_validated'):
            self.assertIs(data[name],False)
        self.assertIn('actual environment collision-group isolation',data['NOT_TESTED'])


class WorkerTests(unittest.TestCase):
    def test_initialization_cap(self):
        self.assertIsNone(worker.deadline_reason(179.999,False))
        self.assertEqual(worker.deadline_reason(180.,False),'initialization_limit')
    def test_initialized_no_longer_uses_init_cap(self):self.assertIsNone(worker.deadline_reason(181.,True))
    def test_absolute_cap(self):self.assertEqual(worker.deadline_reason(300.,True),'child_wall_limit')
    def test_shutdown_cap(self):self.assertEqual(worker.deadline_reason(220.,True,20.),'shutdown_limit')
    def test_invalid_clock(self):
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):worker.deadline_reason(value,True)
    def test_invalid_shutdown_clock(self):
        with self.assertRaises(ValueError):worker.deadline_reason(100.,True,-1.)
    def test_evidence_roundtrip(self):
        data={'instrument_passed':False,'training_ready':False}
        self.assertEqual(worker.verify_evidence(worker.evidence(data)),data)
    def test_tampered_evidence(self):
        data=worker.evidence({'instrument_passed':False});data['report']['instrument_passed']=True
        with self.assertRaises(ValueError):worker.verify_evidence(data)
    def test_evidence_bytes(self):
        data=worker.evidence({'value':'数据'});data['bytes']+=1
        with self.assertRaises(ValueError):worker.verify_evidence(data)
    def test_oversized_evidence(self):
        with self.assertRaises(ValueError):worker.evidence({'data':'x'*33000})
    def test_nan_evidence(self):
        with self.assertRaises(ValueError):worker.evidence({'value':float('nan')})
    def test_log_overflow_warning(self):
        self.assertTrue(worker.overflow_warning(b'PHYSX Contact Buffer Overflow'))
        self.assertFalse(worker.overflow_warning(b'instrument initialized'))
    def test_exited_child_not_signalled(self):
        proc=Mock();proc.poll.return_value=0
        with patch.object(worker.os,'killpg') as kill:
            worker.stop_child(proc);kill.assert_not_called()
    def test_running_child_terminated(self):
        proc=Mock(pid=411);proc.poll.return_value=None
        with patch.object(worker.os,'killpg') as kill:
            worker.stop_child(proc)
            kill.assert_called_once_with(411,signal.SIGTERM)
            proc.wait.assert_called_once_with(timeout=3)
    def test_stubborn_child_killed(self):
        proc=Mock(pid=411);proc.poll.return_value=None
        proc.wait.side_effect=[subprocess.TimeoutExpired('synthetic-child',3),0]
        with patch.object(worker.os,'killpg') as kill:
            worker.stop_child(proc)
            self.assertEqual([item.args for item in kill.call_args_list],[(411,signal.SIGTERM),(411,signal.SIGKILL)])
    def test_child_exited_during_signal(self):
        proc=Mock(pid=411);proc.poll.return_value=None
        with patch.object(worker.os,'killpg',side_effect=ProcessLookupError):worker.stop_child(proc)
    def test_default_worker_never_spawns(self):
        args=argparse.Namespace(execute_instrument=False)
        with patch.object(worker.subprocess,'Popen') as start,patch.object(worker,'event'):
            self.assertEqual(worker.main(args),0);start.assert_not_called()
    def test_execute_requires_literal_commit(self):
        args=argparse.Namespace(execute_instrument=True,expected_commit='$(git rev-parse HEAD)')
        with patch.object(worker.subprocess,'Popen') as start:
            with self.assertRaises(ValueError):worker.main(args)
            start.assert_not_called()
    def test_plan_entrypoints_are_cpu_only(self):
        for filename in ('contact_instrument.py','contact_instrument_worker.py'):
            result=subprocess.run([sys.executable,'-B',str(ROOT/filename)],capture_output=True,text=True,timeout=10,check=True)
            data=json.loads(result.stdout)
            self.assertFalse(data['gpu_started'])
            self.assertEqual(data['ppo_updates'],0)
    def test_no_old_worker_or_training_launch(self):
        text=(ROOT/'contact_instrument_worker.py').read_text()
        self.assertNotIn('runpy',text)
        self.assertNotIn('train.py',text)
        self.assertNotIn('gm-run',text)
        self.assertIn(".open('x')",text)
    def test_source_manifest(self):self.assertEqual(set(worker.audit_sources()['files']),set(worker.INPUTS))


if __name__=='__main__':unittest.main()