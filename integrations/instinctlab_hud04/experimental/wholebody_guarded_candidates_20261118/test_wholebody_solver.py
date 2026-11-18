import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy import sparse
import wholebody_solver as S
class SyntheticCertificateGuard:
    """TEST STUB: isolate older merit/LP tests; never a runtime certificate."""
    def scan(self,*args):return None
    def compare(self,*args):return True,{'TEST_STUB':True}
class SolverTests(unittest.TestCase):
    def test_plan_native_full997(self):
        visits=np.zeros(997,int)
        for a,b in S.plan():self.assertEqual(b-a+1,11);visits[a:b+1]+=1
        self.assertEqual(len(S.plan()),100);self.assertTrue(np.all(visits>0))
    def test_lp_failure_no_candidate(self):
        self.check_mock(False)
    def test_new_contact_rechecked_and_rejected(self):
        self.check_mock(True)
    def check_mock(self,success):
        class FakeBody:
            margin=.001;shapes=[SimpleNamespace(link='A'),SimpleNamespace(link='B')];fresh=0
            def evaluate(me,model,q,roots,jacobian=False):
                g=np.array([-1-100*roots[0,0,3]])
                record=[dict(kind='self',frame=0,first=0,second=1)]
                if jacobian:return g,sparse.csr_matrix(([1.],([0],[0])),shape=(1,len(q)*35)),record,{}
                me.fresh+=1;return g,record,{}
        body=FakeBody()
        class Model:
            def evaluate(me,q,roots,jacobian=False):
                value=np.zeros((len(q),90))
                return (value,np.zeros((len(q),90,35))) if jacobian else value
        d=np.zeros((11,35));d[:,0]=.01
        lp=SimpleNamespace(success=success,x=d.ravel(),status=0 if success else 1,nit=1)
        q=np.zeros((997,29));roots=np.tile(np.eye(4),(997,1,1));rows=[]
        with patch.object(S,'WholeBodyTerms',return_value=body),patch.object(S,'plan',return_value=[(0,10)]),\
             patch.object(S,'hard_rows',return_value=(sparse.csr_matrix((1,385)),np.zeros(1))),\
             patch.object(S,'joint_feasibility',return_value={'violation':0.}),\
             patch.object(S,'temporal_system',return_value=(np.zeros(1),sparse.csr_matrix((1,385)))),\
             patch.object(S,'temporal_values',return_value=np.zeros(1)),patch.object(S,'linprog',return_value=lp):
            if not success:
                with self.assertRaisesRegex(RuntimeError,'NO motion export'):S.solve(Model(),q,roots,np.zeros((997,90)),rows.append,certificate_guard=SyntheticCertificateGuard())
                self.assertEqual(len(rows),1);self.assertEqual(body.fresh,0)
            else:
                oq,oroot,report=S.solve(Model(),q,roots,np.zeros((997,90)),rows.append,certificate_guard=SyntheticCertificateGuard())
                self.assertEqual(body.fresh,5);self.assertEqual(report['accepted_updates'],0)
                self.assertEqual(len(rows[0]['backtracking']),5)
                self.assertTrue(all(r['all43_and903_rechecked'] for r in rows[0]['backtracking']))
                np.testing.assert_array_equal(oq,q);np.testing.assert_array_equal(oroot,roots)
if __name__=='__main__':unittest.main(verbosity=2)