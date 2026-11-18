import unittest
from types import SimpleNamespace
import foot_guard as F
class GuardTests(unittest.TestCase):
    def test_new_frame_forbidden_even_lower_score(self):
        self.assertFalse(F.guard(({(0,'L')},3.),({(1,'L')},1.))[0])
    def test_score_increase_forbidden_even_lower_count(self):
        self.assertFalse(F.guard(({(0,'L'),(1,'L')},1.),({(1,'L')},3.))[0])
    def test_improvement_permitted(self):
        self.assertTrue(F.guard(({(0,'L'),(1,'L')},3.),({(0,'L')},2.))[0])
    def test_includes_only_boxfeet_terrain(self):
        body=SimpleNamespace(shapes=[SimpleNamespace(kind='box',link='left_ankle_roll_link'),SimpleNamespace(kind='sphere',link='left_ankle_roll_link')])
        r=[dict(kind='terrain',frame=0,first=0),dict(kind='self',frame=1,first=0),dict(kind='terrain',frame=2,first=1)]
        self.assertEqual(F.foot_state(body,[-.01,-1.,-2.],r),({(0,'left_ankle_roll_link')},.01))
if __name__=='__main__':unittest.main(verbosity=2)