import copy
from types import SimpleNamespace as S
import unittest

from contact_material_links_1073 import copy_contact_material_links, material_prim_path
from contact_report_1073 import ContactReport1073, NON_INSTANCED


class ContactMaterialTests(unittest.TestCase):
    def setUp(self):
        self.paths = {1:'/World/Robot/a', 2:'/World/ground',
                      3:'/World/Robot/a/collider',4:'/World/ground/mesh',
                      5:'/Looks/RobotMat',6:'/physicsScene/defaultMaterial',7:'/Looks/RobotMatAlt'}
        self.schema = S(intToSdfPath=self.paths.__getitem__)
        self.registry = [dict(key='expected_pair',actor0=self.paths[1],actor1=self.paths[2],
                              collider0=self.paths[3],collider1=self.paths[4])]
        self.header = S(actor0=1,actor1=2,collider0=3,collider1=4,stage_id=42,
                        proto_index0=NON_INSTANCED,proto_index1=NON_INSTANCED,
                        contact_data_offset=0,num_contact_data=1,type='CONTACT_FOUND')
        self.point = S(material0=5,material1=6,face_index0=-1,face_index1=17,
                       position=[0,0,0],normal=[0,0,1],impulse=[0,0,.1],separation=-.002)

    def binding(self):
        return ContactReport1073(self.registry,self.schema,expected_stage_id=42)

    def test_actual_event_point_side_join(self):
        b=self.binding();b.callback([self.header],[self.point]);r=b.report()
        e=r['events'][0];link=e['contacts'][0]['contact_material_links']
        self.assertEqual(e['collider0'],self.paths[3])
        self.assertEqual(link['material0']['decoded_material_prim_path'],self.paths[5])
        self.assertEqual(link['material1']['decoded_material_prim_path'],self.paths[6])
        self.assertIsNone(link['effective_pair_friction'])
        self.assertFalse(link['runtime_material_values_verified'])
        self.assertFalse(r['actual_target_build_verified'])

    def test_reversed_header_not_silently_canonicalized(self):
        h=copy.copy(self.header);p=copy.copy(self.point)
        h.actor0,h.actor1=h.actor1,h.actor0;h.collider0,h.collider1=h.collider1,h.collider0
        p.material0,p.material1=p.material1,p.material0
        b=self.binding();b.callback([h],[p]);e=b.report()['events'][0]
        self.assertEqual(e['matched_exact_pair'],'expected_pair')
        self.assertEqual(e['collider0'],self.paths[4])
        self.assertEqual(e['contacts'][0]['contact_material_links']['material0']['decoded_material_prim_path'],self.paths[6])

    def test_shared_collider_different_contact_material_retained(self):
        p=copy.copy(self.point);p.material0=7;p.face_index0=18
        self.header.num_contact_data=2
        b=self.binding();b.callback([self.header],[self.point,p]);cs=b.report()['events'][0]['contacts']
        ids=[c['contact_material_links']['material0']['native_uint64_id'] for c in cs]
        self.assertEqual(ids,[5,7])

    def test_missing_fields_not_defaulted(self):
        r=copy_contact_material_links(S(),self.paths.__getitem__)
        for f in ('material0','material1','face_index0','face_index1'):
            self.assertEqual(r[f]['status'],'FIELD_UNAVAILABLE')
        self.assertIsNone(r['material0']['native_uint64_id'])

    def test_zero_not_default_material(self):
        calls=[]
        r=copy_contact_material_links(S(material0=0,material1=0),lambda x:calls.append(x))
        self.assertEqual(calls,[])
        self.assertEqual(r['material0']['status'],'ZERO_ID_UNRESOLVED')

    def test_invalid_id_not_decoded(self):
        for value in (True,-1,1.,2**64):
            calls=[]
            r=copy_contact_material_links(S(material0=value),lambda x:calls.append(x))
            self.assertEqual(calls,[])
            self.assertEqual(r['material0']['status'],'INVALID_ID')

    def test_decoder_failure_not_contact_loss(self):
        self.point.material0=99;b=self.binding();b.callback([self.header],[self.point]);r=b.report()
        self.assertTrue(r['ingestion_complete_within_budget'])
        link=r['events'][0]['contacts'][0]['contact_material_links']
        self.assertEqual(link['material0']['native_uint64_id'],99)
        self.assertEqual(link['material0']['status'],'DECODE_ERROR_OR_UNSUPPORTED_PATH')
        self.assertIsNone(link['material0']['decoded_material_prim_path'])

    def test_stale_stage_rejected_before_material_decode(self):
        calls=[];self.header.stage_id=43
        b=ContactReport1073(self.registry,S(intToSdfPath=lambda v:calls.append(v)),expected_stage_id=42)
        b.callback([self.header],[self.point]);r=b.report()
        self.assertEqual(calls,[]);self.assertEqual(r['events'],[])
        self.assertFalse(r['ingestion_complete_within_budget'])

    def test_transient_point_and_report_no_alias(self):
        b=self.binding();b.callback([self.header],[self.point]);self.point.material0=7
        r=b.report();link=r['events'][0]['contacts'][0]['contact_material_links']
        self.assertEqual(link['material0']['native_uint64_id'],5)
        link['material0']['decoded_material_prim_path']='/Looks/Changed'
        self.assertEqual(b.report()['events'][0]['contacts'][0]['contact_material_links']['material0']['decoded_material_prim_path'],self.paths[5])

    def test_raw_faces_no_sentinel_assumption(self):
        for value in (-1,0,2**32-1):
            r=copy_contact_material_links(S(face_index0=value),self.paths.__getitem__)
            self.assertEqual(r['face_index0']['value'],value)
            self.assertEqual(r['face_index0']['status'],'RAW_INTEGER_NOT_COOKED_FACE_BINDING')

    def test_material_path_not_limited_to_world(self):
        for value in ('/Looks/M','/physicsScene/defaultMaterial','/World/Looks/M'):
            self.assertEqual(material_prim_path(value),value)
        for value in ('','/','/Looks/../M','/Looks//M','/Looks/M.attr','/Looks/*','/Looks/M/'):
            with self.assertRaises(ValueError):material_prim_path(value)

    def test_zero_contact_no_material_evidence(self):
        b=self.binding();self.header.num_contact_data=0
        b.callback([self.header],[]);r=b.report()
        self.assertEqual(r['events'][0]['contacts'],[])
        self.assertFalse(r['subscription_and_time_coverage_verified'])
        self.assertFalse(r['training_ready'])


if __name__=='__main__':unittest.main()