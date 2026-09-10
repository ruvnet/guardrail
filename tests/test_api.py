import os, unittest, asyncio
from fastapi.testclient import TestClient
from main import app,check_condition,Condition,AnalysisResult,AnalysisRequest,perform_analysis_based_on_type
from unittest.mock import AsyncMock,patch
os.environ['AUTH_TOKEN']='x'*32
class ApiTests(unittest.TestCase):
    def setUp(self): self.client=TestClient(app);self.headers={'Authorization':'Bearer '+'x'*32}
    def payload(self): return {'details':{'ok':False},'conditions':[{'analysis_type':'local','key':'ok','condition_type':'exists'}]}
    def test_http_authenticated_evaluation(self):
        r=self.client.post('/evaluate/',json=self.payload(),headers=self.headers);self.assertEqual(r.status_code,200);self.assertTrue(r.json()['allowed']);self.assertFalse(r.json()['providerUsed'])
    def test_missing_or_short_auth_denied(self):
        self.assertIn(self.client.post('/evaluate/',json=self.payload()).status_code,[401,403])
        with patch.dict(os.environ,{'AUTH_TOKEN':'short'}): self.assertEqual(self.client.post('/evaluate/',json=self.payload(),headers=self.headers).status_code,403)
    def test_request_size_bound(self): self.assertEqual(self.client.post('/evaluate/',content=b'x'*65537,headers=self.headers).status_code,413)
    def test_empty_conditions_invalid(self):
        p=self.payload();p['conditions']=[];self.assertEqual(self.client.post('/evaluate/',json=p,headers=self.headers).status_code,422)
    def test_missing_all_operators_deny(self):
        for kind in ['greater','less','equal','contains','is_type','length_equal','regex_match','key_value_pair']:
            c=Condition(analysis_type='local',key='missing',condition_type=kind,threshold='dict');self.assertFalse(check_condition(AnalysisResult(analysis='local',details={},error=None),c))
    def test_numeric_empty_bool_nonfinite_deny(self):
        c=Condition(analysis_type='local',key='x',condition_type='greater',threshold=0)
        for value in [[],True,float('inf'),float('nan')]:self.assertFalse(check_condition(AnalysisResult(analysis='local',details={'x':value},error=None),c))
    def test_regex_invalid_and_pathological_deny(self):
        for pattern,value in [('(', 'x'),('(a+)+$', 'a'*12000+'!')]:
            c=Condition(analysis_type='local',key='x',condition_type='regex_match',threshold=pattern);self.assertFalse(check_condition(AnalysisResult(analysis='local',details={'x':value},error=None),c))
    def test_type_bool_is_not_int(self):
        c=Condition(analysis_type='local',key='x',condition_type='is_type',threshold='int');self.assertFalse(check_condition(AnalysisResult(analysis='local',details={'x':True},error=None),c))
    def test_parallel_analysis_does_not_mutate(self):
        request=AnalysisRequest()
        with patch('main.perform_analysis',new=AsyncMock(return_value=None)) as mock:
            asyncio.run(perform_analysis_based_on_type(request,'other'));self.assertEqual(request.analysis_type,'sentiment_analysis');self.assertEqual(mock.call_args.args[0].analysis_type,'other')

    def test_unicode_invalid_token_is_denied(self):
        from main import get_current_user
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        with self.assertRaises(HTTPException) as caught:
            get_current_user(HTTPAuthorizationCredentials(scheme='Bearer',credentials='é'*32))
        self.assertEqual(caught.exception.status_code,403)
