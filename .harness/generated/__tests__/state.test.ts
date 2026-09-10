import {it,expect} from 'vitest';
import {mkdtemp,rm,appendFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {SessionLog} from '../src/sessions/log.js';
import {openFieldMemory} from '../src/field-memory.js';
it('session replay survives restart and preserves fork ancestry',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'repo-session-'));
 try {const path=join(dir,'log.jsonl');const log=await SessionLog.open(path);
 await log.append('validation',{pass:true});const fork=await log.fork(0,'candidate');
 await fork.append('benchmark',{p95:1});const before=fork.replay();
 const restored=await SessionLog.open(path,'candidate');expect(restored.replay()).toEqual(before);
 expect(before.eventCount).toBe(3);expect(await restored.validate()).toEqual([]);
 await appendFile(path,'invalid json\n');await expect(SessionLog.open(path)).rejects.toThrow();
 }finally{await rm(dir,{recursive:true,force:true});}
});
it('field memory denies absent authentication and implicit storage',async()=>{
 await expect(openFieldMemory(undefined as any)).rejects.toThrow();
 await expect(openFieldMemory({storagePath:'relative'} as any)).rejects.toThrow('absolute');
 await expect(openFieldMemory({storagePath:'/tmp/explicit',openStorage:()=>({}),verifier:()=>null,identityHashKey:'short'} as any)).rejects.toThrow('32 bytes');
});

it('field memory verifies evidence, rejects replay, and hides unsupported heads',async()=>{
 const {InMemoryFieldStorage}=await import('@metaharness/field-memory');
 const memory=await openFieldMemory({storagePath:'/test/explicit-memory',openStorage:()=>new InMemoryFieldStorage({dimension:384,metric:'cosine'}),verifier:async(proof:unknown)=>proof==='fixture-proof'?{principalId:'fixture',trustDomain:'test'}:null,identityHashKey:new Uint8Array(32).fill(7)});
 const embedding=Array(384).fill(0);embedding[0]=1;
 const update={centroidId:'repo-tests',embedding,configurationId:'candidate',reward:1,cost:0,observedAt:Date.now(),idempotencyKey:'receipt-1',principalProof:'invalid'};
 expect((await memory.update(update)).status).toBe('verification-failed');
 const accepted=await memory.update({...update,principalProof:'fixture-proof'});
 expect(accepted.status).toBe('privacy-buffered');
 expect((await memory.update({...update,principalProof:'fixture-proof'})).status).toBe('duplicate');
 expect(await memory.choose({embedding,allowedConfigurations:['candidate']})).toBeNull();
});
