import {test} from 'node:test';import assert from 'node:assert/strict';
import {Client} from '@modelcontextprotocol/client';import {StdioClientTransport} from '@modelcontextprotocol/client/stdio';import {fileURLToPath} from 'node:url';
import {run,policy} from './runtime.mjs';
test('SDK stdio lifecycle, policy and rejection boundaries',async()=>{
 const transport=new StdioClientTransport({command:process.execPath,args:[fileURLToPath(new URL('./cli.mjs',import.meta.url)),'mcp'],env:{PATH:process.env.PATH||''},stderr:'pipe'});
 const client=new Client({name:'test',version:'1'});await client.connect(transport);
 try{const listed=await client.listTools();assert.equal(listed.tools.length,5);assert.ok(listed.tools.some(t=>t.name===policy.domainTool));const status=await client.callTool({name:'project_status',arguments:{}});assert.equal(JSON.parse(status.content[0].text).automaticPromotion,false);assert.equal((await client.readResource({uri:policy.namespace})).contents.length,1);assert.equal((await client.callTool({name:'project_validate',arguments:{}})).isError,true);assert.equal((await client.callTool({name:'execute',arguments:{command:'touch /tmp/no'}})).isError,true);assert.equal((await client.callTool({name:'project_status',arguments:{path:'/etc/passwd'}})).isError,true);}finally{await client.close();}
});
test('unknown command cannot spawn',async()=>{await assert.rejects(run('shell',{}));});

import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';import {tmpdir} from 'node:os';import {join} from 'node:path';import {generateKeyPairSync} from 'node:crypto';
 test('domain operation traverses the real SDK and validates data',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'domain-sdk-'));
  const env={PATH:process.env.PATH||''};let args;
  args={details:{approved:false},conditions:[{analysis_type:'local',key:'approved',condition_type:'exists'}]};
  const transport=new StdioClientTransport({command:process.execPath,args:[fileURLToPath(new URL('./cli.mjs',import.meta.url)),'mcp'],env,stderr:'pipe'});const client=new Client({name:'domain-test',version:'1'});await client.connect(transport);
  try{const result=await client.callTool({name:policy.domainTool,arguments:args});assert.notEqual(result.isError,true);const parsed=JSON.parse(result.content[0].text);assert.equal(parsed.allowed,true);assert.equal(parsed.providerUsed,false);assert.equal(parsed.receipt.signed,false);const receipts=await client.callTool({name:'project_receipts',arguments:{}});assert.equal(JSON.parse(receipts.content[0].text).length,1);assert.equal((await client.callTool({name:policy.domainTool,arguments:{...args,command:'unexpected'}})).isError,true);}finally{await client.close();rmSync(dir,{recursive:true,force:true});}
 });
