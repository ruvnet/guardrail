import {spawn} from 'node:child_process';
import {readFileSync,existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {Server} from '@modelcontextprotocol/server';
import {StdioServerTransport} from '@modelcontextprotocol/server/stdio';
export const root=fileURLToPath(new URL('../../',import.meta.url));
export const policy=JSON.parse(readFileSync(new URL('./policy.json',import.meta.url)));
let active=false; const receipts=[];
function remember(result){const receipt={sha256:createHash('sha256').update(JSON.stringify(result)).digest('hex'),signed:false,at:new Date().toISOString()};receipts.push(receipt);if(receipts.length>32)receipts.shift();return {...result,receipt};}
export async function run(action,input={}) {
 if(!['test','benchmark','domain'].includes(action))throw Error('Unknown operation');
 const serialized=JSON.stringify(input);if(Buffer.byteLength(serialized)>32768)throw Error('Input limit');
 if(action!=='domain' && process.env.RUV_ALLOW_VALIDATION!=='1')throw Error('Validation requires local opt in');
 if(active)throw Error('Busy');
 active=true;
 try {
  if(action==='domain' && policy.repository==='Agent-Name-Service') {
   if(!input || Object.keys(input).join(',')!=='binding')throw Error('Expected binding only');
   const trustedPath=process.env.ANS_TRUSTED_ISSUER_FILE;
   if(!trustedPath)throw Error('Trusted issuer not configured');
   const trusted=readFileSync(trustedPath,'utf8');
   const {verifyBinding}=await import('../../v2/identity.mjs');
   return remember({valid:verifyBinding(input.binding,trusted),issuerPinned:true,revocationChecked:false});
  }
  const python=process.env.RUV_PYTHON || (existsSync(join(root,'.venv/bin/python'))?join(root,'.venv/bin/python'):'python');
  const commands=policy.repository==='guardrail'?{test:[python,['-m','unittest','discover','-s','tests','-v']],benchmark:[python,['scripts/benchmark.py']],domain:[python,['scripts/policy.py']]}:{test:[process.execPath,['--test','v2/identity.test.mjs']],benchmark:[process.execPath,['v2/benchmark.mjs']]};
  const [command,args]=commands[action];
  return await new Promise((resolve,reject)=>{
   const child=spawn(command,args,{cwd:root,env:{PATH:process.env.PATH||'',PYTHONPATH:root},stdio:['pipe','pipe','pipe'],detached:process.platform!=='win32'});
   let size=0,output='',error='',failure=null;
   function stop(message){failure=message;try{process.platform!=='win32'?process.kill(-child.pid,'SIGKILL'):child.kill('SIGKILL');}catch{}}
   const timer=setTimeout(()=>stop('Deadline exceeded'),30000);
   child.stdout.on('data',chunk=>{size+=chunk.length;if(size>65536)stop('Output limit');else output+=chunk;});
   child.stderr.on('data',chunk=>{size+=chunk.length;if(size>65536)stop('Output limit');else error+=chunk;});
   child.once('error',()=>{clearTimeout(timer);reject(Error('Unable to start operation'));});
   child.once('close',code=>{clearTimeout(timer);if(failure||code!==0)return reject(Error('Operation failed'));try{resolve(remember(action==='domain'?JSON.parse(output):{success:true,action,output,error}));}catch{reject(Error('Invalid operation response'));}});
   child.stdin.on('error',()=>{});child.stdin.end(action==='domain'?serialized:'');
  });
 } finally {active=false;}
}
export async function startMcp(){
 const server=new Server({name:policy.repository.toLowerCase()+'-agent',version:'2.0.0-alpha.1'},{capabilities:{tools:{},resources:{}}});
 const names=['project_status','project_validate','project_benchmark','project_receipts',policy.domainTool];
 server.setRequestHandler('tools/list',async()=>({tools:names.map(name=>({name,description:'Bounded local project operation. No deployment or remote publication.',inputSchema:{type:'object',properties:name===policy.domainTool?(policy.repository==='guardrail'?{details:{type:'object'},conditions:{type:'array',minItems:1,maxItems:32}}:{binding:{type:'object'}}):{},additionalProperties:false}}))}));
 server.setRequestHandler('tools/call',async request=>{try{
   const {name}=request.params,args=request.params.arguments??{};
   if(!names.includes(name)||!args||typeof args!=='object'||Array.isArray(args)||Buffer.byteLength(JSON.stringify(args))>32768)throw Error('Invalid arguments');
   if(name!==policy.domainTool&&Object.keys(args).length)throw Error('No arguments');
   if(name===policy.domainTool && policy.repository==='guardrail' && Object.keys(args).sort().join(',')!=='conditions,details')throw Error('Invalid policy fields');
   const result=name==='project_status'?policy:name==='project_receipts'?receipts:await run(name===policy.domainTool?'domain':name==='project_validate'?'test':'benchmark',args);
   return {content:[{type:'text',text:JSON.stringify(result)}]};
  }catch{return {isError:true,content:[{type:'text',text:'Operation rejected or failed'}]};}});
 server.setRequestHandler('resources/list',async()=>({resources:[{uri:policy.namespace,name:'Execution policy',mimeType:'application/json'}]}));
 server.setRequestHandler('resources/read',async request=>{if(request.params.uri!==policy.namespace)throw Error('Unknown resource');return {contents:[{uri:policy.namespace,mimeType:'application/json',text:JSON.stringify(policy)}]};});
 const transport=new StdioServerTransport(process.stdin,process.stdout,{maxBufferSize:65536});transport.onerror=()=>{process.stderr.write('Invalid MCP input\n');};
 await server.connect(transport);process.stdin.once('end',()=>void server.close());return server;
}
