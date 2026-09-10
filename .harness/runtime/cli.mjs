import {policy,run,startMcp} from './runtime.mjs';
try {
 const command=process.argv[2]||'status';
 if(command==='mcp')await startMcp();
 else if(command==='status')console.log(JSON.stringify(policy));
 else if(['test','benchmark'].includes(command))console.log(JSON.stringify(await run(command)));
 else if(['evaluate','verify'].includes(command)){
  let body='';for await(const chunk of process.stdin){body+=chunk;if(Buffer.byteLength(body)>32768)throw Error('Input limit');}
  console.log(JSON.stringify(await run('domain',JSON.parse(body))));
 }else throw Error('Unknown command');
}catch{process.stderr.write('Operation rejected or failed\n');process.exitCode=1;}
