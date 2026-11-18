// A stand-in for the Python backend: speaks the frame protocol on stdin/stdout.
// Behaviour is chosen by argv[2] so one file covers the supervisor's cases.
'use strict';
const { encode, Decoder } = require('../framing');

const mode = process.argv[2] || 'ok';
const send = (header, body) => process.stdout.write(encode(header, body));

if (mode === 'never-ready') {
  console.error('error: no simulator for this folder');
  process.exit(2);
}

const decoder = new Decoder();
process.stdin.on('data', (chunk) => {
  for (const { header, body } of decoder.push(chunk)) {
    if (header.t === 'http') {
      if (header.path === '/crash') process.exit(7);
      if (header.path === '/hang') continue; // never answers
      send({ t: 'http.start', id: header.id, status: 200, headers: [['content-type', 'text/plain']] });
      send({ t: 'http.body', id: header.id, more: true }, Buffer.from(`${header.method} `));
      send({ t: 'http.body', id: header.id, more: false }, Buffer.concat([Buffer.from(header.path), body]));
    } else if (header.t === 'ws.open') {
      send({ t: 'ws.accept', id: header.id, subprotocol: null });
    } else if (header.t === 'ws.msg') {
      send({ t: 'ws.msg', id: header.id, binary: header.binary }, body);
    } else if (header.t === 'ping') {
      send({ t: 'pong', id: header.id });
    } else if (header.t === 'shutdown') {
      process.exit(0);
    }
  }
});
process.stdin.on('end', () => process.exit(0));
send({ t: 'ready', pid: process.pid, version: 'test' });
