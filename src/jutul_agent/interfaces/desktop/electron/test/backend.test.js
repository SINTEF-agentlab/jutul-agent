// The supervisor and the frame codec, under plain Node (no Electron needed):
//   node --test test/
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { once } = require('node:events');
const { encode, Decoder } = require('../framing');
const { Backend } = require('../backend');

const fake = (mode) => new Backend({ command: process.execPath, args: [path.join(__dirname, '..', 'test-fixtures', 'fake_backend.js'), mode] });

async function text(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString();
}

function waitFor(backend, state) {
  if (backend.state === state) return Promise.resolve();
  return new Promise((resolve) => {
    const on = (s) => {
      if (s === state) {
        backend.off('state', on);
        resolve();
      }
    };
    backend.on('state', on);
  });
}

test('frames survive being split at every byte', () => {
  const frames = [encode({ t: 'a', id: 1 }, Buffer.from([0, 255, 1])), encode({ t: 'b' }), encode({ t: 'c' }, 'héllo')];
  const all = Buffer.concat(frames);
  for (let cut = 1; cut < all.length; cut++) {
    const decoder = new Decoder();
    const out = [...decoder.push(all.subarray(0, cut)), ...decoder.push(all.subarray(cut))];
    assert.deepEqual(out.map((f) => f.header.t), ['a', 'b', 'c']);
    assert.deepEqual([...out[0].body], [0, 255, 1]);
    assert.equal(out[2].body.toString(), 'héllo');
  }
});

test('a malformed stream throws instead of hanging', () => {
  assert.throws(() => new Decoder().push(Buffer.from('GET / HTTP/1.1\r\n\r\n')));
});

test('requests stream back, sockets echo, ping answers', async () => {
  const backend = fake('ok');
  backend.start();
  await backend.whenReady(5000);
  const answer = await backend.request({ method: 'POST', path: '/x', query: '', headers: [] }, Buffer.from('!'));
  assert.equal(answer.status, 200);
  assert.equal(await text(answer.body), 'POST /x!');

  const got = [];
  const closed = new Promise((resolve) => {
    const id = backend.openSocket({ path: '/ws', query: '', headers: [], subprotocols: [] }, {
      onAccept: () => backend.sendSocket(id, 'hi'),
      onMessage: (data) => {
        got.push(data);
        backend.closeSocket(id);
        resolve();
      },
      onClose: resolve,
    });
  });
  await closed;
  assert.deepEqual(got, ['hi']);
  await backend.ping(2000);
  await backend.stop();
  assert.equal(backend.state, 'stopped');
});

test('a crash fails what is in flight, then the backend comes back', async () => {
  const backend = fake('ok');
  backend.start();
  await backend.whenReady(5000);
  const closes = [];
  backend.openSocket({ path: '/ws', query: '', headers: [], subprotocols: [] }, {
    onAccept: () => {},
    onMessage: () => {},
    onClose: (code) => closes.push(code),
  });
  const hanging = backend.request({ method: 'GET', path: '/hang', query: '', headers: [] }, null);
  const crashing = backend.request({ method: 'GET', path: '/crash', query: '', headers: [] }, null);
  await assert.rejects(crashing);
  await assert.rejects(hanging);
  assert.deepEqual(closes, [1006]); // the page sees its socket drop and reconnects
  assert.equal(backend.state, 'restarting');
  // Callers that arrive during the restart wait for it rather than failing.
  await backend.whenReady(5000);
  const answer = await backend.request({ method: 'GET', path: '/again', query: '', headers: [] }, null);
  assert.equal(await text(answer.body), 'GET /again');
  await backend.stop();
});

test('a backend that never starts is a failure with its log, not a restart loop', async () => {
  const backend = fake('never-ready');
  backend.start();
  await waitFor(backend, 'failed');
  assert.match(backend.log.join('\n'), /no simulator for this folder/);
  await assert.rejects(backend.whenReady(100));
  await assert.rejects(backend.request({ method: 'GET', path: '/', query: '', headers: [] }, null));
  // Asking again (the status page's "Try again") starts over cleanly.
  await backend.restart({ command: process.execPath, args: [path.join(__dirname, '..', 'test-fixtures', 'fake_backend.js'), 'ok'] });
  await backend.whenReady(5000);
  await backend.stop();
});

test('an aborted request tells the backend and rejects', async () => {
  const backend = fake('ok');
  backend.start();
  await backend.whenReady(5000);
  const controller = new AbortController();
  const pending = backend.request({ method: 'GET', path: '/hang', query: '', headers: [] }, null, controller.signal);
  controller.abort();
  await assert.rejects(pending);
  assert.equal(backend.http.size, 0);
  await backend.stop();
});
