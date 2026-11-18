// The frame format shared with the Python backend (see ../framing.py): an
// 8-byte prefix (two big-endian uint32: header length, body length), a UTF-8
// JSON header, and a raw body.
'use strict';

const MAX_HEADER = 4 * 1024 * 1024;
const MAX_BODY = 1024 * 1024 * 1024;

function toBuffer(body) {
  if (body == null) return null;
  if (Buffer.isBuffer(body)) return body;
  if (typeof body === 'string') return Buffer.from(body, 'utf8');
  if (body instanceof ArrayBuffer) return Buffer.from(body);
  if (ArrayBuffer.isView(body)) return Buffer.from(body.buffer, body.byteOffset, body.byteLength);
  throw new TypeError('unsupported frame body');
}

function encode(header, body) {
  const head = Buffer.from(JSON.stringify(header), 'utf8');
  const payload = toBuffer(body);
  const prefix = Buffer.allocUnsafe(8);
  prefix.writeUInt32BE(head.length, 0);
  prefix.writeUInt32BE(payload ? payload.length : 0, 4);
  return payload ? Buffer.concat([prefix, head, payload]) : Buffer.concat([prefix, head]);
}

/** Incremental decoder: feed it chunks as they arrive, get whole frames back. */
class Decoder {
  constructor() {
    this.chunks = [];
    this.length = 0;
  }

  /** Returns the frames completed by `chunk`. Throws on a malformed stream. */
  push(chunk) {
    this.chunks.push(chunk);
    this.length += chunk.length;
    const frames = [];
    for (;;) {
      if (this.length < 8) break;
      const buf = this.peek(8);
      const headLen = buf.readUInt32BE(0);
      const bodyLen = buf.readUInt32BE(4);
      if (headLen > MAX_HEADER || bodyLen > MAX_BODY) {
        throw new Error(`implausible frame (${headLen} + ${bodyLen} bytes)`);
      }
      const total = 8 + headLen + bodyLen;
      if (this.length < total) break;
      const frame = this.take(total);
      const header = JSON.parse(frame.toString('utf8', 8, 8 + headLen));
      frames.push({ header, body: frame.subarray(8 + headLen) });
    }
    return frames;
  }

  // Joining on every chunk would copy a large body once per pipe read, so the
  // chunks stay a list until a whole frame is there.
  peek(n) {
    if (this.chunks[0].length >= n) return this.chunks[0];
    this.chunks = [Buffer.concat(this.chunks)];
    return this.chunks[0];
  }

  take(n) {
    const all = this.chunks.length === 1 ? this.chunks[0] : Buffer.concat(this.chunks);
    const frame = all.subarray(0, n);
    const rest = all.subarray(n);
    this.chunks = rest.length ? [rest] : [];
    this.length = rest.length;
    return frame;
  }
}

module.exports = { encode, Decoder };
