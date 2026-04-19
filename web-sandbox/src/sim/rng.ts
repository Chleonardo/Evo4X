// SHA-256 (FIPS 180-4) — exact port of EvoRng.cpp, matches Python hashlib.sha256.

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotR(x: number, n: number): number {
  return ((x >>> n) | (x << (32 - n))) >>> 0;
}

function sha256FirstUint32(input: string): number {
  // Encode as UTF-8 bytes
  const bytes: number[] = [];
  for (let i = 0; i < input.length; i++) {
    const c = input.charCodeAt(i);
    if (c < 0x80) {
      bytes.push(c);
    } else if (c < 0x800) {
      bytes.push(0xc0 | (c >> 6), 0x80 | (c & 0x3f));
    } else {
      bytes.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 0x3f), 0x80 | (c & 0x3f));
    }
  }

  const len = bytes.length;
  const bitLen = len * 8;
  bytes.push(0x80);
  while ((bytes.length % 64) !== 56) bytes.push(0);
  // 64-bit big-endian bit length (upper 32 bits always 0 for our short keys)
  bytes.push(0, 0, 0, 0, (bitLen >>> 24) & 0xff, (bitLen >>> 16) & 0xff, (bitLen >>> 8) & 0xff, bitLen & 0xff);

  let h0 = 0x6a09e667 >>> 0, h1 = 0xbb67ae85 >>> 0, h2 = 0x3c6ef372 >>> 0, h3 = 0xa54ff53a >>> 0;
  let h4 = 0x510e527f >>> 0, h5 = 0x9b05688c >>> 0, h6 = 0x1f83d9ab >>> 0, h7 = 0x5be0cd19 >>> 0;

  const W = new Uint32Array(64);

  for (let blk = 0; blk < bytes.length; blk += 64) {
    for (let i = 0; i < 16; i++) {
      W[i] = ((bytes[blk + i * 4] << 24) | (bytes[blk + i * 4 + 1] << 16) |
               (bytes[blk + i * 4 + 2] << 8) | bytes[blk + i * 4 + 3]) >>> 0;
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotR(W[i - 15], 7) ^ rotR(W[i - 15], 18) ^ (W[i - 15] >>> 3);
      const s1 = rotR(W[i - 2], 17) ^ rotR(W[i - 2], 19) ^ (W[i - 2] >>> 10);
      W[i] = (W[i - 16] + s0 + W[i - 7] + s1) >>> 0;
    }

    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = rotR(e, 6) ^ rotR(e, 11) ^ rotR(e, 25);
      const ch = ((e & f) ^ (~e & g)) >>> 0;
      const t1  = (h + S1 + ch + K[i] + W[i]) >>> 0;
      const S0  = rotR(a, 2) ^ rotR(a, 13) ^ rotR(a, 22);
      const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const t2  = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0;
      d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }

    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0; h5 = (h5 + f) >>> 0; h6 = (h6 + g) >>> 0; h7 = (h7 + h) >>> 0;
  }

  return h0; // first 4 bytes as big-endian uint32
}

export function rand01(seed: string, key: string): number {
  return sha256FirstUint32(`${seed}|${key}`) / 4294967296;
}

export function chooseIndex(seed: string, key: string, n: number): number {
  if (n <= 0) return 0;
  return Math.min(n - 1, Math.floor(rand01(seed, key) * n));
}

export function shuffleIndices(seed: string, key: string, n: number): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = chooseIndex(seed, `${key}|shuffle|${i}`, i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
