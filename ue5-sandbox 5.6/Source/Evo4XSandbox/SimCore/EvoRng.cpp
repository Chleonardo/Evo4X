#include "EvoRng.h"

// ============================================================================
// Minimal SHA-256 implementation (FIPS 180-4) for deterministic RNG.
// Matches Python's hashlib.sha256 output exactly.
// ============================================================================

namespace Sha256Internal
{
	static const uint32 K[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
		0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
		0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
		0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
		0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
		0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
		0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};

	FORCEINLINE uint32 RotR(uint32 X, uint32 N) { return (X >> N) | (X << (32 - N)); }
	FORCEINLINE uint32 Ch(uint32 X, uint32 Y, uint32 Z) { return (X & Y) ^ (~X & Z); }
	FORCEINLINE uint32 Maj(uint32 X, uint32 Y, uint32 Z) { return (X & Y) ^ (X & Z) ^ (Y & Z); }
	FORCEINLINE uint32 Sig0(uint32 X) { return RotR(X, 2) ^ RotR(X, 13) ^ RotR(X, 22); }
	FORCEINLINE uint32 Sig1(uint32 X) { return RotR(X, 6) ^ RotR(X, 11) ^ RotR(X, 25); }
	FORCEINLINE uint32 Gam0(uint32 X) { return RotR(X, 7) ^ RotR(X, 18) ^ (X >> 3); }
	FORCEINLINE uint32 Gam1(uint32 X) { return RotR(X, 17) ^ RotR(X, 19) ^ (X >> 10); }

	void HashBytes(const uint8* Data, int32 Len, uint8 OutDigest[32])
	{
		uint32 H[8] = {
			0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
			0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
		};

		// Pre-processing: pad message to 64-byte blocks
		const uint64 BitLen = static_cast<uint64>(Len) * 8;
		// Total padded length: Len + 1 (0x80) + padding + 8 (length) rounded up to 64
		const int32 PaddedLen = ((Len + 9 + 63) / 64) * 64;
		TArray<uint8> Padded;
		Padded.SetNumZeroed(PaddedLen);
		FMemory::Memcpy(Padded.GetData(), Data, Len);
		Padded[Len] = 0x80;
		// Append bit length as big-endian 64-bit
		for (int32 i = 0; i < 8; ++i)
		{
			Padded[PaddedLen - 1 - i] = static_cast<uint8>(BitLen >> (i * 8));
		}

		// Process each 64-byte block
		for (int32 Block = 0; Block < PaddedLen; Block += 64)
		{
			uint32 W[64];
			for (int32 i = 0; i < 16; ++i)
			{
				const int32 Off = Block + i * 4;
				W[i] = (static_cast<uint32>(Padded[Off]) << 24)
				     | (static_cast<uint32>(Padded[Off + 1]) << 16)
				     | (static_cast<uint32>(Padded[Off + 2]) << 8)
				     | (static_cast<uint32>(Padded[Off + 3]));
			}
			for (int32 i = 16; i < 64; ++i)
			{
				W[i] = Gam1(W[i - 2]) + W[i - 7] + Gam0(W[i - 15]) + W[i - 16];
			}

			uint32 A = H[0], B = H[1], C = H[2], D = H[3];
			uint32 E = H[4], F = H[5], G = H[6], HH = H[7];

			for (int32 i = 0; i < 64; ++i)
			{
				const uint32 T1 = HH + Sig1(E) + Ch(E, F, G) + K[i] + W[i];
				const uint32 T2 = Sig0(A) + Maj(A, B, C);
				HH = G; G = F; F = E; E = D + T1;
				D = C; C = B; B = A; A = T1 + T2;
			}

			H[0] += A; H[1] += B; H[2] += C; H[3] += D;
			H[4] += E; H[5] += F; H[6] += G; H[7] += HH;
		}

		// Output digest as big-endian bytes
		for (int32 i = 0; i < 8; ++i)
		{
			OutDigest[i * 4 + 0] = static_cast<uint8>(H[i] >> 24);
			OutDigest[i * 4 + 1] = static_cast<uint8>(H[i] >> 16);
			OutDigest[i * 4 + 2] = static_cast<uint8>(H[i] >> 8);
			OutDigest[i * 4 + 3] = static_cast<uint8>(H[i]);
		}
	}
}

// Internal: compute SHA-256 of "{Seed}|{Key}", take first 4 bytes as uint32, divide by 2^32.
static float ComputeRand01(const FString& Combined)
{
	// Convert to UTF-8 (matches Python str encoding)
	const FTCHARToUTF8 Utf8(*Combined);
	const int32 Len = Utf8.Length();

	uint8 Digest[32];
	Sha256Internal::HashBytes(reinterpret_cast<const uint8*>(Utf8.Get()), Len, Digest);

	// Take first 4 bytes as big-endian uint32
	// Matches Python: int(sha256(...).hexdigest()[:8], 16)
	const uint32 Value = (static_cast<uint32>(Digest[0]) << 24)
	                   | (static_cast<uint32>(Digest[1]) << 16)
	                   | (static_cast<uint32>(Digest[2]) << 8)
	                   | (static_cast<uint32>(Digest[3]));

	return static_cast<float>(Value) / 4294967296.0f;
}

float FEvoRng::Rand01(const FString& Seed, const FString& Key)
{
	const FString Combined = FString::Printf(TEXT("%s|%s"), *Seed, *Key);
	return ComputeRand01(Combined);
}

int32 FEvoRng::ChooseIndex(const FString& Seed, const FString& Key, int32 N)
{
	if (N <= 0) return 0;
	const float R = Rand01(Seed, Key);
	return FMath::Min(N - 1, static_cast<int32>(R * N));
}

float FEvoRng::Rand01(const FString& Seed, const FString& Context, int32 A)
{
	const FString Key = FString::Printf(TEXT("%s|%d"), *Context, A);
	return Rand01(Seed, Key);
}

float FEvoRng::Rand01(const FString& Seed, const FString& Context, int32 A, int32 B)
{
	const FString Key = FString::Printf(TEXT("%s|%d|%d"), *Context, A, B);
	return Rand01(Seed, Key);
}

float FEvoRng::Rand01(const FString& Seed, const FString& Context, int32 A, const FString& B)
{
	const FString Key = FString::Printf(TEXT("%s|%d|%s"), *Context, A, *B);
	return Rand01(Seed, Key);
}

int32 FEvoRng::ChooseIndex(const FString& Seed, const FString& Context, int32 A, int32 N)
{
	const FString Key = FString::Printf(TEXT("%s|%d"), *Context, A);
	return ChooseIndex(Seed, Key, N);
}

TArray<int32> FEvoRng::ShuffleIndices(const FString& Seed, const FString& Key, int32 N)
{
	TArray<int32> Indices;
	Indices.Reserve(N);
	for (int32 i = 0; i < N; ++i)
	{
		Indices.Add(i);
	}

	for (int32 i = N - 1; i > 0; --i)
	{
		const FString SwapKey = FString::Printf(TEXT("%s|shuffle|%d"), *Key, i);
		const int32 J = ChooseIndex(Seed, SwapKey, i + 1);
		Indices.Swap(i, J);
	}

	return Indices;
}
