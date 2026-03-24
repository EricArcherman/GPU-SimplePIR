//go:build cuda_e2e

package pir

import (
	"os"
	"strconv"
	"testing"
)

// BenchmarkNaiveGPUVsSimplePIR compares:
//   - CPU: real SimplePIR Answer() on squished DB + real query (packed multiply).
//   - GPU: naive uncompressed pipeline — H=DB·A, q=A·s+e, ans=DB·q — one thread-block GEMM/GEMV.
//     Two GPU modes: full H2D of DB+A every iteration vs UploadDBAndA once then device-resident iterations.
//
// The GPU path is a shape mockup (same L,M,N as pre-squish SimplePIR), not protocol-identical.
//
// Build: cmake -S gpu -B gpu/build ... && cmake --build gpu/build
// Run:  CGO_ENABLED=1 go test -tags cuda_e2e -bench NaiveGPUVsSimplePIR -run ^$ -benchtime 3x
//
// Optional env: LOG_N, D (same as other PIR benchmarks).
func BenchmarkNaiveGPUVsSimplePIR(b *testing.B) {
	numEntries := uint64(1 << 18)
	d := uint64(256)
	if v, err := strconv.Atoi(os.Getenv("LOG_N")); err == nil && v != 0 {
		numEntries = uint64(1 << v)
	}
	if v, err := strconv.Atoi(os.Getenv("D")); err == nil && v != 0 {
		d = uint64(v)
	}

	pi := SimplePIR{}
	p := pi.PickParams(numEntries, d, SEC_PARAM, LOGQ)
	i := uint64(0)
	if i >= p.L*p.M {
		b.Fatalf("index out of range")
	}

	DB := MakeRandomDB(numEntries, d, &p)
	shared := pi.Init(DB.Info, p)

	snap := MatrixNew(DB.Data.Rows, DB.Data.Cols)
	copy(snap.Data, DB.Data.Data)
	snap.Add(p.P / 2)

	serverState, _ := pi.FakeSetup(DB, p)
	_, qMsg := pi.Query(i, shared, p, DB.Info)
	qs := MakeMsgSlice(qMsg)

	A := shared.Data[0]
	L := uint32(snap.Rows)
	M := uint32(snap.Cols)
	Nmat := uint32(A.Cols)
	if uint64(L) != p.L || uint64(M) != p.M || uint64(Nmat) != p.N {
		b.Fatalf("dimension mismatch L,M,N")
	}

	secret := MatrixRand(p.N, 1, p.Logq, 0)
	errVec := MatrixGaussian(p.M, 1)
	deltaRow := uint32(i % p.M)

	ctx, err := NewNaiveGPUCtx(L, M, Nmat)
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.Destroy()

	if _, _, _, _, err := ctx.RunNaiveGPUe2e(snap, A, secret, errVec, deltaRow, p.Delta()); err != nil {
		b.Fatalf("warmup RunNaiveGPUe2e: %v", err)
	}

	b.Run("CPU_SimplePIR_Answer", func(b *testing.B) {
		for k := 0; k < b.N; k++ {
			pi.Answer(DB, qs, serverState, shared, p)
		}
	})

	b.Run("GPU_naive_E2E_H2D_DB_A_each_iter", func(b *testing.B) {
		for k := 0; k < b.N; k++ {
			if _, _, _, _, err := ctx.RunNaiveGPUe2e(snap, A, secret, errVec, deltaRow, p.Delta()); err != nil {
				b.Fatalf("RunNaiveGPUe2e: %v", err)
			}
		}
	})

	b.Run("GPU_naive_E2E_device_resident_DB_A", func(b *testing.B) {
		if err := ctx.UploadDBAndA(snap, A); err != nil {
			b.Fatal(err)
		}
		if _, _, _, _, err := ctx.RunNaiveGPUe2eDeviceResident(secret, errVec, deltaRow, p.Delta()); err != nil {
			b.Fatalf("warmup RunNaiveGPUe2eDeviceResident: %v", err)
		}
		b.ResetTimer()
		for k := 0; k < b.N; k++ {
			if _, _, _, _, err := ctx.RunNaiveGPUe2eDeviceResident(secret, errVec, deltaRow, p.Delta()); err != nil {
				b.Fatalf("RunNaiveGPUe2eDeviceResident: %v", err)
			}
		}
	})

	pi.Reset(DB, p)
	_ = qMsg
}

func TestNaiveGpuE2eSmoke(t *testing.T) {
	N := uint64(1 << 10)
	d := uint64(32)
	pi := SimplePIR{}
	p := pi.PickParams(N, d, SEC_PARAM, LOGQ)
	DB := MakeRandomDB(N, d, &p)
	shared := pi.Init(DB.Info, p)

	snap := MatrixNew(DB.Data.Rows, DB.Data.Cols)
	copy(snap.Data, DB.Data.Data)
	snap.Add(p.P / 2)

	_, _ = pi.FakeSetup(DB, p)
	A := shared.Data[0]

	secret := MatrixRand(p.N, 1, p.Logq, 0)
	errVec := MatrixGaussian(p.M, 1)

	ctx, err := NewNaiveGPUCtx(uint32(snap.Rows), uint32(snap.Cols), uint32(A.Cols))
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Destroy()

	st, qu, an, tot, err := ctx.RunNaiveGPUe2e(snap, A, secret, errVec, 0, p.Delta())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GPU naive E2E (H2D DB+A each call) ms: setup=%.3f query=%.3f answer=%.3f total=%.3f", st, qu, an, tot)

	if err := ctx.UploadDBAndA(snap, A); err != nil {
		t.Fatal(err)
	}
	st2, qu2, an2, tot2, err := ctx.RunNaiveGPUe2eDeviceResident(secret, errVec, 0, p.Delta())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("GPU naive E2E (device-resident DB+A) ms: setup=%.3f query=%.3f answer=%.3f total=%.3f", st2, qu2, an2, tot2)

	pi.Reset(DB, p)
}
