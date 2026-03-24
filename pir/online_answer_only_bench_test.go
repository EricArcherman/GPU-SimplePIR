//go:build cutlass

package pir

import (
	"os"
	"strconv"
	"testing"
)

// BenchmarkSimplePIRPackedAnswerOnlineOnly measures only the SimplePIR **Answer** core:
// packed multiply squished_D · Q → answer vector (one query, full DB rows).
//
// **Not included in the timed loop:** picking params, allocating the DB, squishing, uploading D
// to the GPU (when resident), or building the client query — those mirror “offline / preload”.
//
// **Included per iteration:** whatever MatrixMulVecPacked does on that path:
//   - **CPU_packed_D_host:** C packed matvec on host (D and Q in RAM).
//   - **GPU_packed_upload_D_each_iter:** H2D for full D slice + Q, kernel, D2H result (no resident D).
//   - **GPU_packed_resident_D:** H2D for Q (+ out); kernel reads D already on device; D2H result.
//
// This uses the **custom packed CUDA kernel** in gpu/matmul_packed_gpu.cu. It is **not** CUTLASS.
// CUTLASS is only used for dense 32-bit GEMMs (e.g. Query-shaped A·s), a different operation.
//
//	LOG_N=18 D=256 go test -tags cutlass -bench BenchmarkSimplePIRPackedAnswerOnlineOnly -run '^$' -benchtime 20x
func BenchmarkSimplePIRPackedAnswerOnlineOnly(b *testing.B) {
	numEntries := uint64(1 << 16)
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
		b.Fatalf("bad index")
	}

	run := func(name string, gpuPacked, residentD bool) {
		b.Run(name, func(b *testing.B) {
			if gpuPacked {
				_ = os.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "1")
			} else {
				_ = os.Setenv("SIMPLEPIR_GPU_PACKED_ANSWER", "0")
			}
			if residentD {
				_ = os.Setenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT", "1")
			} else {
				_ = os.Setenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT", "0")
			}
			ResetCutlassResidentServerForBench()

			DB := MakeRandomDB(numEntries, d, &p)
			shared := pi.Init(DB.Info, p)
			_, _ = pi.FakeSetup(DB, p)
			_, qMsg := pi.Query(i, shared, p, DB.Info)
			q := qMsg.Data[0]

			b.ResetTimer()
			for k := 0; k < b.N; k++ {
				_ = MatrixMulVecPacked(DB.Data, q, DB.Info.Basis, DB.Info.Squishing)
			}
			b.StopTimer()
			pi.Reset(DB, p)
		})
	}

	run("CPU_packed_D_host", false, false)
	run("GPU_packed_upload_D_each_iter", true, false)
	run("GPU_packed_resident_D", true, true)
}
