//go:build cutlass

package pir

import (
	"os"
	"sync"
)

func init() {
	matMulVecServerHook = matMulVecTryCutlassResidentA
}

var (
	residentMu      sync.Mutex
	residentPersist *CutlassPersist
	residentACache  *Matrix
)

// ResetCutlassResidentServerForBench drops GPU-resident A (for benchmarks/tests).
func ResetCutlassResidentServerForBench() {
	residentMu.Lock()
	defer residentMu.Unlock()
	if residentPersist != nil {
		residentPersist.Destroy()
		residentPersist = nil
	}
	residentACache = nil
	resetPackedGPUDBState()
}

// matMulVecTryCutlassResidentA performs Query-shaped A·s on GPU with A resident when:
//
//	SIMPLEPIR_USE_CUTLASS=1, SIMPLEPIR_CUTLASS_RESIDENT_A=1,
//	b is a column vector with b.Rows == a.Cols (exact inner dimension; no padded secret rows).
func matMulVecTryCutlassResidentA(out, a, b *Matrix) bool {
	if os.Getenv("SIMPLEPIR_CUTLASS_RESIDENT_A") != "1" || os.Getenv("SIMPLEPIR_USE_CUTLASS") != "1" {
		return false
	}
	if b.Cols != 1 || b.Rows != a.Cols {
		return false
	}
	if !useCutlassForMatMulVec(a.Rows, a.Cols) {
		return false
	}

	residentMu.Lock()
	defer residentMu.Unlock()

	if residentPersist == nil {
		p, err := NewCutlassPersist()
		if err != nil {
			return false
		}
		residentPersist = p
	}
	if residentACache != a {
		if err := residentPersist.UploadLHS(a); err != nil {
			return false
		}
		residentACache = a
	}
	if err := residentPersist.GemmRHSFromHost(b, out); err != nil {
		return false
	}
	return true
}
