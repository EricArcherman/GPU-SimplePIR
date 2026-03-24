//go:build cutlass

package pir

/*
#cgo CFLAGS: -I${SRCDIR}/../gpu
#cgo LDFLAGS: -L${SRCDIR}/../gpu/build -lsimplepir_cutlass -lstdc++ -lcudart
#include "matmul_packed_gpu.h"
*/
import "C"

import (
	"os"
	"sync"
	"unsafe"
)

func init() {
	matMulVecPackedServerHook = matMulVecPackedTryGPU
	packedGPUDBSyncHook = syncPackedGPUDBIfEnabled
	packedGPUDBClearHook = clearPackedGPUDB
}

var (
	packedDBMu   sync.Mutex
	packedDBHost *Matrix // full squished DB.Data after successful upload; nil if disabled or failed
)

// resetPackedGPUDBState frees device squished DB and clears Go metadata (e.g. between benchmarks).
func resetPackedGPUDBState() {
	packedDBMu.Lock()
	defer packedDBMu.Unlock()
	C.simplepir_packed_db_release()
	packedDBHost = nil
}

func syncPackedGPUDBIfEnabled(DB *Database) {
	if os.Getenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT") != "1" {
		return
	}
	if DB == nil || DB.Data == nil || len(DB.Data.Data) == 0 {
		return
	}
	d := DB.Data
	packedDBMu.Lock()
	defer packedDBMu.Unlock()
	rc := C.simplepir_packed_db_upload(
		(*C.uint32_t)(unsafe.Pointer(&d.Data[0])),
		C.size_t(d.Rows),
		C.size_t(d.Cols),
	)
	if rc != 0 {
		C.simplepir_packed_db_release()
		packedDBHost = nil
		return
	}
	packedDBHost = d
}

func clearPackedGPUDB() {
	packedDBMu.Lock()
	defer packedDBMu.Unlock()
	C.simplepir_packed_db_release()
	packedDBHost = nil
}

// packedSubmatrixRowOffset returns the row index of a within db when a.Data is a contiguous
// subslice of db.Data with matching Cols (SimplePIR Answer uses SelectRows).
func packedSubmatrixRowOffset(db, a *Matrix) (uint64, bool) {
	if db == nil || a == nil || a.Cols != db.Cols {
		return 0, false
	}
	if len(db.Data) == 0 || len(a.Data) == 0 {
		return 0, false
	}
	if uint64(len(db.Data)) != db.Rows*db.Cols || uint64(len(a.Data)) != a.Rows*a.Cols {
		return 0, false
	}
	elem := unsafe.Sizeof(db.Data[0])
	dbBase := uintptr(unsafe.Pointer(&db.Data[0]))
	aBase := uintptr(unsafe.Pointer(&a.Data[0]))
	aEnd := aBase + uintptr(len(a.Data))*elem
	dbEnd := dbBase + uintptr(len(db.Data))*elem
	if aBase < dbBase || aEnd > dbEnd {
		return 0, false
	}
	diff := aBase - dbBase
	rowStride := uintptr(db.Cols) * elem
	if rowStride == 0 || diff%rowStride != 0 {
		return 0, false
	}
	row := uint64(diff / rowStride)
	if row+a.Rows > db.Rows {
		return 0, false
	}
	return row, true
}

// matMulVecPackedTryGPU runs SimplePIR Answer packed multiply on CUDA when SIMPLEPIR_GPU_PACKED_ANSWER=1.
// Requires pir.c layout (basis=10, compression=3).
// When SIMPLEPIR_GPU_PACKED_DB_RESIDENT=1 and matrix a is a row slice of the synced DB, skips H2D for a.
func matMulVecPackedTryGPU(out, a, b *Matrix, basis, compression uint64) bool {
	if os.Getenv("SIMPLEPIR_GPU_PACKED_ANSWER") != "1" {
		return false
	}
	if basis != 10 || compression != 3 {
		return false
	}

	if os.Getenv("SIMPLEPIR_GPU_PACKED_DB_RESIDENT") == "1" {
		packedDBMu.Lock()
		ph := packedDBHost
		row, ok := packedSubmatrixRowOffset(ph, a)
		packedDBMu.Unlock()
		if ok {
			rc := C.simplepir_matmul_vec_packed_gpu_resident(
				(*C.uint32_t)(unsafe.Pointer(&out.Data[0])),
				C.size_t(row),
				C.size_t(a.Rows),
				C.size_t(a.Cols),
				(*C.uint32_t)(unsafe.Pointer(&b.Data[0])),
			)
			if rc == 0 {
				return true
			}
		}
	}

	rc := C.simplepir_matmul_vec_packed_gpu(
		(*C.uint32_t)(unsafe.Pointer(&out.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&a.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&b.Data[0])),
		C.size_t(a.Rows),
		C.size_t(a.Cols),
	)
	return rc == 0
}
