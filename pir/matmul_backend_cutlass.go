//go:build cutlass

package pir

/*
#cgo CFLAGS: -I${SRCDIR}/../gpu
#cgo LDFLAGS: -L${SRCDIR}/../gpu/build -lsimplepir_cutlass -lstdc++ -lcudart
#include "pir.h"
#include "../gpu/cutlass_bridge.h"
*/
import "C"

import (
	"os"
	"strconv"
	"unsafe"
)

// cutlassMinMulOps returns the minimum (aRows*aCols*bCols) for using CUTLASS.
// Default 0 means always try GPU when SIMPLEPIR_USE_CUTLASS=1 (full path).
// Set SIMPLEPIR_CUTLASS_MIN_OPS (e.g. 1000000) to skip small problems and reduce launch overhead.
func cutlassMinMulOps() uint64 {
	s := os.Getenv("SIMPLEPIR_CUTLASS_MIN_OPS")
	if s == "" {
		return 0
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

func useCutlassForGemm(aRows, aCols, bCols C.size_t) bool {
	if os.Getenv("SIMPLEPIR_USE_CUTLASS") != "1" {
		return false
	}
	ops := uint64(aRows) * uint64(aCols) * uint64(bCols)
	return ops >= cutlassMinMulOps()
}

// useCutlassForMatMulVec reports whether the cutlass path applies to (aRows×aCols)·(aCols×1).
func useCutlassForMatMulVec(aRows, aCols uint64) bool {
	return useCutlassForGemm(C.size_t(aRows), C.size_t(aCols), 1)
}

func matMulBackend(outPtr, aPtr, bPtr *C.Elem, aRows, aCols, bCols C.size_t) {
	if !useCutlassForGemm(aRows, aCols, bCols) {
		C.matMul(outPtr, aPtr, bPtr, aRows, aCols, bCols)
		return
	}

	status := C.simplepir_matmul_cutlass_u32(
		(*C.uint32_t)(unsafe.Pointer(outPtr)),
		(*C.uint32_t)(unsafe.Pointer(aPtr)),
		(*C.uint32_t)(unsafe.Pointer(bPtr)),
		aRows,
		aCols,
		bCols,
	)
	if status != 0 {
		C.matMul(outPtr, aPtr, bPtr, aRows, aCols, bCols)
	}
}

// MatrixMulVec is (aRows x aCols) * (aCols x 1) using the first aCols entries of b.
func matMulVecBackend(outPtr, aPtr, bPtr *C.Elem, aRows, aCols C.size_t) {
	if !useCutlassForGemm(aRows, aCols, 1) {
		C.matMulVec(outPtr, aPtr, bPtr, aRows, aCols)
		return
	}

	status := C.simplepir_matmul_cutlass_u32(
		(*C.uint32_t)(unsafe.Pointer(outPtr)),
		(*C.uint32_t)(unsafe.Pointer(aPtr)),
		(*C.uint32_t)(unsafe.Pointer(bPtr)),
		aRows,
		aCols,
		1,
	)
	if status != 0 {
		C.matMulVec(outPtr, aPtr, bPtr, aRows, aCols)
	}
}
