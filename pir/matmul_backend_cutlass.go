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
	"unsafe"
)

func matMulBackend(outPtr, aPtr, bPtr *C.Elem, aRows, aCols, bCols C.size_t) {
	if os.Getenv("SIMPLEPIR_USE_CUTLASS") != "1" {
		C.matMul(outPtr, aPtr, bPtr, aRows, aCols, bCols)
		return
	}

	ops := uint64(aRows) * uint64(aCols) * uint64(bCols)
	if ops < 1_000_000 {
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
