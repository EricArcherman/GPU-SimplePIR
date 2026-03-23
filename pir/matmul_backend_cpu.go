//go:build !cutlass

package pir

// #include "pir.h"
import "C"

func matMulBackend(outPtr, aPtr, bPtr *C.Elem, aRows, aCols, bCols C.size_t) {
	C.matMul(outPtr, aPtr, bPtr, aRows, aCols, bCols)
}
