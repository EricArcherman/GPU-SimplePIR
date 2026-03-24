//go:build cutlass

package pir

/*
#cgo CFLAGS: -I${SRCDIR}/../gpu
#cgo LDFLAGS: -L${SRCDIR}/../gpu/build -lsimplepir_cutlass -lstdc++ -lcudart
#include "cutlass_bridge.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// CutlassPersist keeps a LHS matrix (or A+B pair) on the GPU for repeated CUTLASS GEMMs.
type CutlassPersist struct {
	h C.SimplepirCutlassPersistHandle
}

func NewCutlassPersist() (*CutlassPersist, error) {
	p := C.simplepir_cutlass_persist_create()
	if p == nil {
		return nil, fmt.Errorf("simplepir_cutlass_persist_create failed")
	}
	return &CutlassPersist{h: p}, nil
}

func (c *CutlassPersist) Destroy() {
	if c == nil || c.h == nil {
		return
	}
	C.simplepir_cutlass_persist_destroy(c.h)
	c.h = nil
}

// UploadLHS stores A (rows×cols) on device for Query-shaped GEMMs: out = A * B with B from host each call.
func (c *CutlassPersist) UploadLHS(A *Matrix) error {
	if c == nil || c.h == nil {
		return fmt.Errorf("nil persist")
	}
	if len(A.Data) == 0 {
		return fmt.Errorf("empty A")
	}
	rc := C.simplepir_cutlass_persist_upload_lhs(c.h, (*C.uint32_t)(unsafe.Pointer(&A.Data[0])),
		C.size_t(A.Rows), C.size_t(A.Cols))
	if rc != 0 {
		return fmt.Errorf("simplepir_cutlass_persist_upload_lhs rc=%d", rc)
	}
	return nil
}

// GemmRHSFromHost computes out = A * B where A is resident; B is (A.Cols × B.Cols), out is (A.Rows × B.Cols).
func (c *CutlassPersist) GemmRHSFromHost(B, out *Matrix) error {
	if c == nil || c.h == nil {
		return fmt.Errorf("nil persist")
	}
	if len(B.Data) == 0 || len(out.Data) == 0 {
		return fmt.Errorf("empty matrix")
	}
	rc := C.simplepir_cutlass_persist_gemm_rhs_from_host(c.h, (*C.uint32_t)(unsafe.Pointer(&B.Data[0])),
		C.size_t(B.Cols), (*C.uint32_t)(unsafe.Pointer(&out.Data[0])))
	if rc != 0 {
		return fmt.Errorf("simplepir_cutlass_persist_gemm_rhs_from_host rc=%d", rc)
	}
	return nil
}

// UploadAB stores DB (L×M) and A (M×N) for Setup-shaped GEMMs: H = DB * A.
func (c *CutlassPersist) UploadAB(DB, A *Matrix) error {
	if c == nil || c.h == nil {
		return fmt.Errorf("nil persist")
	}
	rc := C.simplepir_cutlass_persist_upload_ab(c.h,
		(*C.uint32_t)(unsafe.Pointer(&DB.Data[0])), C.size_t(DB.Rows), C.size_t(DB.Cols),
		(*C.uint32_t)(unsafe.Pointer(&A.Data[0])), C.size_t(A.Cols))
	if rc != 0 {
		return fmt.Errorf("simplepir_cutlass_persist_upload_ab rc=%d", rc)
	}
	return nil
}

// GemmABResident computes H = DB * A with both on device (must match last UploadAB). H is L×N.
func (c *CutlassPersist) GemmABResident(L, M, N uint64, outH *Matrix) error {
	if c == nil || c.h == nil {
		return fmt.Errorf("nil persist")
	}
	rc := C.simplepir_cutlass_persist_gemm_ab_resident(c.h, (*C.uint32_t)(unsafe.Pointer(&outH.Data[0])),
		C.size_t(L), C.size_t(M), C.size_t(N))
	if rc != 0 {
		return fmt.Errorf("simplepir_cutlass_persist_gemm_ab_resident rc=%d", rc)
	}
	return nil
}

// CutlassMatmulHostFull runs one GEMM with H2D for both factors (same as matmul backend path).
func CutlassMatmulHostFull(out, a, b *Matrix) error {
	if len(out.Data) == 0 || len(a.Data) == 0 || len(b.Data) == 0 {
		return fmt.Errorf("empty matrix")
	}
	rc := C.simplepir_matmul_cutlass_u32(
		(*C.uint32_t)(unsafe.Pointer(&out.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&a.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&b.Data[0])),
		C.size_t(a.Rows), C.size_t(a.Cols), C.size_t(b.Cols),
	)
	if rc != 0 {
		return fmt.Errorf("simplepir_matmul_cutlass_u32 rc=%d", rc)
	}
	return nil
}
