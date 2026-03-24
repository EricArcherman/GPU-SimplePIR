//go:build cuda_e2e

package pir

/*
#cgo CFLAGS: -I${SRCDIR}/../gpu
#cgo LDFLAGS: -L${SRCDIR}/../gpu/build -lsimplepir_naive -lstdc++ -lcudart
#include "naive_simplepir.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// NaiveGPUCtx holds device buffers for the uncompressed SimplePIR-shaped mock pipeline.
type NaiveGPUCtx struct {
	h C.NaiveGpuSimplepirHandle
}

func NewNaiveGPUCtx(L, M, N uint32) (*NaiveGPUCtx, error) {
	p := C.naive_gpu_ctx_create(C.uint32_t(L), C.uint32_t(M), C.uint32_t(N))
	if p == nil {
		return nil, fmt.Errorf("naive_gpu_ctx_create failed")
	}
	return &NaiveGPUCtx{h: p}, nil
}

func (c *NaiveGPUCtx) Destroy() {
	if c == nil || c.h == nil {
		return
	}
	C.naive_gpu_ctx_destroy(c.h)
	c.h = nil
}

// UploadDBAndA copies uncompressed DB and A to the device once (for device-resident iterations).
func (c *NaiveGPUCtx) UploadDBAndA(db, A *Matrix) error {
	if c == nil || c.h == nil {
		return fmt.Errorf("nil context")
	}
	if len(db.Data) == 0 || len(A.Data) == 0 {
		return fmt.Errorf("empty matrix")
	}
	rc := C.naive_gpu_ctx_upload_db_A(
		c.h,
		(*C.uint32_t)(unsafe.Pointer(&db.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&A.Data[0])),
	)
	if rc != 0 {
		return fmt.Errorf("naive_gpu_ctx_upload_db_A failed rc=%d", rc)
	}
	return nil
}

// RunNaiveGPUe2e runs setup (H=DB·A), query (q=A·s+e plus delta), answer (ans=DB·q). Returns phase times in ms.
func (c *NaiveGPUCtx) RunNaiveGPUe2e(db, A, secret, err *Matrix, deltaRow uint32, delta uint64) (setupMS, queryMS, answerMS, totalMS float64, errOut error) {
	if c == nil || c.h == nil {
		return 0, 0, 0, 0, fmt.Errorf("nil context")
	}
	if len(db.Data) == 0 || len(A.Data) == 0 || len(secret.Data) == 0 || len(err.Data) == 0 {
		return 0, 0, 0, 0, fmt.Errorf("empty matrix")
	}
	var st, qu, an, tot C.double
	rc := C.naive_gpu_ctx_run_e2e(
		c.h,
		(*C.uint32_t)(unsafe.Pointer(&db.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&A.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&secret.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&err.Data[0])),
		C.uint32_t(deltaRow),
		C.uint32_t(delta),
		&st, &qu, &an, &tot,
	)
	if rc != 0 {
		return 0, 0, 0, 0, fmt.Errorf("naive_gpu_ctx_run_e2e failed rc=%d", rc)
	}
	return float64(st), float64(qu), float64(an), float64(tot), nil
}

// RunNaiveGPUe2eDeviceResident is like RunNaiveGPUe2e but assumes UploadDBAndA was already called:
// no H2D for DB or A; each call still uploads secret and error vector, runs kernels, and reads one output word.
func (c *NaiveGPUCtx) RunNaiveGPUe2eDeviceResident(secret, err *Matrix, deltaRow uint32, delta uint64) (setupMS, queryMS, answerMS, totalMS float64, errOut error) {
	if c == nil || c.h == nil {
		return 0, 0, 0, 0, fmt.Errorf("nil context")
	}
	if len(secret.Data) == 0 || len(err.Data) == 0 {
		return 0, 0, 0, 0, fmt.Errorf("empty matrix")
	}
	var st, qu, an, tot C.double
	rc := C.naive_gpu_ctx_run_e2e_device_resident(
		c.h,
		(*C.uint32_t)(unsafe.Pointer(&secret.Data[0])),
		(*C.uint32_t)(unsafe.Pointer(&err.Data[0])),
		C.uint32_t(deltaRow),
		C.uint32_t(delta),
		&st, &qu, &an, &tot,
	)
	if rc != 0 {
		return 0, 0, 0, 0, fmt.Errorf("naive_gpu_ctx_run_e2e_device_resident failed rc=%d", rc)
	}
	return float64(st), float64(qu), float64(an), float64(tot), nil
}
