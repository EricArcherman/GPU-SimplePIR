// Print SimplePIR packed-Answer matrix shapes for given LOG_N list (D from env or 256).
//
//	go run ./eval/dimensions 16 18 20 22
//	D=512 go run ./eval/dimensions 20
package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/ahenzinger/simplepir/pir"
)

const secParam = uint64(1 << 10)
const logq = uint64(32)

func main() {
	d := uint64(256)
	if v := os.Getenv("D"); v != "" {
		if x, err := strconv.ParseUint(v, 10, 64); err == nil {
			d = x
		}
	}
	args := os.Args[1:]
	if len(args) == 0 {
		args = []string{"18", "20", "22"}
	}
	fmt.Println("LOG_N\tD\tL\tM\tD_squished_rows\tD_squished_cols\tQ_elems_(padded)")
	for _, a := range args {
		ln, err := strconv.Atoi(a)
		if err != nil || ln <= 0 {
			continue
		}
		N := uint64(1) << ln
		pi := pir.SimplePIR{}
		p := pi.PickParams(N, d, secParam, logq)
		l, m := p.L, p.M
		msq := (m + 3 - 1) / 3
		qpad := m
		if m%3 != 0 {
			qpad += 3 - (m % 3)
		}
		fmt.Printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", ln, d, l, m, l, msq, qpad)
	}
}
