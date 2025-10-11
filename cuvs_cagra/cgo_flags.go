package cuvs

// #cgo CFLAGS: -I/usr/local/include -I/usr/local/cuda-12/include
// #cgo LDFLAGS: -L/usr/local/lib -L/usr/local/cuda-12/lib64 -lcuvs -lcuvs_c -lcudart -lcublas -lcusolver -lcusparse
import "C"