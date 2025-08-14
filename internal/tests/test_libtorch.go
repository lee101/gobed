package main

import (
	"fmt"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

func main() {
	fmt.Println("Testing libtorch integration...")

	// Check if CUDA is available
	if gotch.CudaIsAvailable() {
		fmt.Printf("CUDA is available with %d devices\n", gotch.CudaDeviceCount())
	} else {
		fmt.Println("CUDA not available, using CPU")
	}

	// Try to load the PyTorch model
	modelPath := "model/production_pytorch_full_model.pt"
	fmt.Printf("Attempting to load model from: %s\n", modelPath)

	// Create a device (CPU for now)
	device := gotch.CPU

	// Try to load the model
	vs := ts.NewVarStore(device)

	// For now, let's just test basic tensor operations
	fmt.Println("Testing basic tensor operations...")

	// Create a sample tensor
	tensor := ts.MustRandint(int64(0), int64(30522), []int64{1, 8}, gotch.Int64, device)
	fmt.Printf("Created tensor with shape: %v\n", tensor.MustSize())

	// Print tensor values
	vals := tensor.Int64Values()
	fmt.Printf("Tensor values: %v\n", vals[:5]) // Show first 5 values

	// Clean up
	tensor.MustDrop()

	fmt.Println("Libtorch basic test completed!")
}
