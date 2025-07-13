# ONNX Runtime Integration Guide for GoBED

## Executive Summary

Your Go binding `github.com/yalue/onnxruntime_go` is compiled against ONNX Runtime (ORT) 1.22, whose C-API major version is 22. The `.so` you have on disk is ORT 1.20 (API ≤ 20), so the loader aborts with:

```
The requested API version [22] is not available …
```

**Matching the Go wrapper ↔ ORT shared library ↔ CUDA/cuDNN toolchain versions fixes the error.**

---

## 1. Which ORT release goes with which Go tag?

| onnxruntime_go tag | Requires libonnxruntime.so | CUDA/cuDNN required         |
|--------------------|---------------------------|-----------------------------|
| v1.22.x (current)  | 1.22.* (API 22)           | CUDA 12.x + cuDNN 9.x       |
| v1.20.x            | 1.20.* (API 20)           | CUDA 12.x                   |
| v1.19.x            | 1.19.* (API 19)           | CUDA 11.8                   |
| ≤ v1.11.x          | 1.14 – 1.18               | CUDA 11.8 or none           |

If you do not need GPU, the CPU-only binaries work with the same version numbers.

---

## 2. Pick CPU or GPU

### 2.1 CPU-only build
- Download `onnxruntime-linux-x64-1.22.0.tgz` (or the matching archive for your OS) from the [GitHub releases page](https://github.com/microsoft/onnxruntime/releases) and drop `libonnxruntime.so.1.22.0` somewhere on the runtime `LD_LIBRARY_PATH`.

### 2.2 CUDA (GPU) build
- Install CUDA 12.x and cuDNN 9.x that match ORT 1.22’s requirements.
- Download `onnxruntime-linux-x64-gpu-1.22.0.tgz` (contains `libonnxruntime.so` plus `libonnxruntime_providers_cuda.so`).
- Ensure all CUDA libraries are on `LD_LIBRARY_PATH` before you start Go. Missing `libcudnn*.so` is the most frequent cause of `Failed to create CUDAExecutionProvider` errors.

---

## 3. Go module wiring

```bash
# CPU, pinned to ORT 1.22
go get github.com/yalue/onnxruntime_go@v1.22.0

# If you stay on ORT 1.20 instead:
go get github.com/yalue/onnxruntime_go@v1.20.1
```

If you override a transiently older version, add a `replace` line in `go.mod`:

```
replace github.com/yalue/onnxruntime_go => github.com/yalue/onnxruntime_go v1.20.1
```

---

## 4. Boot-strapping ORT from Go

```go
import (
    ort "github.com/yalue/onnxruntime_go"
)

func main() {
    // 1. Tell the wrapper where the exact .so lives
    ort.SetSharedLibraryPath("/opt/onnxruntime/lib/libonnxruntime.so.1.22.0")

    // 2. Initialise the runtime
    if err := ort.InitializeEnvironment(); err != nil {
        log.Fatalf("ORT init failed: %v", err)
    }
    defer ort.DestroyEnvironment()

    // 3. Session options: choose CPU or GPU
    opts, _ := ort.NewSessionOptions()
    // --- GPU example (device 0) -------------------------------
    cuda := ort.NewCUDAProviderOptions()          // helper ctor
    cuda.DeviceID = 0
    if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
        log.Fatalf("CUDA not available: %v", err)
    }
    // ----------------------------------------------------------

    sess, _ := ort.NewAdvancedSession("model.onnx",
        []string{"input"}, []string{"output"}, nil, nil, opts)
    defer sess.Destroy()

    // run as usual ...
}
```

**Notes:**
- The binding exposes helpers for TensorRT, CoreML, DirectML, OpenVINO, etc. in the same style if you need multi-EP builds.
- If you prefer the simpler Python-style API you can call `sessOptions := ort.SessionOptions{}` and then `sessOptions.SetProviders()` exactly like the C & Python docs show.

---

## 5. Troubleshooting checklist

| Symptom                                      | Most common root-cause         | Fix                                                        |
|-----------------------------------------------|-------------------------------|------------------------------------------------------------|
| OrtGetApiBase / API version not available     | Library/headers mismatch       | Ensure binding tag ↔ ORT .so match (table §1).             |
| CUDAExecutionProvider not in provider list    | Using CPU build of ORT         | Download -gpu archive or build from source with --use_cuda. |
| Very slow GPU run                            | TF32 disabled or tensor copy   | Use {"use_tf32": 1} in provider options or profile I/O.    |
| Seg-fault on startup                         | Missing libcudnn* / libnvrtc*  | Confirm new dependencies are on LD_LIBRARY_PATH.            |
| Works on CPU, hangs on GPU                   | Driver / CUDA minor version    | Verify nvidia-smi driver ≥ CUDA runtime, rebuild if needed. |

---

## 6. What to add to the repo

- `docs/onnx_runtime.md` – copy this guide.
- CI job that downloads the exact `libonnxruntime.so` during build, keyed off `ONNX_VERSION` in `.github/workflows/build.yml`.
- A one-liner in `README.md`:

    > If the program crashes with “requested API version”, see `docs/onnx_runtime.md`.

That should make the next person’s life a lot easier.
