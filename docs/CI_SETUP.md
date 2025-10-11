# gobed CI Overview (October 2025)

This repository now mirrors the split CI strategy we use in `netwrck/githubagent`:

- **Hosted CPU pipeline (`Hosted CI`)** runs on GitHub-hosted Ubuntu runners. It executes a
  focused gofmt check and a trimmed unit suite via `scripts/ci/run_cpu_checks.sh` to cover the
  packages that compile without GPU dependencies (`github.com/lee101/gobed` and
  `github.com/lee101/gobed/ann/simd`).

- **Self-hosted GPU pipeline (`GPU CI`)** runs on our CUDA 12.9 capable runner and invokes
  `scripts/ci/run_gpu_checks.sh`. The script expects the runner to expose the labels
  `self-hosted`, `gpu`, and `cuda-12.9` and to have the cuVS/CAGRA libraries in the default
  install locations (`/usr/local/cuvs`, `/usr/local/cuda`).

## Self-hosted runner expectations

1. Install CUDA 12.9 and the matching NVIDIA driver.
2. Install cuVS and ensure `libcuvs` and friends are on the library path.
3. Keep `libcagra_wrapper.so`, `libgpu_fast_search.so`, and `libtorch` available in the repository
   checkout (the runner mounts the repo at `~/actions-runner/_work/gobed/gobed`).
4. The runner user must have permissions to access the GPU (`nvidia-smi` should succeed).
5. Optional: run `./build_cagra.sh` after cuVS updates to refresh the wrapper library.
6. Ensure `libcagra_wrapper.so` is built (CI script will call `./build_cagra.sh` if it is missing).

## Local parity

To reproduce the hosted checks locally:

```bash
./scripts/ci/run_cpu_checks.sh
```

For GPU validation (requires CUDA/cuVS on your machine):

```bash
./scripts/ci/run_gpu_checks.sh
```

Both scripts exit non-zero on failure to make debugging easier.
