# DeepV Performance Benchmark Report

## Benchmark Results

### Rendering 64X64 Patch

- **Mean**: 0.0062s
- **Std**: 0.0168s
- **Min**: 0.0000s
- **Max**: 0.0564s
- **Iterations**: 10

### Merging 100 Lines

- **Mean**: 0.0024s
- **Std**: 0.0005s
- **Min**: 0.0020s
- **Max**: 0.0030s
- **Iterations**: 5

### Refinement Import

- **Mean**: 1.5184s
- **Std**: 2.1473s
- **Min**: 0.0000s
- **Max**: 4.5552s
- **Iterations**: 3

### Image Preprocessing 256X256

- **Mean**: 0.0086s
- **Std**: 0.0246s
- **Min**: 0.0000s
- **Max**: 0.0824s
- **Iterations**: 10

## Performance Targets

- Rendering (64x64 patch): <10ms PASS
- Merging (100 lines): <1s PASS
- Refinement import: <0.1s FAIL (needs optimization)
- Image preprocessing (256x256): <0.01s PASS

