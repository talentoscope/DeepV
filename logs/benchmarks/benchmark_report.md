# DeepV Performance Benchmark Report

## Benchmark Results

### Rendering 64X64 Patch

- **Mean**: 0.0483s
- **Std**: 0.1422s
- **Min**: 0.0000s
- **Max**: 0.4749s
- **Iterations**: 10

### Merging 100 Lines

- **Mean**: 0.0020s
- **Std**: 0.0000s
- **Min**: 0.0020s
- **Max**: 0.0021s
- **Iterations**: 5

### Refinement Import

- **Mean**: 3.3665s
- **Std**: 4.7610s
- **Min**: 0.0000s
- **Max**: 10.0996s
- **Iterations**: 3

### Image Preprocessing 256X256

- **Mean**: 0.0091s
- **Std**: 0.0253s
- **Min**: 0.0000s
- **Max**: 0.0851s
- **Iterations**: 10

## Performance Targets

- Rendering (64x64 patch): <10ms PASS
- Merging (100 lines): <1s PASS
- Refinement import: <0.1s FAIL (needs optimization)
- Image preprocessing (256x256): <0.01s PASS

