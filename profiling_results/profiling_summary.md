# DeepV Pipeline Performance Profile Summary

**Generated:** 2026-02-06 15:33:49

## Recommendations

### Immediate Actions
- Review `cprofile_results.txt` for top time-consuming functions
- Analyze `pytorch_profile.json` in Chrome tracing for detailed GPU/CPU breakdown
- Check `memory_profile.txt` for memory optimization opportunities

### Optimization Targets
- Target: < 2 seconds per 64×64 patch (current Bézier splatting: 0.0064s OK)
- Memory: < 8GB peak usage for 128×128 images
- CPU/GPU balance: Minimize data transfers between devices

### Files Generated
- `cprofile_results.txt` - Python function timing analysis
- `pytorch_profile.json` - Detailed GPU/CPU profiling (open in Chrome)
- `pytorch_summary.txt` - PyTorch profiler summary tables
- `memory_profile.txt` - Memory usage statistics
