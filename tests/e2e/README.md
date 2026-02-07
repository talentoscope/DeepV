Run the basic tracer test

Install test deps (use your environment's pip):

```bash
python -m pip install -r requirements-dev.txt
pytest tests/e2e/test_trace_basic.py -q
```

This test validates that `analysis/tracing.py` can write per-patch images, model outputs, and pre/post refinement JSON artifacts.
