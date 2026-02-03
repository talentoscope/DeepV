def test_run_pipeline_import():
    import importlib
    rp = importlib.import_module('run_pipeline')
    assert hasattr(rp, 'main')

def test_cleaning_entry_import():
    import importlib
    mod = importlib.import_module('cleaning.scripts.main_cleaning')
    assert hasattr(mod, 'main') or hasattr(mod, 'parse_args')
