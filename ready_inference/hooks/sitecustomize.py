"""Load the inference integration in spawned SGLang workers."""
import os
if os.environ.get('READY_CHECKPOINT'):
    try:
        from ready_inference.adapter import install as install_reader
        from ready_inference.patches import install as install_verifier
        install_reader()
        install_verifier()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(1)
