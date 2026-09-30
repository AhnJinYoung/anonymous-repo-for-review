"""The active backend. Set once per process by `runners.setup()`; every module reads `state.B`."""
B = None
def set_backend(b):
    global B; B = b; return b
