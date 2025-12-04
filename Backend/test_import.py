try:
    from main import app
    print('[OK] Main module loaded successfully')
    print('[OK] FastAPI app instance created')
except Exception as e:
    print(f'[ERROR] Failed to load main: {e}')
    import traceback
    traceback.print_exc()
