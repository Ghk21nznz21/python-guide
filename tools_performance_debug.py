import tracemalloc
import time 
import cProfile
import traceback


def monotorize(func):
	def wrapper(*args, **kwargs):
		start = time.time()
		tracemalloc.start()
		previous, previous_peak = tracemalloc.get_traced_memory()
		result = func(*args, **kwargs)
		current, peak = tracemalloc.get_traced_memory()
		end = time.time()
		print( f'{func.__name__} -> current_mem: {current-previous} mem_peak: {peak-previous_peak} and took {end-start} seconds')
		return result
	return wrapper
		
def monotorize2(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        profiler.print_stats(sort='cumulative')
        return result
    return wrapper

def monotorize3(func):
    def wrapper(*args, **kwargs):
        code = func(*args, **kwargs)
        exec_start = time.time()
        exec_result = None
        try:
            exec_result = eval(code)
        except Exception as e:
            print(f'Error while executing code: {e}')
        exec_end = time.time()
        print(f'{func.__name__} -> Code execution took {exec_end - exec_start} seconds')
        return exec_result
    return wrapper


def debbug(y):
    try: 
        y = 0
    except Exception as e:
        print('The type of error is', e.__class__.__name__)
        print('The error details are', e)
        traceback.print_exc()
    return