import concurrent.futures
import requests
import threading
import time

# Local, declares each theread independent from each other with its own requests.Session() and no data sharing
thread_local = threading.local()


def get_session():
    ''' Setting a new session for each thread '''
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session() 		
    return thread_local.session


def download_info(url):
    session = get_session()
    with session.get(url) as response:
        print(f"Read {len(response.content)} from {url}")


def download_all_info(sites):
    ''' Run maximum of 5 threads at the same time '''
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_info, sites)

    #POOL = pool of threads
    #EXECUTER = control how and when run each thread

if __name__ == "__main__":
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    download_all_info(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")