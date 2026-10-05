import asyncio
import time
import aiohttp


async def download_site(session, url):
    async with session.get(url) as response:
        print("Read {0} from {1}".format(response.content_length, url))


async def download_all_sites(sites):
    async with aiohttp.ClientSession() as session:
        #The tasks can share the session because they are all running on the same thread.
		
        tasks = []
        for url in sites:
            task = asyncio.ensure_future(download_site(session, url))
            #creates a list of tasks using asyncio.ensure_future(), which also takes care of starting them
			
            tasks.append(task)
			
		
        await asyncio.gather(*tasks, return_exceptions=True)
        #asyncio.gather() to keep the session context alive until all of the tasks have completed.


if __name__ == "__main__":
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
	
	
    asyncio.get_event_loop().run_until_complete(download_all_sites(sites))
    #start up the event loop and tell it which tasks to run
	
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")