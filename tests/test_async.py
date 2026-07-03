import asyncio

from nipost._async import worker


def test_worker_runs_job_and_returns_result():
    async def main():
        sem = asyncio.Semaphore(1)
        return await worker(lambda: 6 * 7, sem)

    assert asyncio.run(main()) == 42
