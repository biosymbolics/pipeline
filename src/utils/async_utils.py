"""
Async utilities
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterable, Callable, Coroutine, Iterable, List, TypeVar


async def execute_async(functions: List[Callable[[], Coroutine]]) -> None:
    """
    Execute functions asynchronously

    Args:
        functions (List[Callable[[], Any]]): Functions

    Example:
    ```
    def do_async_stuff():
        closures = [__func(param=param) for param in parameters]
        await execute_async(closures)
    asyncio.run(do_async_stuff())
    ```
    """
    tasks = [asyncio.create_task(func()) for func in functions]
    await asyncio.gather(*tasks)


async def gather_with_concurrency_limit(n: int, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))
