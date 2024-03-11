"""
Async utilities
"""

import asyncio
from typing import AsyncIterable, AsyncIterator, Callable, Coroutine, List, TypeVar


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


T = TypeVar("T")


class ChunkedAsyncIterator(AsyncIterator[AsyncIterator[T]]):
    def __init__(self, source: AsyncIterable[T], chunk_size: int):
        self.source = source
        self.chunk_size = chunk_size
        self.buffer: list[T] = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> AsyncIterator[T]:
        if not self.buffer:  # Fill the buffer if it's empty
            async for item in self.source:
                self.buffer.append(item)
                if len(self.buffer) == self.chunk_size:
                    break
            if not self.buffer:  # No more items to read
                raise StopAsyncIteration

        # Prepare to yield the current chunk
        chunk, self.buffer = self.buffer, []  # Swap out the buffer

        # Define and return an async generator for the current chunk
        async def chunk_generator():
            for item in chunk:
                yield item

        return chunk_generator()
