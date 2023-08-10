"""
Async utilities
"""
import asyncio
from typing import Callable, Coroutine, List


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
