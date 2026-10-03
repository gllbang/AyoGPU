import asyncio

from HeraSys import HeraSys
from HeraSys.backends import reference_pools
from HeraSys.workflows import rag


async def main():
    dag, inputs = rag(
        "What does HeraSys optimize?",
        ["HeraSys jointly schedules concurrent workflows and reuses equivalent computations."],
        max_tokens=32,
    )
    async with HeraSys(reference_pools()) as runtime:
        result = await runtime.submit(dag, inputs, query_id="q0")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
