import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import os

from HeraSys.logger import GLOBAL_INFO_LEVEL, get_logger

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


@dataclass
class EmbeddingRequest:
    """Data class for embedding requests"""

    request_id: str
    query_id: str  # Group requests from same query
    texts: List[str]
    callback_ref: Any  # Ray ObjectRef for result
    timestamp: float = time.time()
    timing_profile: Dict[str, float] = field(default_factory=dict)

@ray.remote
class EmbeddingEngine:
    """Ray Actor for serving embedding requests with async processing

    Features:
    - Async request handling
    - Batches requests for efficient processing
    - Groups requests from same query
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        max_batch_size: int = 20480,
        max_queue_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler_ref: Optional[ray.actor.ActorHandle] = None,
        target_gpu_id: Optional[int] = None,
        **kwargs,
    ):
        self.device = device
        if target_gpu_id is not None:
            self.device = self.device +":" + str(target_gpu_id)

        print(f"CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU device: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name()}")

        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        #self.device = device
        self.name = kwargs.get("name", None)

        # Initialize model
        self.model = self._load_model()

        # Async queues
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)

        # Track requests by query_id
        self.query_requests: Dict[str, List[EmbeddingRequest]] = {}

        # Create event loop
        self.loop = asyncio.get_event_loop()

        # Start processing tasks
        self.running = True
        self.tasks = [
            self.loop.create_task(self._batch_requests()),
            self.loop.create_task(self._process_batches()),
        ]

        self.scheduler_ref = scheduler_ref

    def is_ready(self):
        """Check if the engine is ready"""
        return True

    def _load_model(self):
        """Load the embedding model"""
        from sentence_transformers import SentenceTransformer

        # self.model.encode(["hello","world"])

        model = SentenceTransformer(
            model_name_or_path=self.model_name, device=self.device
        )
        model.half()
        model.eval()

        # warm up
        warm_up_embeddings = model.encode(["hello", "world"])
        logger.debug(f"warm_up_embeddings: {warm_up_embeddings}")

        logger.debug(
            f"Load and warm up embedding model:{self.model_name} successfully on {self.device}"
        )

        return model

    async def submit_batch(self, batch: List[Dict[str, Any]]):
        """
        新的、面向批次的入口方法。
        将收到的批次直接放入 batch_queue。
        """
        if not batch:
            return
        t_engine_received = time.perf_counter()
        if self.batch_queue.full():
            logger.error(f"引擎 '{self.name}' 的 batch_queue 已满，批次可能被丢弃！")
            # 实际生产中可能需要更复杂的背压或重试逻辑
            return

        # 1. 将收到的字典列表转换为内部需要的格式
        batch_requests = []
        for req in batch:
            # 从载荷中获取已有的计时信息
            timing_profile = req.get("timing_profile", {})
            # 添加新的时间戳
            timing_profile["T3_engine_received"] = t_engine_received

            batch_requests.append(
                EmbeddingRequest(
                    request_id=req['request_id'],
                    query_id=req['query_id'],
                    texts=req['texts'],
                    callback_ref=None,
                    # 将更新后的 timing_profile 存入请求对象
                    # 您需要确保 EmbeddingRequest 的 dataclass 中有 timing_profile 字段
                    timing_profile=timing_profile
                )
            )

        batch_texts = [text for req in batch_requests for text in req.texts]

        # 2. 将格式化后的完整批次放入处理队列
        await self.batch_queue.put((batch_requests, batch_texts))
        logger.info(f"引擎 '{self.name}' 已将一个大小为 {len(batch)} 的批次放入处理队列。")


    async def submit_request(
        self, request_id: str, query_id: str, texts: List[str],timing_profile: Dict[str, float]
    ) -> ray.ObjectRef:
        """Submit a new embedding request"""
        timing_profile["T3_engine_received"] = time.perf_counter()
        request = EmbeddingRequest(
            request_id=request_id, query_id=query_id, texts=texts, callback_ref=None,timing_profile=timing_profile
        )

        if self.request_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Request queue is full")

        await self.request_queue.put(request)

        if query_id not in self.query_requests:
            self.query_requests[query_id] = []

        self.query_requests[query_id].append(request)


    async def _batch_requests(self):
        """Async task for batching requests"""
        while self.running:
            try:
                batch_requests, batch_texts = await self._get_next_batch()
                if batch_requests:
                    await self.batch_queue.put((batch_requests, batch_texts))
                else:
                    await asyncio.sleep(0.01)  # Avoid busy waiting
            except Exception as e:
                print(f"Error in batching task: {e}")
                continue

    async def _process_batches(self):
        """Async task for processing batches"""
        while self.running:
            try:
                try:
                    batch_requests, batch_texts = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # print(f"Processing batch requests: {batch_requests}")

                try:
                    for req in batch_requests:
                        req.timing_profile["T4_compute_start"] = time.perf_counter()
                    embeddings = await self.loop.run_in_executor(
                        None, self._compute_embeddings, batch_texts
                    )
                    t_compute_end = time.perf_counter()
                    for req in batch_requests:
                        req.timing_profile["T5_compute_end"] = t_compute_end
                    start_idx = 0
                    for request in batch_requests:
                        try:
                            end_idx = start_idx + len(request.texts)
                            request_embeddings = embeddings[start_idx:end_idx]

                            # create ObjectRef for result
                            print(f"request_embeddings: {request_embeddings.shape}")

                            result_ref = ray.put(request_embeddings)

                            # If scheduler is set, send result to scheduler

                            if self.scheduler_ref is not None:
                                await self.scheduler_ref.on_result.remote(
                                    request.request_id, request.query_id, result_ref,request.timing_profile
                                )
                            else:
                                # If no scheduler is set, use the original callback
                                ray.get(
                                    ray.put(
                                        request_embeddings, _owner=request.callback_ref
                                    )
                                )

                            # clean up request records
                            # if request.query_id in self.query_requests:
                            #     self.query_requests[request.query_id].remove(request)
                            #     if not self.query_requests[request.query_id]:
                            #         del self.query_requests[request.query_id]

                            start_idx = end_idx
                        except Exception as e:
                            print(f"Error processing individual request: {e}")
                            continue

                except Exception as e:
                    print(f"Error computing embeddings: {e}")
                    continue

            except Exception as e:
                print(f"Error in inference task: {e}")
                continue

    async def _get_next_batch(self) -> Tuple[List[EmbeddingRequest], List[str]]:
        """Get next batch of requests to process"""
        batch_requests = []
        batch_texts = []

        # 改：若第一个就超过max，会出现死循环
        #while len(batch_texts) < self.max_batch_size:
        while len(batch_texts) == 0:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)

                if len(batch_texts) + len(request.texts) <= self.max_batch_size:
                    batch_requests.append(request)
                    batch_texts.extend(request.texts)
                else:
                    if len(batch_texts) == 0:
                        logger.warning(
                            f"Request {request.request_id} with {len(request.texts)} texts "
                            f"exceeds max_batch_size {self.max_batch_size}. Processing it as a single batch."
                        )
                        batch_requests.append(request)
                        batch_texts.extend(request.texts)
                        return batch_requests, batch_texts
                    await self.request_queue.put(request)
                    break

            except asyncio.TimeoutError:
                break

        return batch_requests, batch_texts


    # async def _get_next_batch(self) -> Tuple[List[EmbeddingRequest], List[str]]:
    #     """Get next batch of requests to process"""
    #     batch_requests = []
    #     batch_texts = []
    #     processed_queries = set()
    #
    #     # 改：若第一个就超过max，会出现死循环
    #     while len(batch_texts) < self.max_batch_size:
    #     #while len(batch_texts) == 0:
    #         try:
    #             request = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)
    #
    #             if request.query_id in processed_queries:
    #                 pending_requests = self.query_requests[request.query_id]
    #                 for pending_req in pending_requests:
    #                     if (
    #                         len(batch_texts) + len(pending_req.texts)
    #                         <= self.max_batch_size
    #                     ):
    #                         batch_requests.append(pending_req)
    #                         batch_texts.extend(pending_req.texts)
    #             else:
    #                 if len(batch_texts) + len(request.texts) <= self.max_batch_size:
    #                     batch_requests.append(request)
    #                     batch_texts.extend(request.texts)
    #                     processed_queries.add(request.query_id)
    #                 else:
    #                     if len(batch_texts) == 0:
    #                         logger.warning(
    #                             f"Request {request.request_id} with {len(request.texts)} texts "
    #                             f"exceeds max_batch_size {self.max_batch_size}. Processing it as a single batch."
    #                         )
    #                         batch_requests.append(request)
    #                         batch_texts.extend(request.texts)
    #                         return batch_requests, batch_texts
    #                     await self.request_queue.put(request)
    #                     break
    #
    #         except asyncio.TimeoutError:
    #             break
    #
    #     return batch_requests, batch_texts

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a batch of texts"""
        # self.model.encode(["hello","world"])
        with torch.no_grad():

            assert isinstance(texts, list) or isinstance(texts, str)

            begin = time.time()

            batch_size = len(texts) if isinstance(texts, list) else 1
            embeddings = self.model.encode(
                texts, batch_size=512, show_progress_bar=False
            )
            # embeddings = embeddings[:,:768]

            logger.debug(
                f"texts' type: {type(texts)}, len: {len(texts)}, embeddings shape: {embeddings.shape}"
            )
            end = time.time()
            logger.debug(f"embedding time for {len(texts)} texts: {end - begin}")
            return embeddings

    async def shutdown(self):
        """Shutdown the service"""
        self.running = False
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

