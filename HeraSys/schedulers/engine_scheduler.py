import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ray

from HeraSys.configs.config import EngineConfig
from HeraSys.logger import GLOBAL_INFO_LEVEL, get_logger
from HeraSys.queries.query import Query

from asyncio import PriorityQueue
from sortedcontainers import SortedList

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


@dataclass
class EngineRequest:
    """Data class for engine requests"""
    stq: float
    source_id: str
    contribution: float
    is_long_heavy_node: bool
    downstream_urgency: int
    priority: float
    node_name: str
    op_type: str
    request_id: str  # unique id for each request, {query_id}_{node_name}
    query_id: str
    query: Query
    payload: Any  # Request payload (e.g., texts for embedding), generated from the payload_transformer, could be different for different engines
    result_ref: Optional[ray.ObjectRef] = None
    timestamp: float = time.time()
    timing_profile: Dict[str, float] = field(default_factory=dict)
    # 2 多实例
    engine_instance_index: Optional[int] = None

    def __lt__(self, other):
        if self.priority == other.priority:
            if self.contribution == other.contribution:
                return self.timestamp < other.timestamp
            return self.contribution > other.contribution
        return self.priority > other.priority

class BaseEngineScheduler(ABC):
    """Abstract base class for engine schedulers"""

    @abstractmethod
    async def submit_request(self, request: EngineRequest):
        """Submit a request to the engine"""
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the scheduler and all engine instances"""
        pass


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies"""

    @abstractmethod
    def get_next_engine(
        self, engines: List[Any], current_idx: int
    ) -> "tuple[Any, int]":
        """Get the next available engine

        Args:
            engines: list of engine instances
            current_idx: current engine index

        Returns:
            tuple[engine, new_idx]: selected engine and updated index
        """
        pass


class RoundRobinStrategy(SchedulingStrategy):
    """Round-robin scheduling strategy"""

    def get_next_engine(
        self, engines: List[Any], current_idx: int
    ) -> "tuple[Any, int]":
        if not engines:
            raise ValueError("No engines available")
        engine = engines[current_idx]
        new_idx = (current_idx + 1) % len(engines)
        return engine, new_idx


@ray.remote
class EngineScheduler(BaseEngineScheduler):
    def __init__(
        self,
        engine_class,
        engine_config: EngineConfig,  # use EngineConfig as config
        graph_scheduler: ray.actor.ActorHandle,
        **engine_kwargs,
    ):

        self.engine_class = engine_class
        self.name = engine_config.name
        self.num_instances = engine_config.instances
        self.num_gpus = engine_config.num_gpus
        self.num_cpus = engine_config.num_cpus
        self.resources = engine_config.resources
        self.engine_type = engine_config.engine_type
        self.graph_scheduler = graph_scheduler

        # 5 自定义gpu资源
        self.target_gpu_id = engine_config.target_gpu_id

        self.engine_kwargs = {
            **engine_kwargs,
            **(engine_config.model_config or {}),  # merge model_config
        }
        self.engines = []
        self._create_engines()

        # set scheduling strategy, default using round robin
        self.scheduling_strategy = RoundRobinStrategy()

        self.single_engine_mode = self.num_instances == 1
        self.num_long_query_engines = 0 if self.single_engine_mode else engine_config.resources.get(
            "num_long_query_engines", 1)
        self.long_engine_start_index = self.num_instances - self.num_long_query_engines

        # create request queue
        #self.request_queue = asyncio.Queue(maxsize=1000)
        #self.request_queue = PriorityQueue(maxsize=1000)


        self.sorted_sources = SortedList(key=lambda item: item[1])
        self.source_queues: Dict[str, PriorityQueue] = {}
        self.source_stq_map: Dict[str, float] = {}
        self.long_heavy_queue = PriorityQueue(maxsize=1000)


        self.max_batch_size = 8
        self.contribution_threshold_factor = 1.5
        self.idle_wait_time_ms = 5

        self.busy_engine_indices: set[int] = set()

        # initialize current engine index
        self.current_engine_idx = 0

        self.pending_requests: Dict[str, EngineRequest] = {}

        # Start processing tasks
        self.running = True
        self.loop = asyncio.get_event_loop()
        #self.submit_task = self.loop.create_task(self._submit_requests())
        #self.submit_task = self.loop.create_task(self._batching_loop())
        self.short_query_task = self.loop.create_task(self._short_query_loop())
        if not self.single_engine_mode:
            self.long_query_task = self.loop.create_task(self._long_query_loop())
        self.result_task = self.loop.create_task(self._process_results())
        self._is_ready = True


        # The queue for completed requests
        self.complete_queue = asyncio.Queue(maxsize=1000)

        self.trace_enabled = True
        self.trace_events: List[Dict[str, Any]] = []
        self.enable_long_skewing = True
        self.enable_adaptive_batching = True


    def _record_trace(self, event: str, **fields) -> None:
        if not self.trace_enabled:
            return
        record = {
            "ts": time.perf_counter(),
            "component": "engine_scheduler",
            "scheduler_name": self.name,
            "event": event,
        }
        record.update(fields)
        self.trace_events.append(record)

    def get_trace_events(self) -> List[Dict[str, Any]]:
        return list(self.trace_events)

    def clear_trace_events(self) -> None:
        self.trace_events.clear()

    async def update_runtime_config(self, config: Dict[str, Any]) -> None:
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # 4 同时启动引擎
    def _create_engines(self):
        """create engine instances based on EngineConfig"""

        engine_actors = []
        ready_futures = []

        for i in range(self.num_instances):
            # create options dict, only include custom resources in resources
            options = {}
            if self.resources:
                custom_resources = {
                    k: v
                    for k, v in self.resources.items()
                    if k.upper() not in ["CPU", "GPU"]
                }
                if custom_resources:
                    options["resources"] = custom_resources

            # 5 自定义gpu资源
            if self.target_gpu_id is not None:
                runtime_env = {
                    "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
                }
                options["runtime_env"] = runtime_env

            if self.num_gpus:
                options["num_gpus"] = self.num_gpus
            if self.num_cpus:
                options["num_cpus"] = self.num_cpus
            else:
                options["num_cpus"] = 1

            # directly use the current actor's handle
            scheduler_handle = ray.runtime_context.get_runtime_context().current_actor

            # when creating engine instance, pass in scheduler handle
            logger.info(
                f"try to create engine instance with name: {self.name}_{i} for class: {self.engine_class} with options: {options}"
            )

            engine = self.engine_class.options(**options).remote(
                name=f"{self.name}_{i}",
                scheduler_ref=scheduler_handle,  # directly use actor handle
                target_gpu_id=self.target_gpu_id[i] if self.target_gpu_id else None,
                instance_id = i,
                **self.engine_kwargs,
            )

            engine_actors.append(engine)
            ready_futures.append(engine.is_ready.remote())

            #_ = ray.get(engine.is_ready.remote())
            #self.engines.append(engine)

        #在循环结束后，使用 ray.get() 一次性、并行地等待所有实例完成启动
        ray.get(ready_futures)

        #将所有已就绪的引擎句柄一次性加入 self.engines
        self.engines.extend(engine_actors)
        logger.info(
            f"✅ Successfully created and started {len(self.engines)} instances for '{self.name}'."
        )


    # 2 多实例
    def _get_next_engine(self, engine_instance_index: int):
        """use scheduling strategy to get next available engine"""
        if engine_instance_index:
            if engine_instance_index >= len(self.engines):
                engine, new_idx = self.scheduling_strategy.get_next_engine(
                    self.engines, self.current_engine_idx
                )
                self.current_engine_idx = new_idx
                return engine
            engine = self.engines[engine_instance_index]
            return engine
        engine, new_idx = self.scheduling_strategy.get_next_engine(
            self.engines, self.current_engine_idx
        )
        self.current_engine_idx = new_idx
        return engine

    async def add_engine(self):
        """add a new engine instance dynamically"""
        engine = self.engine_class.remote(
            name=f"{self.name}_{self.num_instances}",
            resource=self.resource,
            scheduler_ref=self,
            **self.engine_kwargs,
        )
        self.engines.append(engine)
        self.num_instances += 1
        return len(self.engines)

    async def remove_engine(self, index: Optional[int] = None):
        """remove an engine instance

        Args:
            index: the index of the engine to remove, if None, remove the last one
        """
        if not self.engines or self.num_instances <= 1:
            raise ValueError("Cannot remove the last engine instance")

        if index is None:
            index = len(self.engines) - 1

        if 0 <= index < len(self.engines):
            engine = self.engines.pop(index)
            await engine.shutdown.remote()
            self.num_instances -= 1

            # adjust current engine index
            if self.current_engine_idx >= len(self.engines):
                self.current_engine_idx = 0

            return len(self.engines)
        else:
            raise ValueError(f"Invalid engine index: {index}")


    async def _process_results(self):
        """Task for processing completed results"""
        while self.running:
            try:
                request = await self.complete_queue.get()

                if request.result_ref is not None:
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, ray.get, request.result_ref
                        )
                        logger.debug(f"result in engine scheduler: {result}")

                        query_states = request.query.query_state
                        # Use ray.get to wait for result completion
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            ray.get,
                            query_states.set_node_result.remote(
                                request.request_id.split("::")[
                                    -1
                                ],  # here we assume the request_id is in the format of {query_id}::{node_name} as in the graph scheduler submit_node method
                                request.result_ref,
                            ),
                        )

                        logger.info(
                            f"Successfully set result for request {request.request_id}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Error processing result for request {request.request_id}: {e}"
                        )

                    finally:
                        # clean up completed request
                        self.complete_queue.task_done()
                        if request.request_id in self.pending_requests:
                            del self.pending_requests[request.request_id]

                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in result processing loop: {e}")
                await asyncio.sleep(0.01)

    async def submit_request(self, request: EngineRequest):
        """请求路由器：根据节点类型和模式，分发到不同的队列体系"""
        # 如果是长查询重节点 且 非单引擎模式，放入专用队列
        # 记录请求到达 EngineScheduler 的时间点
        request.timing_profile["T1_engine_scheduler_received"] = time.perf_counter()
        if self.enable_long_skewing and request.is_long_heavy_node and not self.single_engine_mode:
            await self.long_heavy_queue.put(request)
            logger.info(f"长查询重节点 {request.request_id} 已加入专用队列。")
        else:
            # 单引擎模式下，所有节点都走短查询逻辑
            source_id = request.source_id
            if source_id not in self.source_queues:
                self.source_queues[source_id] = PriorityQueue(maxsize=1000)
                self.source_stq_map[source_id] = request.stq
                self.sorted_sources.add((source_id, request.stq))
            await self.source_queues[source_id].put(request)
            print(f"节点 {request.request_id} 已加入 {self.name} 短查询/通用队列。")

        self.pending_requests[request.request_id] = request
        self._record_trace('queue_route', request_id=request.request_id, source_id=request.source_id, priority=request.priority, routed_to=('long' if self.enable_long_skewing and request.is_long_heavy_node and not self.single_engine_mode else 'short'), is_long_heavy=request.is_long_heavy_node)

    def _get_idle_engine_indices(self, start_idx: int, end_idx: int) -> List[int]:
        """获取指定索引范围内的所有空闲引擎的索引"""
        return [i for i in range(start_idx, end_idx) if i not in self.busy_engine_indices]

    def _pick_best_candidate(self, candidates: List[EngineRequest]) -> EngineRequest:
        return max(
            candidates,
            key=lambda req: (req.priority, req.contribution, -req.timestamp),
        )

    async def _short_query_loop(self):
        """专门处理短查询/普通节点的调度循环（在单引擎模式下处理所有节点）"""
        while self.running:
            # 1. 尽力凑一个批次
            batch = await self._form_short_query_batch()

            if not batch:
                await asyncio.sleep(0.002)
                continue

            # 2. 如果批次已满，直接发送
            if len(batch) >= self.max_batch_size:
                await self._submit_batch(batch, "short")
                continue

            # 3. 批次未满且无节点可取，进入等待决策
            engine_pool_end = self.num_instances if self.single_engine_mode else self.long_engine_start_index
            idle_short_engines = self._get_idle_engine_indices(0, engine_pool_end)

            # 决策点4：如果空闲引擎足够多，直接发送
            if len(idle_short_engines) >= 2:
                logger.info(f" {self.name} 空闲短查询/通用引擎充足({len(idle_short_engines)}个)，直接发送不完整的批次。")
                await self._submit_batch(batch, "short")
                continue

            # 决策点5：空闲引擎不足，进行智能等待
            imminent_info = await self.graph_scheduler.check_for_imminent_nodes.remote(self.engine_type)
            if not imminent_info.get("imminent"):
                logger.info(f" {self.name} 无即将到来的节点，直接发送不完整的批次。")
                await self._submit_batch(batch, "short")
            else:
                #await self._submit_batch(batch, "short")
                logger.info(f" {self.name} 发现有将到来的节点{imminent_info}，进行等待。")
                lambda_val = await self.graph_scheduler.get_lambda.remote()
                imminent_contribution = imminent_info.get("contribution", 0)
                if self.enable_adaptive_batching:
                    wait_time = self.idle_wait_time_ms * (1 + lambda_val * 0.5) * (1 + imminent_contribution / 100)
                else:
                    wait_time = 0.0
                logger.info(f" {self.name} 检测到即将到来的节点(贡献度:{imminent_contribution:.2f})，智能等待 {wait_time:.2f}ms。")
                self._record_trace('adaptive_wait', lambda_value=lambda_val, imminent_contribution=imminent_contribution, wait_time_ms=wait_time)
                await asyncio.sleep(wait_time / 1000.0)
                if not any(not q.empty() for q in self.source_queues.values()):
                    await self._submit_batch(batch, "short")
                else:
                    candidates: List[EngineRequest] = []
                    for queue in self.source_queues.values():
                        if queue and not queue.empty():
                            candidates.append(queue._queue[0])

                    if not candidates:
                        break

                    winner = self._pick_best_candidate(candidates)
                    actual_node = await self.source_queues[winner.source_id].get()
                    batch.append(actual_node)
                    await self._submit_batch(batch, "short")

    async def _long_query_loop(self):
        """专门处理长查询重节点，并实现资源借用的调度循环（仅在多引擎模式下运行）"""
        while self.running:
            batch: List[EngineRequest] = []

            # 优先处理长查询重节点
            while not self.long_heavy_queue.empty() and len(batch) < self.max_batch_size:
                batch.append(await self.long_heavy_queue.get())

            if batch:
                self._record_trace('resource_skew_dispatch', queue='long', batch_size=len(batch))
                await self._submit_batch(batch, "long")
                continue

            idle_long_engines = self._get_idle_engine_indices(self.long_engine_start_index, self.num_instances)
            if idle_long_engines:
                short_query_batch = await self._form_short_query_batch()
                if short_query_batch:
                    logger.info(f" {self.name} 长查询引擎借用资源，处理 {len(short_query_batch)} 个短查询节点。")
                    self._record_trace('resource_skew_dispatch', queue='long_borrow', batch_size=len(short_query_batch))
                    await self._submit_batch(short_query_batch, "long_borrow")

            await asyncio.sleep(0.002)

    async def _form_short_query_batch(self) -> List[EngineRequest]:
        """按照“轮轮对比”的策略，为短查询凑一个批次"""
        t_begin = time.perf_counter()
        batch: List[EngineRequest] = []
        candidate_rounds = 0
        while len(batch) < self.max_batch_size:
            if not any(not q.empty() for q in self.source_queues.values()):
                break

            candidates: List[EngineRequest] = []
            for queue in self.source_queues.values():
                if queue and not queue.empty():
                    candidates.append(queue._queue[0])

            if not candidates:
                break

            candidate_rounds += 1
            winner = self._pick_best_candidate(candidates)
            actual_node = await self.source_queues[winner.source_id].get()
            batch.append(actual_node)

        self._record_trace('form_short_batch', batch_size=len(batch), candidate_rounds=candidate_rounds, duration_ms=(time.perf_counter() - t_begin) * 1000)
        return batch

    async def _submit_batch(self, batch: List[EngineRequest], engine_type: str):
        """
        提交批次并更新引擎状态（已修正批处理逻辑）。
        将一整个批次，作为一个单元，交给一个空闲引擎处理。
        """
        if not batch:
            return

        t_before_submit = time.perf_counter()
        for req in batch:
            req.timing_profile["T2_engine_submitted"] = t_before_submit

        # 1. 根据类型确定引擎池的索引范围
        if engine_type == "short":
            pool_start, pool_end = 0, self.num_instances if self.single_engine_mode else self.long_engine_start_index
        else:  # "long" or "long_borrow"
            pool_start, pool_end = self.long_engine_start_index, self.num_instances

        # 2. 从指定的池中寻找一个空闲引擎
        idle_indices = self._get_idle_engine_indices(pool_start, pool_end)

        if not idle_indices:
            # 如果池中没有任何空闲引擎，则将整个批次放回队列
            logger.warning(f" {self.name} 引擎池全忙，{len(batch)} 个任务将放回队列。")
            for req in batch: await self.submit_request(req)
            return

        # 3. 只要有至少一个空闲引擎，就选择第一个来处理整个批次
        engine_to_use_idx = idle_indices[0]

        # 4. 将该引擎标记为忙碌
        #self.busy_engine_indices.add(engine_to_use_idx)

        logger.info(f" {self.name} 成功将大小为 {len(batch)} 的批次分配给引擎 {engine_to_use_idx}。")
        self._record_trace('dispatch_batch', engine_pool=engine_type, batch_size=len(batch), engine_index=engine_to_use_idx, idle_candidates=len(idle_indices))

        # 5. 将批次中的每一个请求都提交给这同一个引擎
        engine = self.engines[engine_to_use_idx]

        # --- 核心修改点：根据引擎类型进行特判 ---

        # 3.A 如果是 Embedding 引擎
        if self.name in ["embedding_service", "reranker_service", "llm_service"]:
            # 准备要发送的数据：将 EngineRequest 对象列表转换为字典列表
            batch_payload = []
            for request in batch:
                request.engine_instance_index = engine_to_use_idx
                payload_data = request.payload or {}
                batch_payload.append({
                    "request_id": request.request_id,
                    "query_id": request.query_id,
                    "timing_profile": request.timing_profile,
                    **payload_data
                })

            # 一次性提交整个批次
            batch_node_names = [req.request_id.split("::")[-1] for req in batch]
            logger.info( f" {self.name} 向引擎 {engine_to_use_idx} ({self.name}) 提交批次，大小: {len(batch_payload)}，节点顺序: {batch_node_names}")
            await engine.submit_batch.remote(batch_payload)

        # 3.B 如果是其他类型引擎 (如 LLM)
        else:
            logger.info(f" {self.name}  向引擎 {engine_to_use_idx} 循环提交 {len(batch)} 个任务。")
            # 保持原有的循环单点提交逻辑
            for request in batch:
                request.engine_instance_index = engine_to_use_idx
                await engine.submit_request.remote(
                    request_id=request.request_id,
                    query_id=request.query_id,
                    timing_profile=request.timing_profile,
                    **request.payload
                )

    async def on_result(
        self, request_id: str, query_id: str, result_ref_from_engine: Any,timing_profile: Dict[str, float] = None
    ):
        """Handle result callback from engine"""
        if not isinstance(result_ref_from_engine, ray.ObjectRef):
            result_ref_from_engine = ray.put(result_ref_from_engine)
        try:
            t_result_received = time.perf_counter()
            if request_id in self.pending_requests:
                request = self.pending_requests[request_id]
                request.result_ref = result_ref_from_engine

                T0 = request.timing_profile.get("T0_graph_scheduler_sent", 0)
                T1 = request.timing_profile.get("T1_engine_scheduler_received", 0)
                T2 = timing_profile.get("T2_engine_submitted", 0)
                T3 = timing_profile.get("T3_engine_received", 0)
                T4 = timing_profile.get("T4_compute_start", 0)
                T5 = timing_profile.get("T5_compute_end", 0)
                T6 = t_result_received

                # 计算各个阶段的耗时 (单位：毫秒)
                graph_to_sched_transit = (T1 - T0) * 1000
                sched_queue_time = (T2 - T1) * 1000
                sched_to_engine_transit = (T3 - T2) * 1000
                engine_queue_and_batching_time = (T4 - T3) * 1000
                actual_compute_time = (T5 - T4) * 1000
                result_return_transit = (T6 - T5) * 1000

                print(
                    f"\n--- TIMING PROFILE FOR {request_id} ---\n"
                    f"  Graph -> Sched Transit:      {graph_to_sched_transit:8.2f} ms\n"
                    f"  Sched Queue Time:            {sched_queue_time:8.2f} ms\n"
                    f"  Sched -> Engine Transit:     {sched_to_engine_transit:8.2f} ms\n"
                    f"  Engine Batching & Wait:      {engine_queue_and_batching_time:8.2f} ms\n"
                    f"  >>> ACTUAL COMPUTE TIME <<<: {actual_compute_time:8.2f} ms\n"
                    f"  Result Return Transit:       {result_return_transit:8.2f} ms\n"
                    f"---------------------------------------------------\n"
                )

                self._record_trace(
                    "engine_request_timing",
                    request_id=request_id,
                    query_id=query_id,
                    node_name=request.node_name,
                    op_type=request.op_type,
                    graph_to_sched_transit_ms=graph_to_sched_transit,
                    sched_queue_ms=sched_queue_time,
                    sched_to_engine_ms=sched_to_engine_transit,
                    engine_queue_ms=engine_queue_and_batching_time,
                    compute_ms=actual_compute_time,
                    return_ms=result_return_transit,
                )

                # put completed request into complete queue
                await self.complete_queue.put(request)
                logger.info(f"Moved request {request_id} to complete queue")

                # Clean up pending_requests
                del self.pending_requests[request_id]
            else:
                logger.warning(f"Received result for unknown request {request_id}")

        except Exception as e:
            logger.error(f"Error processing result callback: {e}")

    async def shutdown(self):
        """Shutdown the scheduler and all engine instances"""
        self.running = False
        tasks = [self.short_query_task, self.result_task]
        if not self.single_engine_mode:
            tasks.append(self.long_query_task)
        # Cancel processing tasks
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown all engine instances
        for engine in self.engines:
            await engine.shutdown.remote()

        # Clear engine list
        self.engines.clear()
        self.num_instances = 0

    async def is_ready(self) -> bool:
        """Check if the scheduler is fully initialized

        Returns:
            bool: if the scheduler is fully initialized and ready to process requests
        """
        # check if all necessary components are initialized
        if not self.engines:
            return False

        # check if async tasks are running
        if not self.single_engine_mode:
            if not self.long_query_task or self.long_query_task.done():
                return False
        if not self.short_query_task or self.short_query_task.done():
            return False
        if not self.result_task or self.result_task.done():
            return False

        return self._is_ready

