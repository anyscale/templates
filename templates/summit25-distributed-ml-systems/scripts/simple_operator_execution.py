"""
Simplified Streaming Operator Execution Demo

This script demonstrates the core concepts of Ray Data's streaming execution:
1. Task-pool based operators that stream blocks via generators
2. A scheduling loop that selects and dispatches tasks
3. Static resource allocation policies for constraints
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict
import ray


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Block:
    """A block of data with some metadata."""

    data: List[int]
    block_id: int

    def size_bytes(self) -> int:
        """Estimate memory size."""
        return len(self.data) * 8  # 8 bytes per int

    def num_rows(self) -> int:
        return len(self.data)


@dataclass
class Resources:
    """Resource allocation tracking."""

    cpu: float = 0.0
    memory: int = 0  # bytes

    def __add__(self, other):
        return Resources(cpu=self.cpu + other.cpu, memory=self.memory + other.memory)

    def __le__(self, other):
        return self.cpu <= other.cpu and self.memory <= other.memory


# ============================================================================
# Task Functions (run as Ray tasks)
# ============================================================================


@ray.remote(num_cpus=1, num_returns="streaming")
def map_task_generator(*blocks: Block, transform_fn, task_id: int) -> Iterator[Block]:
    """
    A Ray task that processes blocks and yields results as a streaming generator.
    This mimics how Ray Data's _map_task works.
    """
    print(f"  [Task {task_id}] Started processing {len(blocks)} blocks")

    for block in blocks:
        # Transform the data
        transformed_data = transform_fn(block.data)

        # Build output block(s)
        output_block = Block(data=transformed_data, block_id=block.block_id)

        print(
            f"  [Task {task_id}] Yielding block {output_block.block_id} "
            f"with {output_block.num_rows()} rows"
        )

        # Yield block incrementally (streaming!)
        yield output_block

    print(f"  [Task {task_id}] Completed")


# ============================================================================
# Operator Classes
# ============================================================================


class OperatorState:
    """
    Tracks the execution state for an operator.
    Similar to OpState in streaming_executor_state.py
    """

    def __init__(self, operator: "Operator"):
        self.operator = operator
        self.input_queue: deque = deque()
        self.output_queue: deque = deque()
        self.active_tasks: Dict[int, ray.ObjectRefGenerator] = {}
        self.next_task_id = 0
        self.completed_tasks = 0
        self.inputs_done = False

    def has_pending_input(self) -> bool:
        """Check if there are inputs ready to process."""
        return len(self.input_queue) > 0

    def has_output(self) -> bool:
        """Check if there are outputs ready."""
        return len(self.output_queue) > 0

    def num_active_tasks(self) -> int:
        """Number of currently running tasks."""
        return len(self.active_tasks)


class Operator:
    """
    Base operator class that processes blocks of data.
    Similar to MapOperator in map_operator.py
    """

    def __init__(
        self,
        name: str,
        transform_fn,
        resource_per_task: Resources,
        max_concurrency: Optional[int] = None,
    ):
        self.name = name
        self.transform_fn = transform_fn
        self.resource_per_task = resource_per_task
        self.max_concurrency = max_concurrency
        self.state: Optional[OperatorState] = None

    def initialize(self):
        """Initialize operator state."""
        self.state = OperatorState(self)

    def add_input(self, block_refs: List[ray.ObjectRef]):
        """Add input block references to the operator's queue."""
        self.state.input_queue.append(block_refs)
        print(
            f"[{self.name}] Added {len(block_refs)} block refs to input queue "
            f"(queue size: {len(self.state.input_queue)})"
        )

    def mark_inputs_done(self):
        """Signal that no more inputs will be added."""
        self.state.inputs_done = True
        print(f"[{self.name}] No more inputs will be added")

    def should_dispatch(self) -> bool:
        """
        Check if operator can dispatch a new task.
        Similar to the eligibility check in select_operator_to_run()
        """
        if not self.state.has_pending_input():
            return False

        if self.max_concurrency is not None:
            if self.state.num_active_tasks() >= self.max_concurrency:
                return False

        return True

    def dispatch_task(self):
        """
        Dispatch a new task from the input queue.
        Similar to dispatch_next_task() in streaming_executor_state.py
        """
        if not self.state.has_pending_input():
            return

        block_refs = self.state.input_queue.popleft()
        task_id = self.state.next_task_id
        self.state.next_task_id += 1

        print(
            f"[{self.name}] Dispatching task {task_id} with {len(block_refs)} block refs"
        )

        # Submit Ray task that returns a streaming generator
        gen = map_task_generator.remote(
            *block_refs, transform_fn=self.transform_fn, task_id=task_id
        )
        self.state.active_tasks[task_id] = gen

    def current_resource_usage(self) -> Resources:
        """Calculate current resource usage."""
        num_tasks = self.state.num_active_tasks()
        return Resources(
            cpu=self.resource_per_task.cpu * num_tasks,
            memory=self.resource_per_task.memory * num_tasks,
        )

    def is_completed(self) -> bool:
        """Check if operator has finished all work."""
        return (
            self.state.inputs_done
            and len(self.state.input_queue) == 0
            and len(self.state.active_tasks) == 0
        )


# ============================================================================
# Resource Manager & Backpressure Policy
# ============================================================================


class StaticResourcePolicy:
    """
    A simple static resource allocation policy.
    Similar to ResourceManager in resource_manager.py
    """

    def __init__(self, total_resources: Resources):
        self.total_resources = total_resources
        print(
            f"\n[ResourcePolicy] Total resources: "
            f"{total_resources.cpu} CPU, {total_resources.memory} bytes memory"
        )

    def get_available_resources(self, operators: List[Operator]) -> Resources:
        """Calculate available resources."""
        used = Resources()
        for op in operators:
            used = used + op.current_resource_usage()

        return Resources(
            cpu=self.total_resources.cpu - used.cpu,
            memory=self.total_resources.memory - used.memory,
        )

    def can_dispatch(self, operator: Operator, operators: List[Operator]) -> bool:
        """
        Check if operator can dispatch a task given resource constraints.
        Similar to backpressure policy checks.
        """
        available = self.get_available_resources(operators)
        needed = operator.resource_per_task

        can_run = needed <= available

        # Note: We don't log here because this is called during operator selection
        # for ALL operators, not just the one that gets dispatched. Logging here
        # would be too verbose and confusing.

        return can_run


# ============================================================================
# Streaming Executor
# ============================================================================


class SimpleStreamingExecutor:
    """
    A simplified streaming executor that schedules tasks on operators.
    Demonstrates the core loop from StreamingExecutor._scheduling_loop_step()
    """

    def __init__(
        self, operators: List[Operator], resource_policy: StaticResourcePolicy
    ):
        self.operators = operators
        self.resource_policy = resource_policy

        # Initialize all operators
        for op in operators:
            op.initialize()

    def process_completed_tasks(self):
        """
        Process completed tasks and move outputs to queues.
        Similar to process_completed_tasks() in streaming_executor_state.py
        """
        for op in self.operators:
            if not op.state.active_tasks:
                continue

            # Collect all task generators
            task_refs = list(op.state.active_tasks.values())

            # Check which tasks have produced outputs (non-blocking)
            ready, _ = ray.wait(task_refs, num_returns=len(task_refs), timeout=0)

            # Process ready outputs
            for ref in ready:
                # Find which task this belongs to
                task_id = None
                for tid, task_ref in op.state.active_tasks.items():
                    if task_ref == ref:
                        task_id = tid
                        break

                if task_id is None:
                    continue

                try:
                    # Get the next block ObjectRef from the streaming generator
                    block_ref = next(ref)

                    # Add the ObjectRef to output queue (not the actual block!)
                    op.state.output_queue.append(block_ref)
                    print(
                        f"[{op.name}] Task {task_id} produced output block ref "
                        f"(output queue: {len(op.state.output_queue)})"
                    )

                except StopIteration:
                    # Task completed
                    print(f"[{op.name}] Task {task_id} completed")
                    del op.state.active_tasks[task_id]
                    op.state.completed_tasks += 1
                except Exception as e:
                    print(f"[{op.name}] Task {task_id} failed: {e}")
                    del op.state.active_tasks[task_id]

    def select_operator_to_run(self) -> Optional[Operator]:
        """
        Select which operator should dispatch the next task.
        Similar to select_operator_to_run() in streaming_executor_state.py
        """
        eligible_ops = []

        for op in self.operators:
            # Check if operator can dispatch
            if not op.should_dispatch():
                continue

            # Check resource constraints
            if not self.resource_policy.can_dispatch(op, self.operators):
                continue

            eligible_ops.append(op)

        if not eligible_ops:
            return None

        # Rank operators by current resource usage (prefer lower usage)
        # This implements a simple version of _rank_operators()
        ranked = sorted(eligible_ops, key=lambda op: op.current_resource_usage().memory)

        return ranked[0]

    def transfer_outputs(self):
        """
        Transfer outputs from one operator to the next.
        This simulates the output_queue -> input_queue flow.
        """
        for current_op, next_op in zip(self.operators, self.operators[1:]):

            # Transfer all available outputs
            while current_op.state.has_output():
                # Get the ObjectRef from the output queue
                block_ref = current_op.state.output_queue.popleft()
                # Pass the ObjectRef directly to the next operator (no materialization!)
                next_op.add_input([block_ref])

    def update_operator_states(self):
        """
        Update operator states after task completion.
        Similar to update_operator_states() in streaming_executor_state.py

        This marks downstream operators' inputs as done when upstream operators complete.
        """
        for current_op, next_op in zip(self.operators, self.operators[1:]):

            # If upstream operator is complete and has no more outputs,
            # mark the downstream operator's inputs as done
            if (
                current_op.is_completed()
                and not current_op.state.has_output()
                and not next_op.state.inputs_done
            ):
                next_op.mark_inputs_done()

    def scheduling_loop_step(self) -> bool:
        """
        Execute one step of the scheduling loop.
        This is the simplified version of _scheduling_loop_step()
        """
        print("\n" + "=" * 70)
        print("SCHEDULING STEP")
        print("=" * 70)

        # Phase 1: Process completed tasks
        print("\n[Phase 1] Processing completed tasks...")
        self.process_completed_tasks()

        # Transfer outputs between operators
        self.transfer_outputs()

        # Update operator states (mark downstream inputs as done when upstream completes)
        self.update_operator_states()

        # Phase 2: Dispatch new tasks
        # This loop continues dispatching until no more operators can run
        # due to resource constraints or lack of inputs
        print("\n[Phase 2] Dispatching new tasks...")
        dispatched_count = 0

        while True:
            # Select next operator to run (considers resources)
            op = self.select_operator_to_run()
            if op is None:
                # No more operators can run (either no inputs or insufficient resources)
                if dispatched_count > 0:
                    print(f"  Dispatched {dispatched_count} task(s)")
                    print(
                        "  (Stopped: no eligible operators - waiting for resources/inputs)"
                    )
                break

            op.dispatch_task()
            dispatched_count += 1

            # Note: Resource usage is recalculated on next iteration automatically
            # since can_dispatch() calls current_resource_usage() which computes fresh

        if dispatched_count == 0:
            print("  No tasks dispatched (no eligible operators)")

        # Phase 3: Print status
        print("\n[Phase 3] Current status:")
        for op in self.operators:
            print(
                f"  {op.name}: "
                f"input_queue={len(op.state.input_queue)}, "
                f"active_tasks={op.state.num_active_tasks()}, "
                f"output_queue={len(op.state.output_queue)}, "
                f"completed={op.state.completed_tasks}"
            )

        # Check if all operators are done
        all_done = all(op.is_completed() for op in self.operators)

        if all_done:
            print("\n✓ All operators completed!")

        return not all_done

    def run(self):
        """Run the executor until completion."""
        print("\n" + "=" * 70)
        print("STARTING STREAMING EXECUTION")
        print("=" * 70)

        step = 0
        max_steps = 50  # Safety limit to prevent infinite loops

        while step < max_steps:
            # Check if all operators are done before doing work
            if all(op.is_completed() for op in self.operators):
                print(f"\n✓ All operators completed after {step} scheduling steps!")
                break

            # Run one scheduling step
            continue_execution = self.scheduling_loop_step()
            step += 1

            if not continue_execution:
                print(f"\n✓ Execution finished after {step} scheduling steps!")
                break

            time.sleep(0.5)  # Small delay for readability
        else:
            # Only reached if we hit max_steps
            print(f"\n⚠ Reached maximum steps ({max_steps})!")

        print("\n" + "=" * 70)
        print("EXECUTION COMPLETED")
        print("=" * 70)

        # Collect final outputs as ObjectRefs (no materialization here)
        final_op = self.operators[-1]
        result_refs = list(final_op.state.output_queue)
        final_op.state.output_queue.clear()

        return result_refs


# ============================================================================
# Demo: Two-Operator Pipeline
# ============================================================================


def demo():
    """
    Create a simple two-operator pipeline:
    1. Operator 1: Multiply each number by 2
    2. Operator 2: Add 10 to each number
    """

    print("Initializing Ray...")
    ray.init()

    # Create input data
    print("\nCreating input data...")
    input_blocks = [
        Block(data=[1, 2, 3], block_id=0),
        Block(data=[4, 5, 6], block_id=1),
        Block(data=[7, 8, 9], block_id=2),
        Block(data=[10, 11, 12], block_id=3),
    ]

    print(f"Created {len(input_blocks)} input blocks")

    # Define transform functions
    def multiply_by_2(data: List[int]) -> List[int]:
        # Simulate some processing time
        time.sleep(0.1)
        return [x * 2 for x in data]

    def add_10(data: List[int]) -> List[int]:
        # Simulate some processing time
        time.sleep(0.1)
        return [x + 10 for x in data]

    # Create operators with resource constraints
    op1 = Operator(
        name="MultiplyOperator",
        transform_fn=multiply_by_2,
        resource_per_task=Resources(cpu=1.0, memory=1000),
        max_concurrency=2,  # Can run at most 2 tasks concurrently
    )

    op2 = Operator(
        name="AddOperator",
        transform_fn=add_10,
        resource_per_task=Resources(cpu=1.0, memory=1000),
        max_concurrency=2,
    )

    operators = [op1, op2]

    # Create resource policy (limit to 3 CPUs and 3000 bytes)
    resource_policy = StaticResourcePolicy(
        total_resources=Resources(cpu=3.0, memory=3000)
    )

    # Create executor (which will initialize operators)
    executor = SimpleStreamingExecutor(operators, resource_policy)

    # Add input blocks to first operator after initialization
    # Put blocks in object store and pass ObjectRefs
    for block in input_blocks:
        block_ref = ray.put(block)
        op1.add_input([block_ref])
    op1.mark_inputs_done()
    result_refs = executor.run()

    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    # Materialize only for display/verification
    blocks = ray.get(result_refs)
    for block in blocks:
        print(f"Block {block.block_id}: {block.data}")

    # Verify correctness
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    expected = [
        [12, 14, 16],  # (1,2,3) * 2 + 10
        [18, 20, 22],  # (4,5,6) * 2 + 10
        [24, 26, 28],  # (7,8,9) * 2 + 10
        [30, 32, 34],  # (10,11,12) * 2 + 10
    ]

    for i, (result, exp) in enumerate(zip(blocks, expected)):
        status = "✓" if result.data == exp else "✗"
        print(f"{status} Block {i}: got {result.data}, expected {exp}")

    ray.shutdown()


if __name__ == "__main__":
    demo()
