"""
Thread Management Utilities for GUI Operations
"""

import threading
import time
from typing import Callable, Any, Optional, Dict, Tuple
from core.utils import get_logger

logger = get_logger(__name__)


class CancellableThread:
    """A thread that supports cancellation and proper cleanup."""

    def __init__(self, target: Callable, args: tuple = (), kwargs: Optional[Dict] = None,
                 name: Optional[str] = None, daemon: bool = True):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.name = name or f"CancellableThread-{id(self)}"
        self.daemon = daemon

        self.thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.finished_event = threading.Event()
        self.result = None
        self.exception = None
        self.is_running = False

    def start(self):
        """Start the thread."""
        if self.thread and self.thread.is_alive():
            logger.warning(f"Thread {self.name} is already running")
            return

        self.cancel_event.clear()
        self.finished_event.clear()
        self.is_running = True

        self.thread = threading.Thread(
            target=self._run_wrapper,
            name=self.name,
            daemon=self.daemon
        )
        self.thread.start()
        logger.debug(f"Started thread: {self.name}")

    def _run_wrapper(self):
        """Wrapper that handles execution and cleanup."""
        try:
            # Check for cancellation before starting
            if self.cancel_event.is_set():
                logger.debug(f"Thread {self.name} cancelled before execution")
                return

            # Execute the target function
            self.result = self.target(*self.args, **self.kwargs)
            logger.debug(f"Thread {self.name} completed successfully")

        except Exception as e:
            self.exception = e
            logger.error(f"Thread {self.name} failed: {e}")

        finally:
            self.is_running = False
            self.finished_event.set()

    def cancel(self, timeout: float = 5.0):
        """Cancel the thread and wait for it to finish."""
        if not self.thread or not self.thread.is_alive():
            return True

        logger.debug(f"Cancelling thread: {self.name}")
        self.cancel_event.set()

        # Wait for thread to finish
        self.thread.join(timeout)

        if self.thread.is_alive():
            logger.warning(f"Thread {self.name} did not finish within {timeout}s")
            return False

        logger.debug(f"Thread {self.name} cancelled successfully")
        return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for thread to finish. Returns True if finished, False if timeout."""
        if not self.thread:
            return True

        return self.finished_event.wait(timeout)

    def is_cancelled(self) -> bool:
        """Check if thread was cancelled."""
        return self.cancel_event.is_set()

    def is_alive(self) -> bool:
        """Check if thread is still running."""
        return bool(self.thread and self.thread.is_alive())


class ThreadManager:
    """Manager for background threads with automatic cleanup."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.active_threads: Dict[str, CancellableThread] = {}
        self.lock = threading.RLock()
        self._shutdown = False

        # Cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def submit_task(self, task_id: str, target: Callable, *args, **kwargs) -> Optional[CancellableThread]:
        """Submit a cancellable task."""
        with self.lock:
            if self._shutdown:
                logger.warning("ThreadManager is shutting down")
                return None

            # Cancel existing task with same ID
            if task_id in self.active_threads:
                self.cancel_task(task_id)

            # Create and start new thread
            thread = CancellableThread(target, args, kwargs, name=f"{task_id}")
            self.active_threads[task_id] = thread
            thread.start()

            return thread

    def cancel_task(self, task_id: str, timeout: float = 3.0) -> bool:
        """Cancel a specific task."""
        with self.lock:
            thread = self.active_threads.get(task_id)
            if not thread:
                return True

            success = thread.cancel(timeout)
            if success:
                del self.active_threads[task_id]

            return success

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for a task to complete."""
        with self.lock:
            thread = self.active_threads.get(task_id)
            if not thread:
                return True

            return thread.wait(timeout)

    def get_task_result(self, task_id: str) -> Tuple[Any, Optional[Exception]]:
        """Get task result and exception. Returns (result, exception)."""
        with self.lock:
            thread = self.active_threads.get(task_id)
            if not thread:
                return None, None

            return thread.result, thread.exception

    def is_task_running(self, task_id: str) -> bool:
        """Check if a task is currently running."""
        with self.lock:
            thread = self.active_threads.get(task_id)
            return bool(thread and thread.is_running)

    def cancel_all(self, timeout: float = 5.0):
        """Cancel all active tasks."""
        with self.lock:
            task_ids = list(self.active_threads.keys())

        for task_id in task_ids:
            self.cancel_task(task_id, timeout)

    def _cleanup_worker(self):
        """Background worker to clean up finished threads."""
        while not self._shutdown:
            time.sleep(1.0)  # Check every second

            with self.lock:
                # Remove finished threads
                finished_tasks = [
                    task_id for task_id, thread in self.active_threads.items()
                    if not thread.is_alive() and thread.finished_event.is_set()
                ]

                for task_id in finished_tasks:
                    del self.active_threads[task_id]
                    logger.debug(f"Cleaned up finished task: {task_id}")

    def shutdown(self, timeout: float = 10.0):
        """Shutdown the thread manager."""
        logger.info("Shutting down ThreadManager...")
        self._shutdown = True

        # Cancel all active tasks
        self.cancel_all(timeout)

        logger.info("ThreadManager shutdown complete")


class BackgroundTaskRunner:
    """High-level interface for running background tasks with UI callbacks."""

    def __init__(self, thread_manager: ThreadManager):
        self.thread_manager = thread_manager
        self.ui_callbacks: Dict[str, Callable] = {}

    def run_task(self, task_id: str, task_func: Callable, 
                 on_complete: Optional[Callable] = None,
                 on_error: Optional[Callable] = None,
                 *args, **kwargs):
        """Run a background task with completion callbacks."""

        def wrapped_task():
            try:
                result = task_func(*args, **kwargs)
                if on_complete:
                    self._schedule_ui_callback(on_complete, result)
                return result
            except Exception as e:
                logger.error(f"Background task {task_id} failed: {e}")
                if on_error:
                    self._schedule_ui_callback(on_error, e)
                raise

        return self.thread_manager.submit_task(task_id, wrapped_task)

    def _schedule_ui_callback(self, callback: Callable, *args):
        """Schedule a callback to run on the UI thread."""
        # This would need to be called from the main thread context
        # The callback should handle scheduling to UI thread
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"UI callback failed: {e}")

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        return self.thread_manager.cancel_task(task_id)


# Global thread manager instance
_thread_manager = None

def get_thread_manager() -> ThreadManager:
    """Get the global thread manager instance."""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager

def shutdown_thread_manager():
    """Shutdown the global thread manager."""
    global _thread_manager
    if _thread_manager:
        _thread_manager.shutdown()
        _thread_manager = None
