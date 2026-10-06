"""Nav2 NavigateToPose lifecycle management.

Owner: C.
Contract: repository-root Workplan.md, section 4.5.

This module is the only owner of the ``navigate_to_pose`` action client. It
turns the callback-driven action into three calls the behaviour manager can
make from its timer without blocking::

    request_id = client.send(goal)
    progress = client.poll(request_id)   # interfaces.NavigationProgress
    client.cancel(request_id)

Each request keeps its own record, and every action callback is bound to the
id of the request that created it, so a late result for an old goal can never
change the state of a newer one.

Two Nav2 Humble behaviours, read from the installed
``nav2_util/simple_action_server.hpp`` and
``nav2_behavior_tree/bt_action_server_impl.hpp``, shape the rest:

* A goal sent while another is executing preempts it, and the preempted goal
  is *aborted* (``accept_pending_goal`` aborts the current handle). Our own
  change of plan would look exactly like a navigation failure. This client
  remembers which requests it superseded and reports them as ``CANCELED``, so
  exploration does not penalise a frontier for it.

* A cancelled behaviour tree ends with ``terminate_all``, which also
  terminates a goal that arrived while the cancel was being processed. So
  after a cancel, any new goal is held back until the cancelled goal's result
  has arrived.

Switching targets is therefore safest as a plain ``send`` of the new goal;
``cancel`` is for stopping with nothing to replace it.

Nav2 Humble's result message is empty and never says why a goal failed. The
``reason`` strings are this client's description of what it observed.

Timing uses the node's clock (simulation time under ``use_sim_time``).
Timeouts are evaluated on every ``send``, ``poll`` and ``cancel``, so the
client needs no timer of its own.
"""

from dataclasses import dataclass
from functools import partial
import threading

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle

from cave_explorer.interfaces import NavigationProgress, NavigationStatus

_RESULT_STATUS = {
    GoalStatus.STATUS_SUCCEEDED: NavigationStatus.SUCCEEDED,
    GoalStatus.STATUS_ABORTED: NavigationStatus.ABORTED,
    GoalStatus.STATUS_CANCELED: NavigationStatus.CANCELED,
}


def _sec_to_ns(seconds: float) -> int:
    return int(round(seconds * 1e9))


@dataclass
class _Request:
    """Everything known about one request. Guarded by ``NavigationClient._lock``."""

    goal: PoseStamped
    timeout_ns: int
    created_ns: int
    status: NavigationStatus = NavigationStatus.PENDING
    distance_remaining: float | None = None
    reason: str = ''
    dispatched_ns: int | None = None
    handle: ClientGoalHandle | None = None
    # True once the server can no longer be holding a live goal for this request.
    server_done: bool = False
    cancel_wanted_ns: int | None = None
    cancel_sent: bool = False
    superseded_by: int | None = None

    @property
    def cancel_in_flight(self) -> bool:
        return self.cancel_wanted_ns is not None and not self.server_done

    def finish(self, status: NavigationStatus, reason: str = '') -> bool:
        """Record a terminal status. The first verdict wins; returns whether it applied."""
        if self.status.is_terminal:
            return False
        self.status = status
        self.reason = reason
        return True


class NavigationClient:
    """Sole owner of the ``navigate_to_pose`` action client.

    Safe to call from any thread: every entry point and every action callback
    takes the same lock, so it works under single- or multi-threaded executors.
    """

    def __init__(self, node, action_name: str = 'navigate_to_pose', *,
                 goal_timeout_sec: float = 120.0,
                 server_wait_sec: float = 10.0,
                 cancel_grace_sec: float = 5.0,
                 history: int = 64,
                 callback_group=None,
                 clock=None):
        """Create the action client. Never blocks, even if Nav2 is not running yet.

        ``goal_timeout_sec`` bounds how long Nav2 may work on one goal; ``send``
        can override it per goal. ``server_wait_sec`` bounds how long a request
        may wait to be sent. ``cancel_grace_sec`` bounds how long a cancelled
        goal may hold back the next one if Nav2 never reports its result.
        ``clock`` defaults to the node's clock and exists so tests can drive time.
        """
        for name, value in (('goal_timeout_sec', goal_timeout_sec),
                            ('server_wait_sec', server_wait_sec),
                            ('cancel_grace_sec', cancel_grace_sec)):
            if value <= 0.0:
                raise ValueError(f'{name} must be positive, got {value}')

        self._action_name = action_name
        self._logger = node.get_logger()
        self._clock = clock if clock is not None else node.get_clock()
        self._client = ActionClient(node, NavigateToPose, action_name,
                                    callback_group=callback_group)
        self._goal_timeout_ns = _sec_to_ns(goal_timeout_sec)
        self._server_wait_ns = _sec_to_ns(server_wait_sec)
        self._cancel_grace_ns = _sec_to_ns(cancel_grace_sec)
        self._history = history

        self._lock = threading.RLock()
        self._requests: dict[int, _Request] = {}
        self._next_id = 1
        self._now_ns = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def server_ready(self) -> bool:
        """Whether the action server has been discovered."""
        return self._client.server_is_ready()

    def wait_for_server(self, timeout_sec: float) -> bool:
        """Block up to ``timeout_sec`` for the server. Start-up only - never the main loop."""
        return self._client.wait_for_server(timeout_sec=timeout_sec)

    def send(self, goal: PoseStamped, timeout_sec: float | None = None) -> int:
        """Queue a goal and return its request id immediately.

        Problems never raise: a goal that cannot be sent becomes an ``ERROR``
        that ``poll`` reports, so the main loop keeps running.
        """
        with self._lock:
            self._tick()
            request_id = self._next_id
            self._next_id += 1
            timeout_ns = (self._goal_timeout_ns if timeout_sec is None
                          else _sec_to_ns(timeout_sec))
            request = _Request(goal=goal, timeout_ns=timeout_ns, created_ns=self._now_ns)
            self._requests[request_id] = request

            if not goal.header.frame_id:
                self._settle_unsent(request_id, request, NavigationStatus.ERROR,
                                    'goal has no header.frame_id')
            self._dispatch()
            self._prune()
            return request_id

    def cancel(self, request_id: int) -> None:
        """Ask Nav2 to stop a request.

        The status stays ``ACTIVE`` until Nav2 reports back, because a goal can
        still finish before the cancel lands. Superseded requests are already
        being stopped by preemption and are left alone.
        """
        with self._lock:
            self._tick()
            request = self._requests.get(request_id)
            if (request is None or request.status.is_terminal
                    or request.superseded_by is not None):
                return
            if request.dispatched_ns is None:
                self._settle_unsent(request_id, request, NavigationStatus.CANCELED,
                                    'cancelled before it was sent to Nav2')
                return
            self._request_cancel(request)

    def poll(self, request_id: int) -> NavigationProgress:
        """Report one request's current state. Unknown ids are an ``ERROR``, not an exception."""
        with self._lock:
            self._tick()
            request = self._requests.get(request_id)
            if request is None:
                return NavigationProgress(NavigationStatus.ERROR,
                                          reason=f'unknown request id {request_id}')
            return NavigationProgress(request.status, request.distance_remaining,
                                      request.reason)

    # ------------------------------------------------------------------
    # Time-driven bookkeeping. Every method below runs with the lock held.
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = self._clock.now().nanoseconds
        if self._now_ns is not None and now < self._now_ns:
            self._abandon_all('simulation time went backwards; request abandoned')
        self._now_ns = now

        for request_id, request in list(self._requests.items()):
            if (request.dispatched_ns is not None and not request.server_done
                    and now - request.dispatched_ns > request.timeout_ns):
                timeout_sec = request.timeout_ns / 1e9
                if self._settle(request_id, request, NavigationStatus.ERROR,
                                f'timed out after {timeout_sec:g} s'):
                    # A superseded goal is already being stopped by preemption,
                    # and cancelling it could take the new goal down with it.
                    if request.superseded_by is None:
                        self._request_cancel(request)
            if request.cancel_in_flight and now - request.cancel_wanted_ns > self._cancel_grace_ns:
                grace_sec = self._cancel_grace_ns / 1e9
                self._logger.warn(
                    f'navigation request {request_id}: no result {grace_sec:g} s after '
                    'cancelling; no longer holding back new goals for it')
                request.server_done = True

        self._dispatch()

    def _dispatch(self) -> None:
        """Send the newest waiting request if nothing is blocking it."""
        waiting = [request_id for request_id, request in self._requests.items()
                   if request.dispatched_ns is None and not request.status.is_terminal]
        if not waiting:
            return
        newest = max(waiting)
        for request_id in waiting:
            if request_id != newest:
                self._settle_unsent(request_id, self._requests[request_id],
                                    NavigationStatus.CANCELED,
                                    f'superseded by request {newest} before it was sent to Nav2')

        request = self._requests[newest]
        blocker = next((request_id for request_id, other in self._requests.items()
                        if other.cancel_in_flight), None)
        if blocker is None and self._client.server_is_ready():
            self._send_to_server(newest, request)
            return

        if self._now_ns - request.created_ns > self._server_wait_ns:
            if blocker is not None:
                reason = f'held back by the cancellation of request {blocker} for too long'
            else:
                reason = f'{self._action_name} server not available'
            self._settle_unsent(newest, request, NavigationStatus.ERROR, reason)

    def _send_to_server(self, request_id: int, request: _Request) -> None:
        # Nav2 runs one NavigateToPose at a time: whatever it is still working
        # on will be preempted by this goal.
        for other_id, other in self._requests.items():
            if (other_id != request_id and other.dispatched_ns is not None
                    and not other.server_done and other.superseded_by is None):
                other.superseded_by = request_id

        request.dispatched_ns = self._now_ns
        action_goal = NavigateToPose.Goal()
        action_goal.pose = request.goal
        future = self._client.send_goal_async(
            action_goal, feedback_callback=partial(self._on_feedback, request_id))
        future.add_done_callback(partial(self._on_goal_response, request_id))

        position = request.goal.pose.position
        self._logger.info(
            f'navigation request {request_id}: sent goal '
            f'({position.x:.2f}, {position.y:.2f}) in {request.goal.header.frame_id}')

    def _request_cancel(self, request: _Request) -> None:
        if request.cancel_wanted_ns is None:
            request.cancel_wanted_ns = self._now_ns
        # Without a handle yet, the cancel goes out when the goal response arrives.
        if request.handle is not None and not request.cancel_sent and not request.server_done:
            request.cancel_sent = True
            request.handle.cancel_goal_async()

    def _abandon_all(self, reason: str) -> None:
        for request_id, request in self._requests.items():
            self._settle(request_id, request, NavigationStatus.ERROR, reason)
            request.server_done = True

    def _settle(self, request_id: int, request: _Request,
                status: NavigationStatus, reason: str = '') -> bool:
        """Apply a terminal verdict and log it once."""
        if not request.finish(status, reason):
            return False
        message = f'navigation request {request_id}: {status.value}'
        if reason:
            message += f' ({reason})'
        if status in (NavigationStatus.SUCCEEDED, NavigationStatus.CANCELED):
            self._logger.info(message)
        else:
            self._logger.warn(message)
        return True

    def _settle_unsent(self, request_id: int, request: _Request,
                       status: NavigationStatus, reason: str) -> None:
        self._settle(request_id, request, status, reason)
        request.server_done = True

    def _prune(self) -> None:
        """Forget the oldest finished requests beyond ``history``."""
        excess = len(self._requests) - self._history
        for request_id in sorted(self._requests):
            if excess <= 0:
                return
            request = self._requests[request_id]
            if request.status.is_terminal and request.server_done:
                del self._requests[request_id]
                excess -= 1

    # ------------------------------------------------------------------
    # Action callbacks. Each is bound to the id of the request that sent it.
    # ------------------------------------------------------------------

    def _on_goal_response(self, request_id: int, future) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return
            exception = future.exception()
            if exception is not None:
                self._settle(request_id, request, NavigationStatus.ERROR,
                             f'sending the goal failed: {exception}')
                request.server_done = True
            elif not future.result().accepted:
                self._settle(request_id, request, NavigationStatus.REJECTED,
                             f'goal rejected by {self._action_name}')
                request.server_done = True
            else:
                request.handle = future.result()
                if request.status is NavigationStatus.PENDING:
                    request.status = NavigationStatus.ACTIVE
                request.handle.get_result_async().add_done_callback(
                    partial(self._on_result, request_id))
                if request.cancel_wanted_ns is not None:
                    self._request_cancel(request)
            self._tick()

    def _on_result(self, request_id: int, future) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return
            request.server_done = True
            exception = future.exception()
            if exception is not None:
                self._settle(request_id, request, NavigationStatus.ERROR,
                             f'getting the result failed: {exception}')
            else:
                code = future.result().status
                status = _RESULT_STATUS.get(code)
                if status is None:
                    self._settle(request_id, request, NavigationStatus.ERROR,
                                 f'unexpected result status {code}')
                elif (request.superseded_by is not None
                        and status is not NavigationStatus.SUCCEEDED):
                    self._settle(request_id, request, NavigationStatus.CANCELED,
                                 f'superseded by request {request.superseded_by}')
                elif status is NavigationStatus.CANCELED:
                    who = ('at our request' if request.cancel_wanted_ns is not None
                           else 'by another client, e.g. the RViz Nav2 panel')
                    self._settle(request_id, request, status, f'cancelled {who}')
                elif status is NavigationStatus.ABORTED:
                    self._settle(request_id, request, status,
                                 f'aborted by {self._action_name}')
                else:
                    self._settle(request_id, request, status)
            self._tick()

    def _on_feedback(self, request_id: int, feedback_msg) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status.is_terminal:
                return
            # Feedback travels on a topic and can overtake the goal response,
            # but Nav2 only publishes it for a goal it has accepted - so it is
            # proof enough that the goal is active.
            if request.status is NavigationStatus.PENDING:
                request.status = NavigationStatus.ACTIVE
            distance = float(feedback_msg.feedback.distance_remaining)
            request.distance_remaining = max(0.0, distance)
