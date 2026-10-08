"""Tests for navigation_client against a scriptable stand-in for Nav2.

Each case reproduces a way the template's action handling fails, or a Nav2
behaviour the client has to survive. The stub runs a real rclpy ActionServer
on its own executor thread, so the futures, goal handles and status codes are
the genuine article. The client node is spun from the test thread, the way the
main node's timer drives it in production.

The stub uses a unique action name per test so a simulation running on the
same machine can never receive these goals.
"""

import itertools
import os
import threading
import time
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import pytest
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node

from cave_explorer.interfaces import NavigationStatus
from cave_explorer.navigation_client import NavigationClient

_names = itertools.count()


def _unique(prefix):
    return f'{prefix}_{os.getpid()}_{next(_names)}'


def _goal(label, frame_id='map'):
    """A goal whose label travels in pose.position.x, so the stub can script it."""
    goal = PoseStamped()
    goal.header.frame_id = frame_id
    goal.pose.position.x = float(label)
    goal.pose.orientation.w = 1.0
    return goal


def _label(goal_request):
    return int(goal_request.pose.pose.position.x)


class FakeClock:
    """Simulation time that moves only when a test moves it."""

    def __init__(self):
        self.ns = 1_000 * 10**9

    def now(self):
        return SimpleNamespace(nanoseconds=self.ns)

    def advance(self, seconds):
        self.ns += int(seconds * 1e9)


class StubNav2(Node):
    """Scriptable navigate_to_pose server.

    ``script`` maps a goal label to ``succeed`` or ``abort`` (after one feedback
    message), ``reject``, or ``hold`` (keep executing until ``release`` or a
    cancel). Unscripted labels succeed.

    Like Nav2's SimpleActionServer, a goal that starts executing aborts any goal
    still executing. Clearing ``cancel_gate`` makes the server sit on cancel
    requests - the window in which real Nav2 would terminate a pending goal.
    """

    def __init__(self, script, action_name):
        super().__init__(_unique('stub_navigate_to_pose'))
        self.script = script
        self.received = []
        self.outcomes = {}
        self.cancel_gate = threading.Event()
        self.cancel_gate.set()
        self.stopping = threading.Event()
        self._released = {}
        self._executing = set()
        self._preempted = set()
        self._lock = threading.Lock()
        self._server = ActionServer(
            self, NavigateToPose, action_name,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup())

    def release(self, label, outcome):
        with self._lock:
            self._released[label] = outcome

    def wait_idle(self, timeout=2.0):
        """Wait for every execute loop to return, so teardown never races them."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._executing:
                    break
            time.sleep(0.01)
        time.sleep(0.05)   # let rclpy send the final result before the node goes away

    def _on_goal(self, goal_request):
        label = _label(goal_request)
        with self._lock:
            self.received.append(label)
        if self.script.get(label) == 'reject':
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        label = _label(goal_handle.request)
        behaviour = self.script.get(label, 'succeed')
        with self._lock:
            self._preempted |= self._executing
            self._executing.add(label)
        distance = 10.0
        try:
            while True:
                feedback = NavigateToPose.Feedback()
                feedback.distance_remaining = distance
                goal_handle.publish_feedback(feedback)
                distance = max(0.0, distance - 0.5)

                with self._lock:
                    preempted = label in self._preempted
                    released = self._released.get(label)
                if preempted or self.stopping.is_set():
                    return self._finish(goal_handle, label, 'aborted')
                if goal_handle.is_cancel_requested and self.cancel_gate.is_set():
                    return self._finish(goal_handle, label, 'canceled')
                outcome = released if behaviour == 'hold' else behaviour
                if outcome == 'succeed':
                    return self._finish(goal_handle, label, 'succeeded')
                if outcome == 'abort':
                    return self._finish(goal_handle, label, 'aborted')
                time.sleep(0.02)
        finally:
            with self._lock:
                self._executing.discard(label)

    def _finish(self, goal_handle, label, outcome):
        {'succeeded': goal_handle.succeed,
         'aborted': goal_handle.abort,
         'canceled': goal_handle.canceled}[outcome]()
        with self._lock:
            self.outcomes[label] = outcome
        return NavigateToPose.Result()


class Harness:
    """One stub server, one client node, and a fake clock for the client."""

    def __init__(self, script):
        self.action_name = _unique('test_navigate_to_pose')
        self.stub = StubNav2(script, self.action_name)
        self._server_executor = MultiThreadedExecutor(num_threads=6)
        self._server_executor.add_node(self.stub)
        self._server_thread = threading.Thread(target=self._server_executor.spin, daemon=True)
        self._server_thread.start()

        self.clock = FakeClock()
        self.node = Node(_unique('navigation_client_test'))
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self.client = self.make_client(self.action_name)
        assert self.spin_until(self.client.server_ready), 'stub server never appeared'

    def make_client(self, action_name):
        return NavigationClient(self.node, action_name, clock=self.clock,
                                goal_timeout_sec=30.0, server_wait_sec=10.0,
                                cancel_grace_sec=5.0)

    def spin_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._executor.spin_once(timeout_sec=0.02)
            if predicate():
                return True
        return False

    def spin_for(self, seconds):
        self.spin_until(lambda: False, timeout=seconds)

    def status(self, request_id):
        return self.client.poll(request_id).status

    def wait_status(self, request_id, status, timeout=5.0):
        return self.spin_until(lambda: self.status(request_id) is status, timeout)

    def close(self):
        self.stub.stopping.set()
        self.stub.cancel_gate.set()
        self.stub.wait_idle()
        self._server_executor.shutdown(timeout_sec=2.0)
        self._server_thread.join(timeout=2.0)
        self._executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()
        self.stub.destroy_node()


@pytest.fixture
def harness():
    rclpy.init()
    created = []

    def make(script=None):
        created.append(Harness(script or {}))
        return created[-1]

    yield make
    for each in created:
        each.close()
    rclpy.shutdown()


# ----------------------------------------------------------------------
# The five template defects
# ----------------------------------------------------------------------

def test_rejected_goal_is_terminal_and_does_not_block_the_next(harness):
    """Template: a rejection left ready_for_next_goal_ False, so the robot never moved again."""
    h = harness({0: 'reject', 1: 'succeed'})
    rejected = h.client.send(_goal(0))
    assert h.wait_status(rejected, NavigationStatus.REJECTED)
    assert 'rejected' in h.client.poll(rejected).reason

    accepted = h.client.send(_goal(1))
    assert h.wait_status(accepted, NavigationStatus.SUCCEEDED)


def test_aborted_goal_is_not_reported_as_arrival(harness):
    """Template: goal_reached_callback read .result and threw .status away."""
    h = harness({0: 'abort'})
    request_id = h.client.send(_goal(0))
    assert h.wait_status(request_id, NavigationStatus.ABORTED)
    progress = h.client.poll(request_id)
    assert progress.is_terminal and not progress.succeeded


def test_superseded_goal_is_cancelled_not_failed_and_leaves_the_new_goal_alone(harness):
    """Template: one shared future per client, so an old result clobbered the new goal.

    Nav2 aborts the goal it preempts. Reported raw, exploration would penalise
    a frontier for our own change of plan.
    """
    h = harness({0: 'hold', 1: 'hold'})
    old = h.client.send(_goal(0))
    assert h.wait_status(old, NavigationStatus.ACTIVE)

    new = h.client.send(_goal(1))
    assert h.wait_status(new, NavigationStatus.ACTIVE)
    assert h.spin_until(lambda: h.stub.outcomes.get(0) == 'aborted')   # Nav2's verdict
    assert h.wait_status(old, NavigationStatus.CANCELED)                # ours
    assert 'superseded' in h.client.poll(old).reason

    h.spin_for(0.3)
    assert h.status(new) is NavigationStatus.ACTIVE
    h.stub.release(1, 'succeed')
    assert h.wait_status(new, NavigationStatus.SUCCEEDED)


def test_feedback_distance_is_kept_not_discarded(harness):
    """Template: feedback_callback only logged distance_remaining."""
    h = harness({0: 'hold'})
    request_id = h.client.send(_goal(0))
    assert h.spin_until(lambda: h.client.poll(request_id).distance_remaining is not None)
    assert h.status(request_id) is NavigationStatus.ACTIVE

    h.stub.release(0, 'succeed')
    assert h.wait_status(request_id, NavigationStatus.SUCCEEDED)
    assert h.client.poll(request_id).succeeded


def test_missing_server_neither_blocks_nor_hangs(harness):
    """Template: wait_for_server() had no timeout, so __init__ froze without Nav2."""
    h = harness()
    started = time.monotonic()
    client = h.make_client(_unique('no_such_navigate_to_pose'))
    request_id = client.send(_goal(0))
    assert time.monotonic() - started < 1.0
    assert client.poll(request_id).status is NavigationStatus.PENDING

    h.clock.advance(11.0)
    progress = client.poll(request_id)
    assert progress.status is NavigationStatus.ERROR
    assert 'not available' in progress.reason


# ----------------------------------------------------------------------
# Cancellation and timeouts
# ----------------------------------------------------------------------

def test_cancel_stops_the_goal_on_the_server(harness):
    h = harness({0: 'hold'})
    request_id = h.client.send(_goal(0))
    assert h.wait_status(request_id, NavigationStatus.ACTIVE)

    h.client.cancel(request_id)
    assert h.wait_status(request_id, NavigationStatus.CANCELED)
    assert 'at our request' in h.client.poll(request_id).reason
    assert h.stub.outcomes[0] == 'canceled'


def test_timeout_reports_error_and_stops_the_goal_on_the_server(harness):
    h = harness({0: 'hold'})
    request_id = h.client.send(_goal(0), timeout_sec=20.0)
    assert h.wait_status(request_id, NavigationStatus.ACTIVE)

    h.clock.advance(21.0)
    progress = h.client.poll(request_id)
    assert progress.status is NavigationStatus.ERROR
    assert 'timed out' in progress.reason

    assert h.spin_until(lambda: h.stub.outcomes.get(0) == 'canceled')
    # Nav2's later CANCELED result does not overwrite the timeout verdict.
    assert h.status(request_id) is NavigationStatus.ERROR


def test_new_goal_waits_until_a_cancelled_goal_has_finished(harness):
    """Nav2 ends a cancelled tree with terminate_all, killing any goal sent meanwhile."""
    h = harness({0: 'hold', 1: 'succeed'})
    old = h.client.send(_goal(0))
    assert h.wait_status(old, NavigationStatus.ACTIVE)

    h.stub.cancel_gate.clear()
    h.client.cancel(old)
    new = h.client.send(_goal(1))
    h.spin_for(0.5)
    assert h.stub.received == [0]
    assert h.status(new) is NavigationStatus.PENDING

    h.stub.cancel_gate.set()
    assert h.wait_status(old, NavigationStatus.CANCELED)
    assert h.wait_status(new, NavigationStatus.SUCCEEDED)
    assert h.stub.received == [0, 1]


def test_held_goals_collapse_to_the_newest_and_can_be_cancelled_unsent(harness):
    h = harness({0: 'hold', 1: 'succeed', 2: 'succeed', 3: 'succeed'})
    old = h.client.send(_goal(0))
    assert h.wait_status(old, NavigationStatus.ACTIVE)
    h.stub.cancel_gate.clear()
    h.client.cancel(old)

    stale = h.client.send(_goal(1))
    dropped = h.client.send(_goal(2))
    assert h.status(stale) is NavigationStatus.CANCELED
    assert 'superseded' in h.client.poll(stale).reason
    h.client.cancel(dropped)
    assert h.status(dropped) is NavigationStatus.CANCELED

    fresh = h.client.send(_goal(3))
    h.stub.cancel_gate.set()
    assert h.wait_status(fresh, NavigationStatus.SUCCEEDED)
    assert h.stub.received == [0, 3]


# ----------------------------------------------------------------------
# Inputs the main loop must survive without an exception
# ----------------------------------------------------------------------

def test_unknown_request_id_is_an_error_not_an_exception(harness):
    h = harness()
    progress = h.client.poll(9999)
    assert progress.status is NavigationStatus.ERROR
    assert 'unknown' in progress.reason


def test_goal_without_a_frame_never_reaches_nav2(harness):
    h = harness()
    request_id = h.client.send(_goal(0, frame_id=''))
    progress = h.client.poll(request_id)
    assert progress.status is NavigationStatus.ERROR
    assert 'frame_id' in progress.reason
    h.spin_for(0.2)
    assert h.stub.received == []


def test_simulation_time_reset_abandons_live_requests(harness):
    h = harness({0: 'hold'})
    request_id = h.client.send(_goal(0))
    assert h.wait_status(request_id, NavigationStatus.ACTIVE)

    h.clock.advance(-50.0)
    progress = h.client.poll(request_id)
    assert progress.status is NavigationStatus.ERROR
    assert 'backwards' in progress.reason
