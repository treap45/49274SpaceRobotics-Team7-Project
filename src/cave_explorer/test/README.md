# Module tests

This directory is a placeholder for tests described in the repository-root
Workplan.md. No test cases have been implemented yet.

- A: detection coordinates, empty detections, image metadata and collection.
- B: invalid depth, timestamp/TF matching, stable track IDs and inspection goals.
- C: frontier selection, rejected/canceled goals, stale callbacks and retry limits.
- All: integration from detection through inspection and resumed exploration.

Add tests with filenames such as `test_perception.py` as modules are implemented.
Use controlled inputs for module tests and separate simulation runs for end-to-end
validation. Do not count placeholder files as passing functional tests.
