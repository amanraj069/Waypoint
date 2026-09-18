"""The HTTP face of waypoint: the same pipeline the CLI runs, over a socket.

`api.main:app` is a FastAPI application. It owns no simulation logic of its own
-- planning, verification, flight and run files all come from the packages the
CLI uses -- so anything the web app can do, `scripts/explore.py` can still do,
and the two cannot drift apart.
"""
