fix:
	uv run ruff check --select I --fix .
	uv run ruff format

test:
	-docker rm -f qdrant-test >/dev/null 2>&1 || true
	docker run -d --name qdrant-test -p 6333:6333 qdrant/qdrant >/dev/null
	sleep 1

	uv run test.py

	docker rm -f qdrant-test >/dev/null