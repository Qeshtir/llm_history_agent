test:
	docker-compose run -e PYTHONPATH=/app bot pytest -v tests/ 