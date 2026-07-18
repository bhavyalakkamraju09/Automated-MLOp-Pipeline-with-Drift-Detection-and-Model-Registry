.PHONY: up down train monitor logs repro clean

up:
	docker-compose up -d
	@echo "Airflow:  http://localhost:8080 (airflow/airflow)"
	@echo "MLflow:   http://localhost:5000"
	@echo "Grafana:  http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

down:
	docker-compose down

repro:
	dvc repro

train:
	python src/training/train.py

tune:
	python src/training/tune.py

monitor:
	python src/monitoring/drift_detector.py

logs:
	docker-compose logs -f --tail=100

clean:
	docker-compose down -v
	rm -rf data/processed/* models/*.pkl metrics/*.json monitoring/reports/*.html
