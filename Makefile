.PHONY: help setup cluster deploy clean

# Variables
PYTHON_VERSION := 3.9
CLUSTER_NAME := churn-prediction-eks
REGION := us-east-2
ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text)
ECR_REPO := $(ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/churn-api
NAMESPACE := production

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Setup Python environment
	python$(PYTHON_VERSION) -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt
	pre-commit install

download-data: ## Download IBM Telco dataset
	mkdir -p data/raw
	wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv \
		-O data/raw/Telco-Customer-Churn.csv
	@echo "Dataset downloaded: 7,043 rows, 21 features"

local-k8s: ## Start local Kubernetes
	minikube start --cpus 4 --memory 8192 --driver docker --kubernetes-version=v1.28.0
	minikube addons enable metrics-server
	minikube addons enable ingress
	@echo "Minikube started. Run: eval $$(minikube docker-env)"

build: ## Build Docker image
	docker build -t churn-api:latest .
	docker tag churn-api:latest $(ECR_REPO):latest

push: ## Push to ECR
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(ECR_REPO)
	aws ecr create-repository --repository-name churn-api --region $(REGION) || true
	docker push $(ECR_REPO):latest

create-cluster: ## Create EKS cluster
	eksctl create cluster \
		--name $(CLUSTER_NAME) \
		--region $(REGION) \
		--nodegroup-name workers \
		--node-type t3.medium \
		--nodes 2 \
		--nodes-min 2 \
		--nodes-max 4 \
		--managed \
		--version 1.28
	kubectl config use-context $(CLUSTER_NAME)

deploy-local: ## Deploy to Minikube
	kubectl apply -k k8s/overlays/dev/
	kubectl wait --for=condition=available --timeout=300s deployment/churn-api -n $(NAMESPACE)
	@echo "Access via: kubectl port-forward -n $(NAMESPACE) svc/churn-api 8000:8000"

deploy-eks: ## Deploy to EKS
	kubectl apply -k k8s/overlays/prod/
	kubectl wait --for=condition=available --timeout=300s deployment/churn-api -n $(NAMESPACE)
	kubectl get ingress -n $(NAMESPACE)

scale-down: ## Scale EKS to zero (save costs)
	kubectl scale deployment --all --replicas=0 -n $(NAMESPACE)
	eksctl scale nodegroup --cluster=$(CLUSTER_NAME) --nodes=0 workers

scale-up: ## Scale EKS back up
	eksctl scale nodegroup --cluster=$(CLUSTER_NAME) --nodes=2 workers
	kubectl scale deployment churn-api --replicas=2 -n $(NAMESPACE)

delete-cluster: ## Delete EKS cluster (IMPORTANT for cost)
	eksctl delete cluster --name $(CLUSTER_NAME) --region $(REGION)

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html

load-test: ## Run load test
	kubectl run -i --tty load-test --rm --image=williamyeh/wrk --restart=Never -- \
		-c 50 -d 60s -t 4 --latency http://churn-api.$(NAMESPACE).svc.cluster.local:8000/predict

monitor: ## Open monitoring dashboards
	@echo "Grafana: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
	@echo "Prometheus: kubectl port-forward -n monitoring svc/prometheus-server 9090:80"
	open http://localhost:3000

cost-check: ## Check AWS costs
	aws ce get-cost-and-usage \
		--time-period Start=$$(date -d '7 days ago' +%Y-%m-%d),End=$$(date +%Y-%m-%d) \
		--granularity DAILY \
		--metrics UnblendedCost \
		--group-by Type=DIMENSION,Key=SERVICE \
		--filter file://configs/cost-filter.json

clean: ## Clean up everything
	kubectl delete -k k8s/overlays/prod/ --ignore-not-found=true
	docker system prune -f
	rm -rf venv/ .pytest_cache/ htmlcov/