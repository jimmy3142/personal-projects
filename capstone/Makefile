.PHONY: setup-local-env setup-local-k8s destroy-local-k8s

setup-local-env:
	pipenv install
	pipenv install --dev
	pipenv shell

setup-local-k8s:
	minikube start --mount-string="./ml_models:/ml_models" --mount
	minikube image load mlflow-server:v1.0.0
	minikube image load credit-card-fraud-detect:v1.0.0
	kubectl apply -f k8s/app/namespace.yaml
	kubectl apply -f k8s/app/ -n credit-card-fraud-detect

destroy-local-k8s:
	minikube stop
	minikube delete --all


