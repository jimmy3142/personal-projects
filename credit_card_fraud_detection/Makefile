.PHONY: setup-local-env setup-k8s-local destroy-k8s-local

setup-local-env:
	pipenv install
	pipenv install --dev
	pipenv shell

setup-k8s-local:
	minikube start --mount-string="./ml_models:/ml_models" --mount
	minikube image load credit-card-fraud-detect:v1.0.0
	kubectl apply -f k8s_local/app/namespace.yaml
	kubectl apply -f k8s_local/app/ -n credit-card-fraud-detect

destroy-k8s-local:
	minikube stop
	minikube delete --all


