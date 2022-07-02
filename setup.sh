cd Fraud_part2
#docker image build . -t anastasiatoullec/image_pipeline_fraud:latest
#docker push anastasiatoullec/image_pipeline_fraud:latest
docker image pull anastasiatoullec/image_pipeline_fraud:latest
docker network create -d bridge net
#docker container run -d --rm  --network net -p 8000:8000 --name api_container anastasiatoullec/image_pipeline_fraud:latest

cd Fraud_part2/docker_test1
docker image build . -t image_tests_fraud:latest
#docker container run --rm  --network net --name container_tests_fraud image_tests_fraud:latest
cd

cd Fraud_part2/docker_test2
docker image build . -t image_tests_not_fraud:latest
#docker container run --rm  --network net --name container_tests_not_fraud image_tests_not_fraud:latest
cd

cd Fraud_part2/docker_test3
docker image build . -t image_tests_performances:latest
#docker container run --rm  --network net --name container_tests_performances image_tests_performances:latest
cd

cd Fraud_part2
#docker container stop api_container
docker-compose up

#Déploiement K8
kubectl create -f ./k8/fraud-analysis-deployment.yml
kubectl create -f ./k8/fraud-analysis-service.yml
minikube addons enable ingress
kubectl create -f ./k8/fraud-analysis-ingress.yml