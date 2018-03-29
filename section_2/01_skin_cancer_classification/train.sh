cd tensorflow
python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=../bottlenecks \
--how_many_training_steps 4000 \
--model_dir=../inception \
--output_graph=../retrained_graph.pb \
--output_labels=../retrained_labels.txt \
--summaries_dir=../retrain_logs \
--image_dir ../data
