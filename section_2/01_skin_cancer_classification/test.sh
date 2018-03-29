cd tensorflow
python tensorflow/examples/label_image/label_image.py \
--graph=../retrained_graph.pb \
--labels=../retrained_labels.txt \
--input_layer=Mul \
--output_layer=final_result \
--input_mean=128 --input_std=128 \
--image=../data/actinic_keratosis/pic_008.jpg
