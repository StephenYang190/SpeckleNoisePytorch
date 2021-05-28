# Readme
## Name convention
-o use octave convolution
-m use median layer
-c use vanilla convolution
-a use adaptive median layer
-r use median layer in residual block
train command example:
nohup python ven_train.py --mode train --device_id cuda:1 --batch_size 16 --save_folder checkpoints/ccheck --op c > demo_c.out 2>&1 &
