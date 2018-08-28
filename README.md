# MatchingNet   
## Introduction   
Matching Net model used for oneshot classification. This model is trained and evaluated in mini ImageNet. Refer to https://arxiv.org/abs/1606.04080 for detailed description.    
Code is built under tensorflow slim framework. And use part of codes in repository https://github.com/tensorflow/models/tree/master/research/slim.   
Special thanks to https://github.com/zergylord and https://github.com/markdtw/matching-networks. I learned a lot in these implemetation. 
## Demo    
Clone this repository and run the demo.py to try the demo!    
You can also customize the demo to your application. Just run demo with your attributes as the following format   

    python demo.py ^
        --vector_size=80 ^
        --num_classes=10 ^
        --shot=5 ^
        --pic_size=224 ^
        --encode_graph=model/frozen_oneshot_base.pb ^
        --match_graph=model/oneshot_nfce10_5.pb ^
        --support_dir=support_data 
   
   
## Environment   
Windows 10   
Python 3  
Tensorflow r1.1.0   
opencv 3.3.1   
## Prepare Datasets   
### Folder Structure
    - MatchingNet   
      - data   
        - ${DATASET_NAME}   
          - train   
            ...images for train
          - val   
            ...images for validation
      ... other files and directories
### Commands   
For raw images are too big to feed into memory directly, I encode raw images into small vectors using trained CNN. You can following commands from MatchingNet directory to encode images into vectors and generate oneshot samples automatically. A `.tfrecord` file and a `.npz` file will appear in `/MatchingNet/data/mini_imagenet/train`. The `.tfreord` file contains oneshot samples and the `.npz` file contains encoded images.    

    python generate_oneshot_data.py ^
        --dataset_dir=data/mini_imagenet/train ^
        --possible_classes=10 ^
        --shot=5 ^
        --samples=80000 ^
        --from_raw_images=False ^
        --checkpoint_path=model/frozen_mobilenet_v1_224_prediction.pb ^
        --output_node=MobilenetV1/Logits/SpatialSqueeze ^
        --batch_size=50 ^
        --device=GPU ^
        --save_encoded_images=True 
   
#### Flag meaning     
`--dataset_dir` A directory contains raw images or a `.npz` file containing encoded images. If is a `.npz` file, the encoding process will not be carried out. Instead, oneshot samples will be generated using vectors stored in the `.npz` file.     
`--possible_classes` Number of classes to classify.   
`--shot` Number of samples in one support class.   
`--from_raw_images` Weather use CNN to encode images. If is True, images will not be encoded.   
`--checkpoint_path` Path of a frozen graph.   
`--output_node` A node of the frozen graph from which you want to get the output.   
`--batch_size` Number of images encoded at one run. It depends on your computer, if you use CPU to encode images, this number is better set smaller.   
`--device` Use CPU or GPU to run the encoding process.   
`--save_encoded_images` Weather to create `.npz` file containing encoded images. You can reuse the `.npz` file to generate new oneshot samples without encoding images repeatedly. 
#### Tips   
Do not rename the files generated. The names of these files suggest `possible_classes`, `shot`, `samples` and `output_node`, which will be used in training and evaluating process. 
## Train Matching Net   
### Commands
After datasets prepared, you can run following commands to train matching net.   

    python train_oneshot_classifier.py ^
        --train_dir=${TRAIN_DIR} ^
        --data_source=${DATA_PATH} ^
        --learning_rate=1000 ^
        --learning_rate_decay_type=fixed ^
        --fce=True ^
        --processing_step=5 ^
        --vector_size=80 ^
        --fc_num=2 ^
        --gradient_range=1e-6 ^
        --clone_on_cpu=True
#### Flag meaning     
`--train_dir` The directory to store checkpoints and summaries.     
`--data_source` Path to `.tfrecord` file containing oneshot samples.   
`--learning_rate` Learning rate.   
`--learning_rate_decay_type` Specifies how the learning rate is decayed. One of `fixed`, `exponential`, or `polynomial`.    
`--from_raw_images` Weather use CNN to encode images. If is True, images will not be encoded.   
`--fce` Weather to use fully context embedding.   
`--processing_step` Number of processing blocks in f embedding.   
`--vector_size` Size of encoded images. It depends on `output_node`.   
`--fc_num` Number of fully-connected layers in front of match layers.   
`--gradient_range` Range of gradients. Used in gradient clipping.   
### Notice   
For a better training result, I introduce layer normaliztion in matching net and use gradient clipping trick in training process. I cannot get a high-accuracy classify model just by training matching layers. So I add several fully-connected layers in front of matching layers. Matching net structure is in `/MatchingNet/nets/matchnet`. I think my implementation still has many problems I do not realize. If you have any advice, please do not hesitate to tell me. Thank you very much.
## Eval Matching Net   
    python eval_oneshot_classifier.py ^
        --eval_dir=${EVAL_DIR} ^
        --checkpoint_path=${CHECKPOINT_PATH} ^
        --data_source=${DATA_PATH} ^
        --fce=True ^
        --processing_step=5 ^
        --fc_num=2 ^
        --vector_size=80 
## Export Interference Graph    
    python export_oneshot_graph.py ^
        --output_file=${OUTPUT_FILE_PATH} ^
        --possible_classes=5 ^
        --shot=5 ^
        --fc_num=2 ^
        --vector_size=80 ^
        --processing_step=5 ^
        --fce=True
## Freeze Graph
    python freeze_graph.py ^
        --input_graph=${INTERFERENCE_GRAPH_PATH} ^
        --input_checkpoint=${CHECKPOINT_PATH} ^
        --input_binary=true  ^
        --output_graph=${OUTPUT_FILE_PATH} ^
        --output_node_names=${OUTPUT_NODE}
#### Acknowledge   
This file is copied from tensorflow repository.   
