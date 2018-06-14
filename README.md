# MatchingNet   
## Introduction   
Matching Net model used for oneshot classification. This model is trained and evaluated in mini ImageNet. Refer to https://arxiv.org/abs/1606.04080 for detailed description.    
Code is built under tensorflow slim framework. And use part of codes in repository https://github.com/tensorflow/models/tree/master/research/slim.   
Special thanks to https://github.com/zergylord and https://github.com/markdtw/matching-networks. I learned a lot in these implemetation.   
## Environment   
Windows   
Python 3  
Tensorflow r1.1.0
## Prepare Datasets   
    python generate_oneshot_data.py ^
        --dataset_dir=data/mini_imagenet/train ^
        --possible_classes=10 ^
        --shot=5 ^
        --samples=80000 ^
        --from_raw_images=False ^
        --checkpoint_path=model/frozen_mobilenet_v1_224_prediction.pb ^
        --output_node=MobilenetV1/Logits/SpatialSqueeze ^
        --batch_size=10 ^
        --device=CPU ^
        --save_encoded_images=True    
## Train Matching Net   
    python train_oneshot_classifier.py ^
        --train_dir=model/oneshot_test1 ^
        --data_source=data/mini_imagenet/test/oneshot_SpatialSqueeze_5_5_10.tfrecord ^
        --learning_rate=1 ^
        --learning_rate_decay_type=fixed ^
        --fce=True ^
        --processing_step=5 ^
        --vector_size=80 ^
        --fc_num=2 ^
        --clone_on_cpu=True
## Eval Matching Net   
    python eval_oneshot_classifier.py ^
        --eval_dir=model/oneshot_test1 ^
        --checkpoint_path=model/oneshot_test1 ^
        --data_source=data/mini_imagenet/test/oneshot_SpatialSqueeze_5_5_10.tfrecord ^
        --fce=True ^
        --processing_step=5 ^
        --fc_num=2 ^
        --vector_size=80 
## Export Interference Graph    
    python export_oneshot_graph.py ^
        --output_file=model/test.pb ^
        --possible_classes=5 ^
        --shot=5 ^
        --fc_num=2 ^
        --vector_size=80 ^
        --processing_step=5 ^
        --fce=True
## Freeze Graph
## Demo
