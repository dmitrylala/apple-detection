# Apple Detection

Instance segmentation of apples in industrial orchards using Mask-RCNN architecture. Code was written on PyTorch. Check page of [research](https://imaging.cs.msu.ru/en/research/apples) to see tensorboards with results of experiments.

Some examples below (from left ro right - input image, ground truth, neural network prediction)

![fuji_apples1](/images/vis_13.png)

![fuji_apples1](/images/vis_16.png)

## Run train
~~~
python src/train.py <train_cfg.yaml>
~~~

## Run evaluation
~~~
python src/evaluate.py <eval_cfg.yaml>
~~~
