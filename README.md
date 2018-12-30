# uml_proj
To be finished.

Difference from the other paper:
1. Different meta-learning model
2. Training strategy (Teacher-forcing/different labels)

# Trainning:

You can change ```training_exp_num``` or ```batch_size``` if you want. But the following setting can run ~5min in CPU and give somewhat reasonable result during test.
```
 python model.py --training_exp_num 50 --batch_size 100
```

# Test:

This will load the latested checkpoint. To use different checkpoint, you can use the flag ```model_save_dir``` during training to specify a place to put checkpoint.
```
 python model.py --test
```

```show_graph``` will prompt the data cluster graph before doing cluster so that it is easier to see which data can succeed or fail.

```
 python model.py --test --show_graph
```
