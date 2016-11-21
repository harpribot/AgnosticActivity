
# evaluate all torch models and return to the dir
#NOTE: PPMI takes a long time, so it has been commented out in Exp4-Adversarial_Model2/eval_torch_models.py
cd Exp4-Adversarial_Model2
python eval_torch_models.py --load cv/adversary_activity/ep_10_loss_3.698637.t7 --prefix adv0
python eval_torch_models.py --load cv/adversary_reversal/ep_30_loss_3.727351.t7 --prefix adv1

python eval_torch_models.py --load cv/cosine/cosine_off/ep_19_loss_3.791296.t7 --prefix cos0
python eval_torch_models.py --load cv/cosine/cosine_on_20/ep_27_loss_3.926786.t7 --prefix cos20

cd ..

# evaluate all caffe models
cd Exp2-Baseline
python test.py snapshots/caffe_alexnet_train_iter_3000.caffemodel # pickles are stored in the Exp2-Baseline

cd ../Exp3-Siamese_Model1/
python test.py snapshots/caffe_alexnet_train_iter_3000.caffemodel # pickles are stored in the Exp3-Siamese_Model1
