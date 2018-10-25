# script for training and testing densebox for kitti car detection #

### Step ###

* Prepare training and testing data: sh prepare_data.sh 
* Start training: sh train.sh
* Show intermediate detection results given a trained model: sh show_result.sh
* Show detection results and compute average precision given a trained model: sh test_model.sh
* You should expect to see approximately 0.85 map