#include <string>
#include <vector>
#include <sys/param.h>
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/util_others.hpp"
using namespace caffe_fcn_data_layer;
using namespace std;
namespace caffe {

template <typename Dtype>
IImageDataSourceProvider<Dtype>::IImageDataSourceProvider(bool need_shuffle ){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	shuffle_ = need_shuffle;
}


template <typename Dtype>
ImageDataSourceProvider<Dtype>::ImageDataSourceProvider(bool need_shuffle):
		IImageDataSourceProvider<Dtype>(need_shuffle)
{
	lines_id_=0;
}



template <typename Dtype>
ImageDataSourceProvider<Dtype>::~ImageDataSourceProvider()
{

}


template <typename Dtype>
std::pair<std::string, vector<Dtype> > ImageDataSourceProvider<Dtype>::GetOneSample()
{
	CHECK(samples_.size() ==  shuffle_idxs_.size());
	CHECK_LT(lines_id_,samples_.size());
	CHECK_GT(samples_.size(),0);
	int id = shuffle_idxs_[this->lines_id_++];
	if(lines_id_ >=  shuffle_idxs_.size())
	{
		LOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if (this->shuffle_ ) {
			ShuffleImages();
		}
	}
	return samples_[id];
}

template <typename Dtype>
vector<Dtype> IImageDataSourceProvider<Dtype>::SwapInstance(vector<Dtype>& coords,
		int num_anno_points_per_instance, int id1, int id2)
{
	vector<Dtype> res_coords = coords;
	Dtype temp;
	int n_instance = coords.size() / (num_anno_points_per_instance*2);
	if(id1 >= n_instance || id2 >= n_instance)
		return res_coords;
	for(int i=0; i < num_anno_points_per_instance*2 ; ++i)
	{
		temp = res_coords[id1 * num_anno_points_per_instance*2 + i];
		res_coords[id1 * num_anno_points_per_instance*2 + i] =
				res_coords[id2 * num_anno_points_per_instance*2 + i];
		res_coords[id2 * num_anno_points_per_instance*2 + i] = temp;
	}
	return res_coords;
}

template <typename Dtype>
void ImageDataSourceProvider<Dtype>::PushBackSample(const pair< string, vector<Dtype> > & cur_sample){
	samples_.push_back(cur_sample);
	shuffle_idxs_.push_back(shuffle_idxs_.size());
}
template <typename Dtype>
bool ImageDataSourceProvider<Dtype>::ReadSamplesFromFile(
		const string & file_list_name, const string & folder,int num_anno_points_per_instance,
		bool no_expand ,int class_flag_id)
{
	LOG(INFO) << "Opening file " << file_list_name;
	std::ifstream infile(file_list_name.c_str());

	samples_.clear();
	shuffle_idxs_.clear();
	if(infile.good() == false)
		return false;
	int n_img = 0;

	string line;
	string filename;
	while (std::getline(infile, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		iss >> filename;
		filename = folder  +filename;
		Dtype coord;
		vector<Dtype> coords;
		while(iss >> coord) {
		  coords.push_back((coord));
		}
		n_img++;
		pair< string, vector<Dtype> > cur_sample = make_pair(filename, coords);
		PushBackSample(cur_sample);

		if(! no_expand)
		{
			CHECK_EQ(coords.size() % (num_anno_points_per_instance*2), 0 );
			int n_instance = coords.size() / (num_anno_points_per_instance*2);
			for(int i = 1; i < n_instance; ++i)
			{
				cur_sample = make_pair(filename,
						IImageDataSourceProvider<Dtype>::SwapInstance(coords,num_anno_points_per_instance, 0, i));
				PushBackSample(cur_sample);
			}
		}
	}
	infile.close();
	LOG(INFO) << "A total of " << n_img << " images with "<<samples_.size()<<" instances.";
	lines_id_=0;
	if (this->shuffle_ && this->samples_.size() > 0)
	{
		LOG(INFO) << "Shuffle Image at start " ;
		ShuffleImages();
	}
	return true;
}

template <typename Dtype>
string ImageDataSourceProvider<Dtype>::GetSampleFolder(const pair< string, vector<Dtype> > & cur_sample,
		 string concat_symbol){
	std::vector<std::string> splited_name= std_split(cur_sample.first,"/");
	string out_name;
  if(splited_name.size() > 0){
          out_name = splited_name[0];
  }
  for(int i=1 ; i < splited_name.size()-1; ++i){
          out_name = out_name + concat_symbol+ splited_name[i] ;
  }
	return out_name;
}



template <typename Dtype>
ImageDataROISourceProvider<Dtype>::ImageDataROISourceProvider(bool need_shuffle):
		IImageDataSourceProvider<Dtype>(need_shuffle)
{
	lines_id_=0;
	roi_filename_ = "";
}


template <typename Dtype>
ImageDataROISourceProvider<Dtype>::~ImageDataROISourceProvider(){

}
template <typename Dtype>
void ImageDataROISourceProvider<Dtype>::PushBackAnnoSample(const pair< string, vector<Dtype> > & cur_sample,
		int num_anno_points_per_instance){
	CHECK_EQ(cur_sample.second.size() % (num_anno_points_per_instance*2) , 0);
	int anno_sample_idx = this->FindAnnoSampleIdxByName(cur_sample.first);
	if(anno_sample_idx == -1){
		int idx = anno_samples_.size();
		img_name_map_.insert(pair<string, int>(cur_sample.first,idx));
		anno_samples_.push_back(cur_sample);
	}else{
		int new_come_anno_size = cur_sample.second.size();
		const vector<Dtype>& new_come_anno = cur_sample.second;
		vector<Dtype> & old_anno =
				anno_samples_[anno_sample_idx].second;
		for(int i =0; i < new_come_anno_size; ++i){
			old_anno.push_back(new_come_anno[i]);
		}
	}
}

template <typename Dtype>
void ImageDataROISourceProvider<Dtype>::PushBackROISample(const pair< string, vector<Dtype> > & cur_sample){

	CHECK_EQ(cur_sample.second.size() % (num_roi_points_per_instance_*2) , 0);
	int anno_sample_idx = this->FindAnnoSampleIdxByName(cur_sample.first);
	if(anno_sample_idx == -1){
		LOG(INFO)<<"Warning: Image "<<cur_sample.first<<" from ROI_files cannot be found at Anno_file. So image "<<
				cur_sample.first<<" is treated as negative image";
		int idx = anno_samples_.size();
		img_name_map_.insert(pair<string, int>(cur_sample.first,idx));
		anno_samples_.push_back(pair< string, vector<Dtype> >(cur_sample.first,vector<Dtype>()));
	}

	int n_rois = cur_sample.second.size() / (num_roi_points_per_instance_*2);
	for(int i=0; i < n_rois; ++i){
		vector<Dtype> entity(num_roi_points_per_instance_*2,-1);
		copy(cur_sample.second.begin()+ i*num_roi_points_per_instance_*2,
				cur_sample.second.begin()+(i+1)*num_roi_points_per_instance_*2 ,
				entity.begin());
		roi_samples_.push_back(std::pair<int, vector<Dtype> >(anno_sample_idx, entity));
		shuffle_idxs_.push_back(shuffle_idxs_.size());
	}
}

template <typename Dtype>
bool ImageDataROISourceProvider<Dtype>::ReadAnnoOrROISamplesFromFile(const string & file_list_name,
		const string & folder, int num_anno_points_per_instance, bool is_roi_filelist  ){

	LOG(INFO) << "Opening file " << file_list_name;
	std::ifstream infile(file_list_name.c_str());
	if(infile.good() == false)
		return false;
	int n_img = 0;
	string line;
	string filename;
	while (std::getline(infile, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		iss >> filename;
		filename = folder  +filename;
		Dtype coord;
		vector<Dtype> coords;
		while(iss >> coord) {
			coords.push_back((coord));
		}
		n_img++;
		pair< string, vector<Dtype> > cur_sample = make_pair(filename, coords);
		is_roi_filelist? PushBackROISample(cur_sample) :
				PushBackAnnoSample(cur_sample,num_anno_points_per_instance);
	}
	infile.close();
//	string file_type_name = is_roi_filelist? "ROI_file" : "Anno_file";
//	LOG(INFO) << "A total of " << n_img << " images with "<<samples_.size()<<" instances in "<<file_type_name;
	if(is_roi_filelist){
		LOG(INFO) << "A total of " << n_img << " images with "<<roi_samples_.size()<<" instances in ROI_file";
	}else{
		LOG(INFO) << "A total of " << n_img << " images in Anno_file";
	}
	return true;
}

template <typename Dtype>
bool ImageDataROISourceProvider<Dtype>::ReadSamplesFromFile(const string & file_list_name,
		const string & folder, int num_anno_points_per_instance, bool no_expand ,int class_flag_id  ){
	anno_samples_.clear();
	roi_samples_.clear();
	img_name_map_.clear();
	shuffle_idxs_.clear();

	CHECK(this->roi_filename_ != "");
	bool success = ReadAnnoOrROISamplesFromFile(file_list_name,folder, num_anno_points_per_instance,false) &&
			ReadAnnoOrROISamplesFromFile(this->roi_filename_,folder, num_anno_points_per_instance,true);
	if (this->shuffle_ && this->roi_samples_.size() > 0)
	{
		LOG(INFO) << "Shuffle Image at start " ;
		ShuffleImages();
	}
	return success;
}

template<typename Dtype>
std::pair<std::string, vector<Dtype> > ImageDataROISourceProvider<Dtype>::GetOneSample(){
	CHECK(roi_samples_.size() ==  shuffle_idxs_.size());
	CHECK_LT(lines_id_,roi_samples_.size());
	CHECK_GT(roi_samples_.size(),0);
	int id = shuffle_idxs_[this->lines_id_++];
	if(lines_id_ >=  shuffle_idxs_.size())
	{
		LOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if (this->shuffle_ ) {
			ShuffleImages();
		}
	}
	int sample_id = roi_samples_[id].first;
	CHECK_LT(sample_id, anno_samples_.size());
	vector<Dtype>& roi_points = roi_samples_[id].second;
	vector<Dtype>& anno_points = anno_samples_[sample_id].second;
	std::pair<std::string, vector<Dtype> > res;

	res.first = anno_samples_[sample_id].first;
	vector<Dtype> points(roi_points.size() + anno_points.size(), -1);
	copy(roi_points.begin(), roi_points.end(),points.begin());
	copy(anno_points.begin(), anno_points.end(),points.begin()+roi_points.size());
	res.second = points;
	return res;
}

template <typename Dtype>
ImageDataSourceMultiClassProvider<Dtype>::ImageDataSourceMultiClassProvider(bool need_shuffle )
		:IImageDataSourceProvider<Dtype>(need_shuffle)
{

		lines_class_id_ = 0;
		class_ids_.clear();
		image_data_providers_.clear();
		shuffle_class_idxs_.clear();
}

template <typename Dtype>
ImageDataSourceMultiClassProvider<Dtype>::~ImageDataSourceMultiClassProvider(){

}

template <typename Dtype>
void ImageDataSourceMultiClassProvider<Dtype>::ShuffleImages(){
	caffe::rng_t* prefetch_rng =
		        static_cast<caffe::rng_t*>(this->prefetch_rng_->generator());
	shuffle(shuffle_class_idxs_.begin(), shuffle_class_idxs_.end(), prefetch_rng);

}

template <typename Dtype>
int ImageDataSourceMultiClassProvider<Dtype>::GetSamplesSize(){
	int size =0;
	for(int i=0; i < image_data_providers_.size(); ++i){
		size += image_data_providers_[i].GetSamplesSize();
	}
	return size;
}

template <typename Dtype>
std::pair<std::string, vector<Dtype> > ImageDataSourceMultiClassProvider<Dtype>::GetOneSample()
{
	CHECK_EQ(shuffle_class_idxs_.size() , class_ids_.size());
	CHECK_LT(lines_class_id_,class_ids_.size());
	int id = shuffle_class_idxs_[this->lines_class_id_++];
	if(lines_class_id_ >=  shuffle_class_idxs_.size())
	{
		lines_class_id_ = 0;
		if (this->shuffle_ ) {
			ShuffleImages();
		}
	}
	while(image_data_providers_[id].GetSamplesSize() <1){
		id = shuffle_class_idxs_[this->lines_class_id_++];
		if(lines_class_id_ >=  shuffle_class_idxs_.size())
		{
			lines_class_id_ = 0;
			if (this->shuffle_ ) {
				ShuffleImages();
			}
		}
	}
	return image_data_providers_[id].GetOneSample();
}

template <typename Dtype>
void ImageDataSourceMultiClassProvider<Dtype>::PushBackSample(
		const pair< string, vector<Dtype> > & cur_sample,
		int num_anno_points_per_instance,int class_flag_id ){
	int class_id = 0;
	if(class_flag_id >= 0){
		CHECK_LT(class_flag_id , num_anno_points_per_instance);
		class_id =  (cur_sample.second[class_flag_id*2]);
	}

	while(class_id >= class_ids_.size()){
		shuffle_class_idxs_.push_back(shuffle_class_idxs_.size());
		class_ids_.push_back(class_ids_.size());
		image_data_providers_.push_back(ImageDataSourceProvider<Dtype>(this->shuffle_));
	}
	image_data_providers_[class_id].PushBackSample(cur_sample);
}

template <typename Dtype>
bool ImageDataSourceMultiClassProvider<Dtype>::ReadSamplesFromFile(
		const string & file_list_name, const string & folder,int num_anno_points_per_instance,
		bool no_expand, int class_flag_id )
{
	class_ids_.clear();
	image_data_providers_.clear();

	LOG(INFO) << "Opening file " << file_list_name;
	std::ifstream infile(file_list_name.c_str());

	if(infile.good() == false)
		return false;

	int n_img = 0;

	string line;
	string filename;
	while (std::getline(infile, line)) {
		if (line.empty()) continue;
		std::istringstream iss(line);
		iss >> filename;
		filename = folder  +filename;
		Dtype coord;
		vector<Dtype> coords;
		while(iss >> coord) {
		  coords.push_back((coord));
		}
		n_img++;
		pair< string, vector<Dtype> > cur_sample = make_pair(filename, coords);
		PushBackSample(cur_sample,num_anno_points_per_instance,class_flag_id);


		if(! no_expand)
		{
			CHECK_EQ(coords.size() % (num_anno_points_per_instance*2), 0 );
			int n_instance = coords.size() / (num_anno_points_per_instance*2);
			for(int i = 1; i < n_instance; ++i)
			{
				cur_sample = make_pair(filename,
								IImageDataSourceProvider<Dtype>::SwapInstance(coords,num_anno_points_per_instance, 0, i));
				PushBackSample(cur_sample,num_anno_points_per_instance,class_flag_id);
			}
		}
	}
	infile.close();

	LOG(INFO) << "A total of " << n_img << " images with "<<GetSamplesSize()<<" instances.";
	CHECK_GT(GetSamplesSize(),0);
	for(int i=0; i < image_data_providers_.size(); ++i){
		image_data_providers_[i].SetLineId(0);
		if(this->shuffle_){
			LOG(INFO) << "Shuffle Image of class "<< i<<"  ." ;
			image_data_providers_[i].ShuffleImages();
		}
	}

	lines_class_id_ = 0;
	if (this->shuffle_ && this->class_ids_.size() > 0)
	{
		ShuffleImages();
	}
	return true;
}

/**
 * Functions for ImageDataHardNegSourceProvider Layer
 */
template <typename Dtype>
ImageDataHardNegSourceProvider<Dtype>::ImageDataHardNegSourceProvider(bool need_shuffle  )
				:ImageDataSourceProvider<Dtype>::ImageDataSourceProvider(need_shuffle)
{
	stdLengthType = FCNImageDataSourceParameter_STDLengthType_HEIGHT;
	bootstrap_std_length_ = 0;
}

template <typename Dtype>
ImageDataHardNegSourceProvider<Dtype>::~ImageDataHardNegSourceProvider()

{

}


template <typename Dtype>
void ImageDataHardNegSourceProvider<Dtype>::SetUpHardNegParam(
		const FCNImageDataSourceParameter & fcn_img_data_source_param)
{
	this->bootstrap_std_length_ = fcn_img_data_source_param.bootstrap_std_length();
	this->stdLengthType = fcn_img_data_source_param.bootstrap_std_length_type();
	this->shuffle_ = fcn_img_data_source_param.shuffle();
}

template <typename Dtype>
bool ImageDataHardNegSourceProvider<Dtype>::ReadHardSamplesFromFile(
		const string & file_list_name,const string & neg_img_folder,
		int input_height, int input_width)
{
	LOG(INFO) << "Opening file " << file_list_name;
	this->samples_.clear();
	this->shuffle_idxs_.clear();

	char name_pr[512];
	FILE* pred_fd = fopen(file_list_name.c_str(),"r");
	if(pred_fd == NULL)
	{
		LOG(INFO) << "Can not opening file " << file_list_name;
	}

	while(fscanf(pred_fd,"%s",name_pr) == 1)
	{
		string file_name =  string(name_pr);

		file_name = neg_img_folder +file_name;
		int n_face = 0;
		CHECK(fscanf(pred_fd,"%d",&n_face) == 1);
		for(int i=0; i < n_face; i++)
		{
		  float lt_x, lt_y, height,width,score;
		  CHECK(fscanf(pred_fd, "%f %f %f %f %f",&lt_x, &lt_y, &width, &height, &score) == 5);
		  vector<Dtype> loc;
		  Dtype cur_length = 0 ;
		  if(stdLengthType == FCNImageDataSourceParameter_STDLengthType_HEIGHT)
		  {
			  cur_length = height;
		  }
		  else if(stdLengthType == FCNImageDataSourceParameter_STDLengthType_WIDTH)
		  {
			  cur_length = width;
		  }
		  else if(stdLengthType == FCNImageDataSourceParameter_STDLengthType_DIAG)
		  {
			  cur_length = sqrt(height*height + width * width);
		  }
		  else
		  {
			  LOG(ERROR)<< "undefined std_length_mode: "<<stdLengthType;
		  }

		  Dtype scale_needed =  bootstrap_std_length_/cur_length;
		  Dtype half_height = Dtype( (input_height+0.0)/2/scale_needed );
		  Dtype half_width = Dtype( (input_width+0.0)/2/scale_needed );

		  Dtype c_x = lt_x + width/2;
		  Dtype c_y = lt_y + height/2;

		  loc.push_back(c_x - half_width);
		  loc.push_back(c_y - half_height );
		  loc.push_back(c_x + half_width);
		  loc.push_back(c_y + half_height);
		  this->samples_.push_back(std::make_pair(file_name, loc));
//		  LOG(INFO)<<"              add hard_neg:" <<file_name<<" "<<
//				  loc[0]<<" "<<loc[1]<<" "<< loc[2] <<" "<< loc[3];
		  this->shuffle_idxs_.push_back(this->samples_.size());
		}

	}
	CHECK_EQ(this->samples_.size(),  this->shuffle_idxs_.size());
	fclose(pred_fd);
	LOG(INFO) << "A total of " << this->samples_.size() << " images.";

	this->lines_id_=0;
	if (this->shuffle_ && this->samples_.size() > 0)
	{
		LOG(INFO) << "Shuffle Image at start " ;
		this->ShuffleImages();
	}

	return true;
}

/**
 * Functions for ImageDataSourceBootstrapableProvider Layer
 */
template <typename Dtype>
ImageDataSourceBootstrapableProvider<Dtype>::ImageDataSourceBootstrapableProvider()
{

	batch_pos_count_ = 0;
	batch_neg_count_ = 0;
	batch_hard_neg_count_ = 0;
	has_roi_file_ = false;
	this->batch_size_ = 0;
	this->neg_ratio_ = 0;
	this->bootstrap_hard_ratio_ = 0;
	this->shuffle_ = true;
	this->no_expand_pos_anno_ = false;
	multi_class_sample_balance_ = false;

}

template <typename Dtype>
ImageDataSourceBootstrapableProvider<Dtype>::~ImageDataSourceBootstrapableProvider()
{

}
template <typename Dtype>
void ImageDataSourceBootstrapableProvider<Dtype>::SetUpParameter(
		const FCNImageDataParameter & fcn_image_data_param)
{
	CHECK(fcn_image_data_param.has_fcn_image_data_source_param());
	this->SetUpParameter(fcn_image_data_param.fcn_image_data_source_param());
}


template <typename Dtype>
void ImageDataSourceBootstrapableProvider<Dtype>::SetUpParameter(
		const FCNImageDataSourceParameter & fcn_img_data_source_param)
{
	this->batch_size_ = fcn_img_data_source_param.batch_size();
	CHECK_GT(batch_size_, 0);
	this->neg_ratio_ = fcn_img_data_source_param.neg_ratio();
	multi_class_sample_balance_ = fcn_img_data_source_param.multi_class_sample_balance();
	this->bootstrap_hard_ratio_ =fcn_img_data_source_param.bootstrap_hard_ratio();
	this->shuffle_ = fcn_img_data_source_param.shuffle();
	this->no_expand_pos_anno_ = fcn_img_data_source_param.no_expand_pos_anno();
	CHECK(fcn_img_data_source_param.has_pos_samples_source())<<
			"pos_sample_source is necessary ";
	CHECK(fcn_img_data_source_param.has_pos_img_folder())<<
				"pos_img_folder is necessary ";

	this->has_roi_file_ = fcn_img_data_source_param.has_roi_samples_source();
	this->hard_neg_samples.SetUpHardNegParam(fcn_img_data_source_param);
	if(multi_class_sample_balance_)
		pos_samples_ptr.reset(new ImageDataSourceMultiClassProvider<Dtype>(this->shuffle_));
	else{
		if(!has_roi_file_){
			pos_samples_ptr.reset(new ImageDataSourceProvider<Dtype>(this->shuffle_));
		}
		else{
			pos_samples_ptr.reset(new ImageDataROISourceProvider<Dtype>(this->shuffle_));
		}
	}
}

template <typename Dtype>
void ImageDataSourceBootstrapableProvider<Dtype>::ReadPosAndNegSamplesFromFiles(
		const FCNImageDataSourceParameter & fcn_img_data_source_param,
		int num_anno_points_per_instance,int class_flag_id)
{
	this->pos_img_folder_ = fcn_img_data_source_param.pos_img_folder();
	this->pos_samples_ptr->SetShuffleFlag(this->shuffle_);
	if(this->has_roi_file_){
		ImageDataROISourceProvider<Dtype>* tmp_ptr =
				dynamic_cast<ImageDataROISourceProvider<Dtype>*>(this->pos_samples_ptr.get());
		tmp_ptr->SetROIFileName(fcn_img_data_source_param.roi_samples_source());
	}
	this->pos_samples_ptr->ReadSamplesFromFile(
			fcn_img_data_source_param.pos_samples_source(),
			fcn_img_data_source_param.pos_img_folder(),
			num_anno_points_per_instance,no_expand_pos_anno_,class_flag_id);
	CHECK_GT(this->pos_samples_ptr->GetSamplesSize(),0);

	if(fcn_img_data_source_param.has_neg_samples_source())
	{
		this->all_neg_samples.SetShuffleFlag(this->shuffle_);
		string neg_fold = fcn_img_data_source_param.has_neg_img_folder() ?
				fcn_img_data_source_param.neg_img_folder():fcn_img_data_source_param.pos_img_folder() ;
		this->neg_img_folder_ = neg_fold;
		this->all_neg_samples.ReadSamplesFromFile(
				fcn_img_data_source_param.neg_samples_source(),neg_fold,
				num_anno_points_per_instance,true);
		CHECK_GT(this->all_neg_samples.GetSamplesSize(),0);
	}
}

template <typename Dtype>
void ImageDataSourceBootstrapableProvider<Dtype>::FetchSamplesTypeInBatch()
{
	CHECK_GE(neg_ratio_,0);
	CHECK_LE(neg_ratio_,1);
	batch_pos_count_ = (this->all_neg_samples.GetSamplesSize() == 0 &&
			this->hard_neg_samples.GetSamplesSize() == 0) ?
				batch_size_: int(batch_size_*(1-this->neg_ratio_));
	batch_pos_count_ = MIN(batch_size_,MAX(batch_pos_count_,1));

	batch_neg_count_ = batch_size_ - batch_pos_count_;
	if(this->hard_neg_samples.GetSamplesSize() == 0)
	{
	  batch_hard_neg_count_ = 0;
	}
	else if(this->all_neg_samples.GetSamplesSize() == 0)
	{
	  batch_hard_neg_count_ =batch_neg_count_;
	}
	else
	{
	  batch_hard_neg_count_  = int( this->bootstrap_hard_ratio_ *batch_neg_count_);
	}
	CHECK_EQ(batch_pos_count_ +batch_neg_count_,  batch_size_);
	CHECK(batch_neg_count_ >= 0 &&batch_neg_count_< batch_size_);
	CHECK_LE(batch_hard_neg_count_,batch_neg_count_);

	this->cur_batch_sample_type_.clear();
	for(int i = 0 ; i < batch_pos_count_; ++i)
	{
		if(this->has_roi_file_){
			cur_batch_sample_type_.push_back(caffe::SOURCE_TYPE_POSITIVE_WITH_ROI);
		}else
			cur_batch_sample_type_.push_back(caffe::SOURCE_TYPE_POSITIVE);
	}
	for(int i = 0 ; i <batch_neg_count_-batch_hard_neg_count_; ++i )
	{
	  cur_batch_sample_type_.push_back(caffe::SOURCE_TYPE_ALL_NEGATIVE);
	}
	for(int i = 0 ; i < batch_hard_neg_count_; ++i )
	{
	  cur_batch_sample_type_.push_back(caffe::SOURCE_TYPE_HARD_NEGATIVE);
	}
	CHECK_EQ(cur_batch_sample_type_.size() ,  batch_size_);

}

template <typename Dtype>
void ImageDataSourceBootstrapableProvider<Dtype>::FetchBatchSamples()
{
	FetchSamplesTypeInBatch();
	cur_batch_samples_.clear();
	for(int i=0 ; i < batch_size_; ++i)
	{
		IImageDataSourceProvider<Dtype>* source_ptr = NULL;
		switch(cur_batch_sample_type_[i])
		{
			case caffe::SOURCE_TYPE_POSITIVE_WITH_ROI:
			case caffe::SOURCE_TYPE_POSITIVE:
			{
				source_ptr = pos_samples_ptr.get();
				break;
			}
			case caffe::SOURCE_TYPE_ALL_NEGATIVE:
			{
				source_ptr = &all_neg_samples;
				break;
			}
			case caffe::SOURCE_TYPE_HARD_NEGATIVE:
			{
				source_ptr = &hard_neg_samples;
				break;
			}
			default:
			    LOG(FATAL) << "Unknown type " << cur_batch_sample_type_[i];
		}
		CHECK_NOTNULL(source_ptr);
		cur_batch_samples_.push_back(source_ptr->GetOneSample());
	}
}

template <typename Dtype>
std::pair<std::string, vector<Dtype> > &
ImageDataSourceBootstrapableProvider<Dtype>::GetMutableSampleInBatchAt(int id)
{
	return cur_batch_samples_[id];
}

template <typename Dtype>
ImageDataSourceSampleType& ImageDataSourceBootstrapableProvider<Dtype>::
								GetMutableSampleTypeInBatchAt(int id)
{
	return cur_batch_sample_type_[id];
}


template <typename Dtype>
bool ImageDataSourceBootstrapableProvider<Dtype>::ReadHardSamplesForBootstrap(
		const string& detection_result_file, const string & neg_img_folder,
		int input_height, int input_width )
{
	return hard_neg_samples.ReadSamplesFromFile(detection_result_file,neg_img_folder,
			input_height,input_width);
}

template <typename Dtype>
int ImageDataSourceBootstrapableProvider<Dtype>::GetTestIterations(){
	return std::ceil(pos_samples_ptr->GetSamplesSize() / std::floor(batch_size_));
}



INSTANTIATE_CLASS(ImageDataSourceProvider);
INSTANTIATE_CLASS(ImageDataROISourceProvider);
INSTANTIATE_CLASS(ImageDataSourceBootstrapableProvider);
INSTANTIATE_CLASS(ImageDataHardNegSourceProvider);
INSTANTIATE_CLASS(ImageDataSourceMultiClassProvider);
}  // namespace caffe
