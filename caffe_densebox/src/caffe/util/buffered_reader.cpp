#include "caffe/util/buffered_reader.hpp"
#include <map>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/util_others.hpp"
using namespace std;
namespace caffe {

/**
 * for BufferedIMGReader
 */

template <typename Dtype>
BufferedImgReader<Dtype>::BufferedImgReader(){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	round_counter_ = 0;
	Init(0,"");
	this->SetLrPostfix("_lr");
}
template <typename Dtype>
BufferedImgReader<Dtype>::~BufferedImgReader(){

}

template <typename Dtype>
BufferedImgReader<Dtype>::BufferedImgReader(const string& data_name, int buff_size){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	Init(buff_size,data_name);
	this->SetLrPostfix("_lr");
}
template <typename Dtype>
unsigned int BufferedImgReader<Dtype>::PrefetchRand(){
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	return (*prefetch_rng)();
}

template <typename Dtype>
int BufferedImgReader<Dtype>::GetOverWriteIdx(){
	round_counter_ = (round_counter_+1)%buff_size_;
//	return	PrefetchRand()%buff_size_;
	return round_counter_;
}
template <typename Dtype>
void BufferedImgReader<Dtype>::SetDataName(const string& name){
	this->data_name_ = name;
}

template <typename Dtype>
void BufferedImgReader<Dtype>::SetLrPostfix(const string& postfix){
	this->lr_postfix_ = postfix;
}

template <typename Dtype>
void BufferedImgReader<Dtype>::Init(int buff_size,const string &data_name){
	buff_size_ = buff_size;
	data_name_ = data_name;
	while(buffered_blobs_ptr_.size() < buff_size_){
		buffered_blobs_ptr_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
	}
	for(int i=0; i < buff_size_;++i){
		buffered_blobs_ptr_[i]->Reshape(0,0,0,0);
	}
	while(read_write_blobs_mutex_ptr_.size() < buff_size_){
		read_write_blobs_mutex_ptr_.push_back(shared_ptr<boost::shared_mutex>(new boost::shared_mutex()));
	}
	name_map_.clear();
	name_list_.clear();
//	LOG(INFO)<<"				call  BufferedImgReader<Dtype>::Init";
}



template <typename Dtype>
bool BufferedImgReader<Dtype>::ReadBufferedBlob(int blob_id,Blob<Dtype>& dst_blob){
	if(blob_id < buff_size_ && blob_id >= 0){
		dst_blob.Reshape(buffered_blobs_ptr_[blob_id]->shape());
		if(dst_blob.count() == 0){
			return false;
		}
		caffe::caffe_copy(dst_blob.count(),
				buffered_blobs_ptr_[blob_id]->cpu_data(),dst_blob.mutable_cpu_data());
		return true;
	}
	return false;
}

template <typename Dtype>
int BufferedImgReader<Dtype>::UpdateNameTable(const string& filename ){
	int idx = FindBlobIdxByName(filename);

	if(idx < 0){
		// if not full, just add into table.Otherwise, swap one
		if(name_map_.size() < buff_size_){
			idx = name_map_.size();
			name_map_.insert(pair<string, int>(filename,idx));
			name_list_.push_back(filename);
			CHECK_EQ(name_map_.size(),name_list_.size());
		}else{
			idx = GetOverWriteIdx();
			string name_to_delete = name_list_[idx];
			name_list_[idx] = filename;
			name_map_.erase(name_to_delete);
			name_map_.insert(pair<string, int>(filename,idx));
			CHECK_EQ(name_map_.size(),name_list_.size());
		}
	}
	return idx;
}

template <typename Dtype>
int BufferedImgReader<Dtype>::FindBlobIdxByName(const string& name){
	map<string,int>::iterator it = name_map_.find(name);
	if(it != name_map_.end()){
		return it->second;
	}
	return -1;
}


template <typename Dtype>
void BufferedImgReader<Dtype>::LoadToBlob(const string& filename,Blob<Dtype>& dst_blob){
	boost::unique_lock<boost::shared_mutex> table_lock(this->name_table_mutex_);
	int blob_idx = FindBlobIdxByName(filename);
	if(blob_idx >= 0 ){
//		LOG(INFO)<<"			file: "<<filename<<" is already in buffer. Just read.";
		// if is buffered, use read lock and copy data;
		boost::shared_lock<boost::shared_mutex> blob_read_lock(
				*(read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
		ReadBufferedBlob(blob_idx,dst_blob);
		blob_read_lock.unlock();
	}else{
//		LOG(INFO)<<"			file: "<<filename<<" is not in buffer. Append to NameTable.";
//		LOG(INFO)<<"			before_blob_idx:"<<blob_idx <<"table_size: "<<name_map_.size();
		blob_idx = UpdateNameTable(filename);
//		LOG(INFO)<<"			after_blob_idx:"<<blob_idx <<"table_size: "<<name_map_.size();
		boost::unique_lock<boost::shared_mutex> blob_write_lock(
				*(read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
//		LOG(INFO)<<"			file: "<<filename<<" is being read to buffer.";
		WriteFileToBlob(filename,*(buffered_blobs_ptr_[blob_idx].get()));
		ReadBufferedBlob(blob_idx,dst_blob);
//		LOG(INFO)<<"			file: "<<filename<<" is finished reading to buffer.";
		blob_write_lock.unlock();
	}
}


/**
 * for BufferedHDF5Reader
 */

template <typename Dtype>
void BufferedHDF5Reader<Dtype>::WriteFileToBlob(const string& filename,Blob<Dtype>& dst_blob){
	if(exclusive_load_){
		LoadHDF5FileToBlob(filename.c_str(),this->data_name_,dst_blob,hdf5_mutex_);
	}else{
		LoadHDF5FileToBlob(filename.c_str(),this->data_name_,dst_blob);
	}
}


/**
 * for BufferedColorJPGReader
 */


template <typename Dtype>
void BufferedColorJPGReader<Dtype>::Init(int buff_size,const string &data_name){
	this->buff_size_ = buff_size;
	this->data_name_ = data_name;
	while(buffered_cv_mat_.size() < this->buff_size_){
		buffered_cv_mat_.push_back(cv::Mat());
	}
	while(this->read_write_blobs_mutex_ptr_.size() < this->buff_size_){
		this->read_write_blobs_mutex_ptr_.push_back(shared_ptr<boost::shared_mutex>(new boost::shared_mutex()));
	}
	this->name_map_.clear();
	this->name_list_.clear();
//	LOG(INFO)<<"				call  BufferedJPGReader<Dtype>::Init";
}

template <typename Dtype>
BufferedColorJPGReader<Dtype>::BufferedColorJPGReader() {
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	LOG(INFO)<<"Call BufferedColorJPGReader()";
	Init(0,"");
	this->SetLrPostfix("_lr");
}

template <typename Dtype>
BufferedColorJPGReader<Dtype>::BufferedColorJPGReader(const string& data_name, int buff_size){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	LOG(INFO)<<"Call BufferedColorJPGReader(const string& data_name, int buff_size)";
	Init(buff_size,data_name);
	this->SetLrPostfix("_lr");
}

template <typename Dtype>
void BufferedColorJPGReader<Dtype>::ShowReadCvMat(int blob_id,const string& dst_filename){
	if(blob_id < this->buff_size_ && blob_id >= 0){
//		cvNamedWindow("image", CV_WINDOW_AUTOSIZE);
//		IplImage ipl_img = this->buffered_cv_mat_[blob_id];
//		cvShowImage("image",&ipl_img);
		char path[512];
		sprintf(path,"%s",dst_filename.c_str());
		imwrite(path,this->buffered_cv_mat_[blob_id]);
	}
}

template <typename Dtype>
bool BufferedColorJPGReader<Dtype>::ReadBufferedBlob(int blob_id,Blob<Dtype>& dst_blob){
	if(blob_id < this->buff_size_ && blob_id >= 0){

		return caffe::ReadImgToBlob(this->buffered_cv_mat_[blob_id],
				dst_blob, Dtype(0), Dtype(0), Dtype(0));
	}
	return false;
}

template <typename Dtype>
void BufferedColorJPGReader<Dtype>::WriteFileToCvMat(const string& filename,const int dst_mat_id)
{
	if(dst_mat_id < this->buff_size_ && dst_mat_id >= 0){
//		LOG(INFO)<<"WriteFileToCvMat   in";

		cv::Mat& dst_mat = this->buffered_cv_mat_[dst_mat_id];
		int cv_read_flag = CV_LOAD_IMAGE_COLOR;
		//dst_mat.release();
		dst_mat = cv::imread(filename, cv_read_flag);
		if (!dst_mat.data) {
			std::vector<std::string> splited_name= std_split(filename,".");
			int splited_name_end = splited_name.size()-1;
//			CHECK_GE(splited_name.size(),2);

			transform(splited_name[splited_name_end].begin(), splited_name[splited_name_end].end(),
					splited_name[splited_name_end].begin(), ::toupper);
			string new_filename  = "";
			for(int i=0; i < splited_name_end; ++i){
				new_filename += splited_name[i]+".";
			}
			new_filename  += splited_name[splited_name_end] ;

			dst_mat = cv::imread(new_filename, cv_read_flag);
			if (!dst_mat.data) {
				std::vector<std::string> splited_name= std_split(filename,this->lr_postfix_+".");
				CHECK_EQ(splited_name.size(),2);
				string non_lr_name  = splited_name[0] +"."+ splited_name[1] ;
				cv::Mat lr_dst_mat = cv::imread(non_lr_name, cv_read_flag);
				if (!lr_dst_mat.data) {
					transform(splited_name[1].begin(), splited_name[1].end(), splited_name[1].begin(), ::toupper);
					string new_filename  = splited_name[0] +"."+ splited_name[1] ;
					lr_dst_mat = cv::imread(new_filename, cv_read_flag);
					if (!lr_dst_mat.data){
						LOG(ERROR) << "Could not open or find file " << filename <<" or file "<<non_lr_name;
						return;
					}
				}
				cv::flip(lr_dst_mat,dst_mat,1);
				lr_dst_mat.release();
			}
		}

//		LOG(INFO)<<"					fuck: dim:"<<this->buffered_cv_mat_[dst_mat_id].dims<<"   in image: "<<filename;

//		LOG(INFO)<<"WriteFileToCvMat   out";
	}else{
		LOG(ERROR) << "Invalid dst_mat_id: " << dst_mat_id;
	}
}



template <typename Dtype>
void BufferedColorJPGReader<Dtype>::LoadToBlob(const string& filename,Blob<Dtype>& dst_blob){
	boost::unique_lock<boost::shared_mutex> table_lock(this->name_table_mutex_);
	int blob_idx = this->FindBlobIdxByName(filename);
	if(blob_idx >= 0 ){
		boost::shared_lock<boost::shared_mutex> blob_read_lock(
				*(this->read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
		ReadBufferedBlob(blob_idx,dst_blob);
		blob_read_lock.unlock();
	}else{
		blob_idx = this->UpdateNameTable(filename);
		boost::unique_lock<boost::shared_mutex> blob_write_lock(
				*(this->read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
		WriteFileToCvMat(filename,blob_idx);
		ReadBufferedBlob(blob_idx,dst_blob);
		blob_write_lock.unlock();
	}
}

template <typename Dtype>
bool BufferedColorJPGReader<Dtype>::ReadBufferedCvMat(int blob_id,cv::Mat& dst_mat){
	if(blob_id < this->buff_size_ && blob_id >= 0){
		dst_mat = this->buffered_cv_mat_[blob_id];
		return true;
	}
	return false;
}

template <typename Dtype>
void BufferedColorJPGReader<Dtype>::LoadToCvMat(const string& filename,cv::Mat& dst_mat){
	boost::unique_lock<boost::shared_mutex> table_lock(this->name_table_mutex_);
	int blob_idx = this->FindBlobIdxByName(filename);
	if(blob_idx >= 0 ){
		boost::shared_lock<boost::shared_mutex> blob_read_lock(
				*(this->read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
		ReadBufferedCvMat(blob_idx,dst_mat);
		blob_read_lock.unlock();
	}else{
		blob_idx = this->UpdateNameTable(filename);
		boost::unique_lock<boost::shared_mutex> blob_write_lock(
				*(this->read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
		WriteFileToCvMat(filename,blob_idx);
		ReadBufferedCvMat(blob_idx,dst_mat);
		blob_write_lock.unlock();
	}
}


template <typename Dtype>
void BufferedColorJPGReader<Dtype>::Show(const string& filename,const string& dst_filename ){
	boost::unique_lock<boost::shared_mutex> table_lock(this->name_table_mutex_);
	int blob_idx = this->FindBlobIdxByName(filename);
	if(blob_idx >= 0 ){
		boost::shared_lock<boost::shared_mutex> blob_read_lock(
				*(this->read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
		ShowReadCvMat(blob_idx,dst_filename);
		blob_read_lock.unlock();
	}else{
		blob_idx = this->UpdateNameTable(filename);
		boost::unique_lock<boost::shared_mutex> blob_write_lock(
				*(this->read_write_blobs_mutex_ptr_[blob_idx].get()));
		table_lock.unlock();
		WriteFileToCvMat(filename,blob_idx);
		ShowReadCvMat(blob_idx,dst_filename);
		blob_write_lock.unlock();
	}
}


/**
 * for BufferedColorJPGPairReader
 */

template <typename Dtype>
BufferedColorJPGPairReader<Dtype>::BufferedColorJPGPairReader() {
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	CHECK_EQ(this->buffered_cv_mat_.size(),this->buffered_cv_mat2_.size());
	Init(0,"");
	this->SetLrPostfix("_lr");
}

template <typename Dtype>
BufferedColorJPGPairReader<Dtype>::BufferedColorJPGPairReader(const string& data_name, int buff_size){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	CHECK_EQ(this->buffered_cv_mat_.size(),this->buffered_cv_mat2_.size());
	Init(buff_size,data_name);
	this->SetLrPostfix("_lr");
}

template <typename Dtype>
void BufferedColorJPGPairReader<Dtype>::Init(int buff_size,const string &data_name){
	this->buff_size_ = buff_size;
	this->data_name_ = data_name;
	while(this->buffered_cv_mat_.size() <  this->buff_size_){
		this->buffered_cv_mat_.push_back(cv::Mat());
		this->buffered_cv_mat2_.push_back(cv::Mat());
		CHECK_EQ(this->buffered_cv_mat_.size(),this->buffered_cv_mat2_.size());
	}
	while(this->read_write_blobs_mutex_ptr_.size() < this->buff_size_){
		this->read_write_blobs_mutex_ptr_.push_back(shared_ptr<boost::shared_mutex>(new boost::shared_mutex()));
	}
	this->name_map_.clear();
	this->name_list_.clear();
//	LOG(INFO)<<"				call  BufferedColorJPGPairReader<Dtype>::Init";
}


template <typename Dtype>
bool BufferedColorJPGPairReader<Dtype>::ReadBufferedBlob(int blob_id,Blob<Dtype>& dst_blob){
	if(blob_id < this->buff_size_ && blob_id >= 0){
		return caffe::ReadImgToBlob(this->buffered_cv_mat_[blob_id],
				this->buffered_cv_mat2_[blob_id],dst_blob);
	}
	return false;
}

template <typename Dtype>
void BufferedColorJPGPairReader<Dtype>::WriteFileToCvMat(const string& filename,const int dst_mat_id)
{
	if(dst_mat_id < this->buff_size_ && dst_mat_id >= 0){

		cv::Mat& dst_mat = this->buffered_cv_mat_[dst_mat_id];
		int cv_read_flag = CV_LOAD_IMAGE_COLOR;
		dst_mat = cv::imread(filename, cv_read_flag);
		if (!dst_mat.data) {
			LOG(ERROR) << "Could not open or find file " << filename;
		}
//		LOG(INFO)<<"					fuck: dim:"<<this->buffered_cv_mat_[dst_mat_id].dims<<"   in image: "<<filename;
		vector<string> split_str = caffe::std_split(filename,".");
		string new_name= split_str[0];
		for(int i = 1; i < split_str.size()-1; ++i){
			new_name.append("." + split_str[i]);
		}
		new_name.append(this->data_name_ + string(".jpg") );
		cv::Mat& dst_mat2 = this->buffered_cv_mat2_[dst_mat_id];

		dst_mat2 = cv::imread(new_name, cv_read_flag);
//		LOG(INFO)<<" dst_mat_id: "<<dst_mat_id<<"  buff_size: "<<this->buff_size_;

		if (!dst_mat2.data) {
			LOG(ERROR) << "Could not open or find file " << new_name;
		}

	}else{
		LOG(ERROR) << "Invalid dst_mat_id: " << dst_mat_id;
	}
}

/**
 * for BufferedColorIMGAndAVIReader
 */
template <typename Dtype>
BufferedColorIMGAndAVIReader<Dtype>::BufferedColorIMGAndAVIReader(){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	BufferedColorJPGReader<Dtype>::Init(1,"");
	cv_read_color_flag = CV_LOAD_IMAGE_COLOR;
	need_video_buffer = false;
	SetExtensionNames();
	this->SetLrPostfix("_lr");
}

template <typename Dtype>
BufferedColorIMGAndAVIReader<Dtype>::BufferedColorIMGAndAVIReader(const string& data_name,
		int buff_size){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	BufferedColorJPGReader<Dtype>::Init(buff_size,data_name);
	cv_read_color_flag = CV_LOAD_IMAGE_COLOR;
	if(data_name ==  string("")){
		need_video_buffer = false;
	}else{
		need_video_buffer = true;
	}
	this->SetLrPostfix("_lr");
	SetExtensionNames();
}

template <typename Dtype>
void BufferedColorIMGAndAVIReader<Dtype>::SetExtensionNames(){
	img_extension_names_.clear();
	video_extension_names_.clear();
	img_extension_names_.push_back(string("jpg"));
	img_extension_names_.push_back(string("png"));
	img_extension_names_.push_back(string("bmp"));
	video_extension_names_.push_back(string("avi"));
}

template <typename Dtype>
void BufferedColorIMGAndAVIReader<Dtype>::SetColorFlag(int flag){
	cv_read_color_flag = flag;
}

template <typename Dtype>
bool BufferedColorIMGAndAVIReader<Dtype>::ParseFileName(const string& filename,
		string& parsed_filename,bool& is_img, long& frame_id){
	std::vector<std::string> splited_name= std_split(filename,".");
	string file_extension = splited_name[splited_name.size()-1];
	for(int i=0; i < img_extension_names_.size(); ++i){
		if(file_extension.find(img_extension_names_[i],0) != string::npos){
			parsed_filename = filename;
			is_img = true;
			frame_id = 0;
			return true;
		}
	}
	for(int i=0; i < video_extension_names_.size(); ++i){
		if(file_extension.find(video_extension_names_[i],0) != string::npos){
			stringstream ss;
			std::vector<std::string> splited_extension = std_split(file_extension,"_");
			ss << splited_extension[splited_extension.size()-1];
			ss >> frame_id;
			is_img = false;
			parsed_filename = splited_name[0] + "." + video_extension_names_[i];
			return true;
		}
	}
	return false;
}

template <typename Dtype>
bool BufferedColorIMGAndAVIReader<Dtype>::BufferedReadFrameFromVideo(const string& filename, const long frame_id,cv::Mat& dst){

	if(need_video_buffer){
		char buffered_name[256];
		int cur_frame_id = frame_id;
		std::vector<std::string> splited_name= std_split(filename,"/");
		string out_name;
		for(int i=0 ; i < splited_name.size(); ++i){
			out_name = out_name + "_"+ splited_name[i] ;
		}

		sprintf(buffered_name,"%s%s.%d.jpg",this->data_name_.c_str(),out_name.c_str(),cur_frame_id);
		dst = cv::imread(buffered_name, cv_read_color_flag);
		if (!dst.data) {
			bool result = ReadFrameFromVideo(filename, frame_id,dst);
			imwrite(buffered_name,dst);
			LOG(INFO)<<"write buffer: "<< buffered_name;
			return result;
		}
		return true;
	}else{
		return ReadFrameFromVideo(filename, frame_id,dst);
	}
}

template <typename Dtype>
void BufferedColorIMGAndAVIReader<Dtype>::WriteFileToCvMat(const string& filename,const int dst_mat_id)
{

	if(dst_mat_id < this->buff_size_ && dst_mat_id >= 0){
		cv::Mat& dst_mat = this->buffered_cv_mat_[dst_mat_id];
		string parsed_filename;
		bool is_img;
		long frame_id;
		if(!ParseFileName( filename, parsed_filename, is_img, frame_id)){
			LOG(ERROR) << "Could not parse filename " << filename;
			return;
		}

		if(is_img){
			dst_mat = cv::imread(parsed_filename, cv_read_color_flag);
			if (!dst_mat.data) {
				LOG(ERROR) << "Could not open or find file " << filename;
				return;
			}
			return;
		}
		else{
			if(cv_read_color_flag == CV_LOAD_IMAGE_COLOR){
				if(!BufferedReadFrameFromVideo(parsed_filename, frame_id,dst_mat)){
					LOG(ERROR) << "Could not open or find file " << filename;
					return;
				}
			}else{
				cv::Mat temp_mat;
				if(!BufferedReadFrameFromVideo(parsed_filename, frame_id,temp_mat)){
					LOG(ERROR) << "Could not open or find file " << filename;
					return;
				}
				cv::cvtColor(temp_mat,dst_mat,CV_BGR2GRAY);
			}
		}
	}else{
		LOG(ERROR) << "Invalid dst_mat_id: " << dst_mat_id;
	}
}




INSTANTIATE_CLASS(BufferedColorIMGAndAVIReader);
INSTANTIATE_CLASS(BufferedColorJPGPairReader);
INSTANTIATE_CLASS(BufferedColorJPGReader);
INSTANTIATE_CLASS(BufferedHDF5Reader);

}  // namespace caffe
