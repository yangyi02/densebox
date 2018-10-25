#ifndef CAFFE_BUFFERED_READER_H_
#define CAFFE_BUFFERED_READER_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include <map>


namespace caffe {

template <typename Dtype>
class BufferedImgReader {
public:
	BufferedImgReader();
	BufferedImgReader(const string& data_name, int buff_size);
	virtual ~BufferedImgReader();

	virtual void Init(int buff_size,const string &data_name);
	virtual void SetDataName(const string &data_name);
	virtual void LoadToBlob(const string& filename, Blob<Dtype>& dst_blob);
	virtual void SetLrPostfix(const string& postfix);
protected:

	// return the blob_idx for filename
	int UpdateNameTable(const string& filename);
	virtual int  GetOverWriteIdx();
	virtual void WriteFileToBlob(const string& filename,Blob<Dtype>& dst_blob) = 0;
	// return false if blob is empty or blob_id is invalid
	virtual bool ReadBufferedBlob(int blob_id,Blob<Dtype>& dst_blob);
	// return blob_id. If can not find , return -1.
	int  FindBlobIdxByName(const string& name);
	unsigned int PrefetchRand();

	shared_ptr<Caffe::RNG> prefetch_rng_;
	int buff_size_;
	int round_counter_;

	std::vector<shared_ptr<Blob<Dtype> > > buffered_blobs_ptr_;
	vector< shared_ptr<boost::shared_mutex> > read_write_blobs_mutex_ptr_;
	map<string,int> name_map_;
	vector<string> name_list_;

//	map<hid_t,int> valid_hdf5_file_id_;

	boost::shared_mutex name_table_mutex_;
	string data_name_;
	string lr_postfix_;

//	boost::shared_mutex hdf5_mutex_;

};


template <typename Dtype>
class BufferedHDF5Reader :  virtual public BufferedImgReader<Dtype>{
public:
	BufferedHDF5Reader():BufferedImgReader<Dtype>(){
		exclusive_load_ = false;
	};
	BufferedHDF5Reader(const string& data_name, int buff_size)
	:BufferedImgReader<Dtype>(data_name,buff_size){
		exclusive_load_ = false;
	}
	virtual ~BufferedHDF5Reader(){};
protected:
	virtual void WriteFileToBlob(const string& filename,Blob<Dtype>& dst_blob);
	boost::shared_mutex hdf5_mutex_;
	bool exclusive_load_;
};

template <typename Dtype>
class BufferedColorJPGReader : virtual  public BufferedImgReader<Dtype>{
public:
	BufferedColorJPGReader();
	BufferedColorJPGReader(const string& data_name, int buff_size);
	virtual ~BufferedColorJPGReader(){};
	virtual void Init(int buff_size,const string &data_name);
	virtual void LoadToBlob(const string& filename, Blob<Dtype>& dst_blob);
	virtual void LoadToCvMat(const string& filename, cv::Mat& dst_mat);
	virtual void Show(const string& filename, const string& dst_filename);
protected:
	virtual void WriteFileToBlob(const string& filename,Blob<Dtype>& dst_blob){};
	virtual bool ReadBufferedBlob(int blob_id,Blob<Dtype>& dst_blob);
	virtual bool ReadBufferedCvMat(int blob_id,cv::Mat& dst_mat);
	virtual void WriteFileToCvMat(const string& filename,const int dst_mat_id);

	virtual void ShowReadCvMat(int blob_id,const string& dst_filename);

	std::vector<cv::Mat > buffered_cv_mat_;
};

template <typename Dtype>
class BufferedColorIMGAndAVIReader :  virtual public BufferedColorJPGReader<Dtype>{
public:
	BufferedColorIMGAndAVIReader();
	BufferedColorIMGAndAVIReader(const string& data_name, int buff_size);
	virtual ~BufferedColorIMGAndAVIReader(){};
	bool ParseFileName(const string& filename, string& parsed_filename,bool& is_img, long& frame_id);
	void SetColorFlag(int flag);
	void SetVideoBufferFlag(int flag);
protected:
	virtual void WriteFileToCvMat(const string& filename,const int dst_mat_id);

	void SetExtensionNames();

	bool BufferedReadFrameFromVideo(const string& filename, const long frame_id,cv::Mat& dst);

	vector<string> img_extension_names_;
	vector<string> video_extension_names_;

	//CV_LOAD_IMAGE_GRAYSCALE or  CV_LOAD_IMAGE_COLOR
	int cv_read_color_flag;

	bool need_video_buffer;
};



template <typename Dtype>
class BufferedColorJPGPairReader :   virtual public BufferedColorJPGReader<Dtype>{
public:
	BufferedColorJPGPairReader();
	BufferedColorJPGPairReader(const string& data_name, int buff_size);
	virtual ~BufferedColorJPGPairReader(){};
	virtual void Init(int buff_size,const string &data_name);

protected:
	virtual void WriteFileToBlob(const string& filename,Blob<Dtype>& dst_blob){};
	virtual bool ReadBufferedBlob(int blob_id,Blob<Dtype>& dst_blob);
	virtual void WriteFileToCvMat(const string& filename,const int dst_mat_id);

	std::vector<cv::Mat > buffered_cv_mat2_;
};

}  // namespace caffe

#endif   // CAFFE_BUFFERED_READER_H_
