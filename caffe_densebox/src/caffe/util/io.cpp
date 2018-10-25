#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <sys/stat.h>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}
int CreateDir(const char *sPathName, int beg) {
	char DirName[256];
	strcpy(DirName, sPathName);
	int i, len = strlen(DirName);
	if (DirName[len - 1] != '/')
		strcat(DirName, "/");

	len = strlen(DirName);

	for (i = beg; i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			if (access(DirName, F_OK) != 0) {
        if (mkdir(DirName, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
          LOG(ERROR)<< "Failed to create folder " << sPathName;
        }
			}
			DirName[i] = '/';
		}
	}

	return 0;
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}


template <typename Dtype>
cv::Mat BlobImgDataToCVMat(const Blob<Dtype>& data,int sampleID, const Dtype  mean_b ,
		const Dtype mean_g  , const Dtype mean_r,int start_c )
{
	CHECK_GE(data.channels(), 3+start_c);
	CHECK_LT(sampleID, data.num());
	cv::Mat cv_img_original(data.height(), data.width(),CV_8UC3);
	Dtype mean_bgr[3];
	mean_bgr[0] = mean_b;
	mean_bgr[1] = mean_g;
	mean_bgr[2] = mean_r;
	const Dtype* blob_data = data.cpu_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < data.height(); ++h) {
			for (int w = 0; w < data.width(); ++w) {
				uchar value = static_cast<uchar>(MIN(blob_data[data.offset(sampleID, c+start_c, h, w)]
										+ mean_bgr[c],255));
				value = MAX(0,value);
				cv_img_original.at<cv::Vec3b>(h, w)[c]= value;
			}
		}
	}
	return cv_img_original;
}

template cv::Mat BlobImgDataToCVMat(const Blob<float>& data,int sampleID, const float  mean_b ,
		const float mean_g  , const float mean_r ,int start_c );

template cv::Mat BlobImgDataToCVMat(const Blob<double>& data,int sampleID, const double  mean_b ,
		const double mean_g  , const double mean_r ,int start_c );


template <typename Dtype>
bool ReadImgToBlob(const string& filename, Blob<Dtype>& dst,
		Dtype mean_c1, Dtype mean_c2, Dtype mean_c3){

	int cv_read_flag = CV_LOAD_IMAGE_COLOR;
	Dtype mean[3];
	mean[0] = mean_c1;
	mean[1] = mean_c2;
	mean[2] = mean_c3;
	cv::Mat cv_img = cv::imread(filename, cv_read_flag);
	if (!cv_img.data) {
//		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	int num_channels = 3;
	dst.Reshape(1,3,cv_img.rows,cv_img.cols);
	Dtype* dst_data = dst.mutable_cpu_data();
	for (int c = 0; c < num_channels; ++c) {
	  for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
		  dst_data[(c*cv_img.rows + h)*cv_img.cols + w]=(
			static_cast<Dtype>(cv_img.at<cv::Vec3b>(h, w)[c])) - mean[c];
		}
	  }
	}
	return true;
}



template bool ReadImgToBlob(const string& filename, Blob<float>& dst,
		float mean_c1, float mean_c2, float mean_c3);
template bool ReadImgToBlob(const string& filename, Blob<double>& dst,
		double mean_c1, double mean_c2, double mean_c3);

template <typename Dtype>
bool ReadImgToBlob(const cv::Mat & cv_img, Blob<Dtype>& dst,
		Dtype mean_c1, Dtype mean_c2, Dtype mean_c3){


	Dtype mean[3];
	mean[0] = mean_c1;
	mean[1] = mean_c2;
	mean[2] = mean_c3;
	if (!cv_img.data) {
	LOG(ERROR) << "param cv_img has no contents"  ;
		return false;
	}
	int num_channels = cv_img.channels();;
	dst.Reshape(1,num_channels,cv_img.rows,cv_img.cols);
	Dtype* dst_data = dst.mutable_cpu_data();
	for (int c = 0; c < num_channels; ++c) {
	  for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
		  dst_data[(c*cv_img.rows + h)*cv_img.cols + w]=(
			static_cast<Dtype>(cv_img.at<cv::Vec3b>(h, w)[c])) - mean[c];
		}
	  }
	}
	return true;
}

template bool ReadImgToBlob(const cv::Mat & cv_img, Blob<float>& dst,
		float mean_c1, float mean_c2, float mean_c3);
template bool ReadImgToBlob(const cv::Mat & cv_img, Blob<double>& dst,
		double mean_c1, double mean_c2, double mean_c3);


template <typename Dtype>
bool ReadImgToBlob(const cv::Mat & cv_img_1,const cv::Mat & cv_img_2, Blob<Dtype>& dst){
if (!cv_img_1.data  || !cv_img_2.data ) {
	LOG(ERROR) << "param cv_img_1 or cv_img_2 has no contents"  ;
		return false;
	}
	CHECK_EQ(cv_img_1.cols,cv_img_2.cols);
	CHECK_EQ(cv_img_1.rows,cv_img_2.rows);
	dst.Reshape(1,cv_img_1.channels() + cv_img_2.channels(),cv_img_1.rows,cv_img_1.cols);
	Dtype* dst_data = dst.mutable_cpu_data();
	for (int c = 0; c < cv_img_1.channels(); ++c) {
	  for (int h = 0; h < cv_img_1.rows; ++h) {
		for (int w = 0; w < cv_img_1.cols; ++w) {
		  *(dst_data++)=static_cast<Dtype>(cv_img_1.at<cv::Vec3b>(h, w)[c]);
		}
	  }
	}
	for (int c = 0; c < cv_img_2.channels(); ++c) {
	  for (int h = 0; h < cv_img_2.rows; ++h) {
		for (int w = 0; w < cv_img_2.cols; ++w) {
		  *(dst_data++)=static_cast<Dtype>(cv_img_2.at<cv::Vec3b>(h, w)[c]);
		}
	  }
	}
	return true;
}

template bool ReadImgToBlob(const cv::Mat & cv_img_1,const cv::Mat & cv_img_2, Blob<float>& dst);
template bool ReadImgToBlob(const cv::Mat & cv_img_1,const cv::Mat & cv_img_2, Blob<double>& dst);



bool ReadDepthImg(const string& filename,cv::Mat & cvmat){

	FILE* file_handle = NULL;
	file_handle = fopen(filename.c_str(),"r");
	if(file_handle == NULL){
		LOG(ERROR)<<"can not open file "<<filename;
		return false;
	}
	int dim,height,width,count;
	if(fscanf(file_handle,"%d %d %d ",&dim,&height,&width) != 3){
		LOG(ERROR)<<"Error depthimg head_info structure "<<filename;
		return false;
	}
	if(dim != 4){
		LOG(ERROR)<<"Error depthimg dim number "<<filename;
		return false;
	}
	count = 0;
	cvmat = cv::Mat(height, width,  CV_32FC4);
	float temp;

	while( fscanf(file_handle, "%f",&temp) == 1){
		int cur_w = count%width;
		int cur_c = count/(height*width);
		int cur_h = (count%(height*width))/width;
		cvmat.at<cv::Vec4f>(cur_h,cur_w)[cur_c] = temp;
		count += 1;
	}
	if(count != height*width*dim){
		LOG(ERROR)<<"Error depthimg data structure "<<filename;
		return false;
	}
	return true;
}

void DepthCVMatToTwoCvMat(const cv::Mat & src, cv::Mat &dst1, cv::Mat &dst2){
	dst1 = cv::Mat(src.rows, src.cols,CV_8UC3);
	dst2 = dst1.clone();
	for (int h = 0; h < src.rows; ++h) {
		for (int w = 0; w < src.cols; ++w) {
			dst1.at<cv::Vec3b>(h, w)[0]= static_cast<uchar>(src.at<cv::Vec4f>(h, w)[0]) ;
			dst1.at<cv::Vec3b>(h, w)[1]= static_cast<uchar>(src.at<cv::Vec4f>(h, w)[1]) ;
			dst1.at<cv::Vec3b>(h, w)[2]= static_cast<uchar>(src.at<cv::Vec4f>(h, w)[2]) ;
			dst2.at<cv::Vec3b>(h, w)[0]= static_cast<uchar>(src.at<cv::Vec4f>(h, w)[3]) ;
			dst2.at<cv::Vec3b>(h, w)[1]= static_cast<uchar>(src.at<cv::Vec4f>(h, w)[3]) ;
			dst2.at<cv::Vec3b>(h, w)[2]= static_cast<uchar>(src.at<cv::Vec4f>(h, w)[3]) ;
		}
	}
}

template <typename Dtype>
void LoadHDF5FileToBlob(const char* filename,vector<string> names,std::vector<shared_ptr<Blob<Dtype> > > &hdf_blobs){
	DLOG(INFO) << "Loading HDF5 file: " << filename;
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) {
		LOG(FATAL) << "Failed opening HDF5 file: " << filename;
	}
	CHECK(names.size()>0);
	hdf_blobs.resize(names.size());
	const int MIN_DATA_DIM = 1;
    const int MAX_DATA_DIM = INT_MAX;
	for (int i = 0; i < names.size(); ++i) {
		hdf_blobs[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
		hdf5_load_nd_dataset(file_id, names[i].c_str(),
			MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs[i].get());
	}
	herr_t status = H5Fclose(file_id);
	CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
}

template void LoadHDF5FileToBlob(const char* filename,vector<string> names,std::vector<shared_ptr<Blob<float> > > &hdf_blobs);
template void LoadHDF5FileToBlob(const char* filename,vector<string> names,std::vector<shared_ptr<Blob<double> > > &hdf_blobs);

template <typename Dtype>
void LoadHDF5FileToBlob(const char* filename,const string&  names,
		Blob<Dtype> &hdf_blobs){

	DLOG(INFO) << "Loading HDF5 file: " << filename;
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) {
		LOG(FATAL) << "Failed opening HDF5 file: " << filename;
	}
	const int MIN_DATA_DIM = 1;
	const int MAX_DATA_DIM = INT_MAX;
	hdf5_load_nd_dataset(file_id, names.c_str(),
			MIN_DATA_DIM, MAX_DATA_DIM, &hdf_blobs);
	herr_t status = H5Fclose(file_id);
	CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
}

template  void LoadHDF5FileToBlob(const char* filename,const string&  names, Blob<float> &hdf_blobs);
template  void LoadHDF5FileToBlob(const char* filename,const string&  names, Blob<double> &hdf_blobs);


template <typename Dtype>
void LoadHDF5FileToBlob(const char* filename,const string&  names,
		Blob<Dtype> &hdf_blobs, boost::shared_mutex &mutex){

	boost::unique_lock<boost::shared_mutex> lock(mutex);
	DLOG(INFO) << "Loading HDF5 file: " << filename;
	hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	lock.unlock();
	if (file_id < 0) {
		LOG(FATAL) << "Failed opening HDF5 file: " << filename;
	}
	const int MIN_DATA_DIM = 1;
	const int MAX_DATA_DIM = INT_MAX;

	lock.lock();
	hdf5_load_nd_dataset(file_id, names.c_str(),
			MIN_DATA_DIM, MAX_DATA_DIM, &hdf_blobs);
	lock.unlock();
	lock.lock();
	herr_t status = H5Fclose(file_id);
	lock.unlock();
	CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
}

template  void LoadHDF5FileToBlob(const char* filename,const string&  names, Blob<float> &hdf_blobs, boost::shared_mutex &mutex);
template  void LoadHDF5FileToBlob(const char* filename,const string&  names, Blob<double> &hdf_blobs, boost::shared_mutex &mutex);


bool ReadFrameFromVideo(const string filename, const long frame_id,cv::Mat& dst){

	cv::VideoCapture capture(filename);
	if(!capture.isOpened()){
		LOG(INFO)<<"Failed to open video "<< filename;
		return false;
	}
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	long frameToStart = frame_id;
	if(frameToStart >= totalFrameNumber){
		LOG(INFO)<<"Frame_id "<< frameToStart <<" exceeds total frame number "<< totalFrameNumber;
		capture.release();
		return false;
	}
	capture.set( CV_CAP_PROP_POS_FRAMES,frameToStart);
	if(!capture.read(dst)){
		capture.release();
		LOG(INFO)<<"Failed to load frame_id "<< frameToStart <<" total frame_num "<<totalFrameNumber;
		return false;
	}
	capture.release();
	return true;
}



#endif  // USE_OPENCV
}  // namespace caffe
