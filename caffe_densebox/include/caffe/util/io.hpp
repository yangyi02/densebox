#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"
//#include "hdf5.h"
//#include "hdf5_hl.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/blob.hpp"
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
    boost::filesystem::temp_directory_path()/"caffe_test.%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

bool ReadProtoFromTextFile(const char* filename, Message* proto);




inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

bool DecodeDatumNative(Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);
int CreateDir(const char *sPathName, int beg = 1); // by zhujin
#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);



template <typename Dtype>
cv::Mat BlobImgDataToCVMat(const Blob<Dtype>& blob,int sampleID, const Dtype  mean_b = 0,
		const Dtype mean_g = 0, const Dtype mean_r = 0,int start_c = 0);

template <typename Dtype>
bool ReadImgToBlob(const string& filename, Blob<Dtype>& dst,
		Dtype mean_c1, Dtype mean_c2, Dtype mean_c3);

/**
 * depth img format: [dim height width data.....]
 * and the data is ordered in BGRD order:  BBBBBBBBBBGGGGGGGGGGRRRRRRRRRRRDDDDDDDD
 */
bool ReadDepthImg(const string& filename,cv::Mat & cvmat);

void DepthCVMatToTwoCvMat(const cv::Mat & src, cv::Mat &dst1, cv::Mat &dst2);

template <typename Dtype>
bool ReadImgToBlob(const cv::Mat & cv_img, Blob<Dtype>& dst,
		Dtype mean_c1, Dtype mean_c2, Dtype mean_c3);

template <typename Dtype>
bool ReadImgToBlob(const cv::Mat & cv_img_1,const cv::Mat & cv_img_2, Blob<Dtype>& dst);



template <typename Dtype>
void LoadHDF5FileToBlob(const char* filename,vector<string> names,
		std::vector<shared_ptr<Blob<Dtype> > > &hdf_blobs);

template <typename Dtype>
void LoadHDF5FileToBlob(const char* filename,const string&  names,
		Blob<Dtype> &hdf_blobs);

template <typename Dtype>
void LoadHDF5FileToBlob(const char* filename,const string&  names,
		Blob<Dtype> &hdf_blobs, boost::shared_mutex &mutex);

bool ReadFrameFromVideo(const string filename, const long frame_id,cv::Mat& dst);



#endif  // USE_OPENCV




}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
