#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/blob_transform.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class BlobTransTest : public ::testing::Test {
 protected:
	BlobTransTest()
      : blob_(new Blob<Dtype>()),
        blob_preshaped_(new Blob<Dtype>(2, 3, 4, 5)) {
	  FillerParameter filler_param;
	  filler_param.set_min(-3);
	  filler_param.set_max(3);
	  filler = new UniformFiller<Dtype>(filler_param);
	}
  virtual ~BlobTransTest() { delete blob_; delete blob_preshaped_; }
  Blob<Dtype>* const blob_;
  Blob<Dtype>* const blob_preshaped_;
  Blob<Dtype> a, b, c;
  FillerParameter filler_param;
  UniformFiller<Dtype> *filler;
};

TYPED_TEST_CASE(BlobTransTest, TestDtypes);

TYPED_TEST(BlobTransTest, TestAdd) {

  this->a.Reshape(2,2,2,2);
  this->b.ReshapeLike(this->a);
  this->c.ReshapeLike(this->a);
  this->filler->Fill(&(this->a));
  this->filler->Fill(&(this->b));
  this->filler->Fill(&(this->c));
  int device = 1;
  if( Caffe::mode() == Caffe::CPU){
  	device = 1;
  }else{
  	device = 2;
  }

  fxnet_transform<op::add>(this->a,this->b,this->c);

  const TypeParam * data_a = this->a.cpu_data();
  const TypeParam * data_b = this->b.cpu_data();
  const TypeParam * data_c = this->c.cpu_data();

  for(int i=0; i < this->a.count(); ++i){
  	EXPECT_NEAR(data_a[i]+data_b[i], data_c[i], 1e-4);
  }

}

TYPED_TEST(BlobTransTest, TestBoundByMinMax) {

  this->a.Reshape(2,2,2,2);
  this->b.ReshapeLike(this->a);
  this->c.ReshapeLike(this->a);
  this->filler->Fill(&(this->a));
  this->filler->Fill(&(this->b));
  this->filler->Fill(&(this->c));

  TypeParam bound_scale = TypeParam(256 -1);
  fxnet_transform<op::BoundByMinMax>(this->a,this->b,bound_scale);

  const TypeParam * data_a = this->a.cpu_data();
  const TypeParam * data_b = this->b.cpu_data();
  for(int i=0; i < this->a.count(); ++i){
  	EXPECT_NEAR(data_a[i] , data_b[i], 0.05);
  }
  bound_scale = TypeParam(128 -1);
  fxnet_transform<op::BoundByMaxAbs>(this->a,this->b,bound_scale);

  const TypeParam * data_a1 = this->a.cpu_data();
  const TypeParam * data_b1 = this->b.cpu_data();
//
  for(int i=0; i < this->a.count(); ++i){
  	EXPECT_NEAR(data_a1[i] , data_b1[i], 0.05);
  }

}




}  // namespace caffe
