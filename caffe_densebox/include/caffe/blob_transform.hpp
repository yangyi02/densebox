/*
 * blob_transform.hpp
 *
 *  Created on: 2015年12月9日
 *      Author: Alan_Huang
 */

#ifndef CAFFE_BLOB_TRANSFORM_HPP_
#define CAFFE_BLOB_TRANSFORM_HPP_


#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include <float.h>
#include <iostream>
#ifdef FXNET_XINLINE
	#error "FXNET_XINLINE must not be defined"
#endif

#ifdef _MSC_VER
#define FXNET_FORCE_INLINE __forceinline
//#pragma warning(disable : 4068)
#else
#define FXNET_FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifdef __CUDACC__
  #define FXNET_XINLINE FXNET_FORCE_INLINE __device__ __host__
#else
  #define FXNET_XINLINE FXNET_FORCE_INLINE
#endif
#define FXNET_CINLINE FXNET_FORCE_INLINE

namespace caffe {

namespace op{

	struct add{
		template < typename Dtype>
		FXNET_XINLINE static void Do(Blob<Dtype>& a, Blob<Dtype>& b, Blob<Dtype>& c){
			caffe::caffe_add(a.count(), a.cpu_data(), b.cpu_data(), c.mutable_cpu_data());
		}
	};

	struct min_and_max{
		template<typename Dtype>
		FXNET_XINLINE static void Do(Blob<Dtype>&a, Dtype &min, Dtype& max){
			Dtype* data = a.mutable_cpu_data();
			for(int i=0; i < a.count(); ++i){
				max = std::max(data[i], max);
				min = std::min(data[i], min);
			}
		}
	};

	struct BoundByMinMax{
		template<typename Dtype>
		FXNET_CINLINE static void Do(Blob<Dtype>&src, Blob<Dtype>& dst, const Dtype bound_scale){
			Dtype min = FLT_MAX, max = -FLT_MAX;
			min_and_max::Do(src, min, max);

			if( max - min < 0.001 ){
				std::cout<<"In BoundByMinMax, max and min are almost the same: max="<<max<<
						" , min="<<min<<" ."<<std::endl;
				min = 0;
				max = bound_scale;
			}

			const Dtype* in_data = src.cpu_data();
			Dtype* out_data = dst.mutable_cpu_data();
			int count = src.count();
			for(int i=0; i < count; ++i){
				Dtype temp = Dtype (int( (in_data[i]-min)/(max-min)*bound_scale ) );
//				std::cout<<"data["<<i<<"]: after scaling: "<<temp<<std::endl;
				out_data[i] = temp / bound_scale * (max-min) + min;
			}
		}
	};

	struct BoundByMaxAbs{
		template<typename Dtype>
		FXNET_CINLINE static void Do(Blob<Dtype>&src, Blob<Dtype>& dst, const Dtype bound_scale){
			Dtype min = FLT_MAX, max = -FLT_MAX;
			min_and_max::Do(src, min, max);
			Dtype abs_max = std::max(std::abs(min), std::abs(max));
			if( abs_max < 0.001 ){
				std::cout<<"In BoundByMinMax, abs_max is too small: abs_max="<<abs_max <<std::endl;
				abs_max = bound_scale;
			}
			const Dtype* in_data = src.cpu_data();
			Dtype* out_data = dst.mutable_cpu_data();
			int count = src.count();
			for(int i=0; i < count; ++i){
				Dtype temp = Dtype (int( (in_data[i])/(abs_max)*bound_scale ) );
//				std::cout<<"data["<<i<<"]: after scaling: "<<temp<<std::endl;
				out_data[i] = temp / bound_scale * abs_max;
			}
		}
	};

}

template<typename Dtype>
struct Expression{
	inline Dtype& self(void){
		return *static_cast<Dtype*>(this);
	}
	inline Dtype* ptrself(void){
		return static_cast<Dtype*>(this);
	}
};

template<typename OP, typename Tsrc>
struct UnaryOpExpression : public Expression<UnaryOpExpression<OP, Tsrc> >{
	Tsrc & src_var_;
	explicit UnaryOpExpression(Tsrc& src_var): src_var_(src_var){};
};

template<typename OP, typename Tleft, typename Tright>
struct BinaryOpExpression : public Expression<BinaryOpExpression<OP, Tleft, Tright> >{
	 Tleft & l_var_;
	 Tright & r_var_;
	explicit BinaryOpExpression( Tleft& l_var,
			 Tright& r_var): l_var_(l_var), r_var_(r_var){}
};

template<typename OP, typename TA, typename TB, typename TC>
struct TripleOpExpression : public Expression<TripleOpExpression< OP, TA, TB, TC> >{
	 TA & a_var_;
	 TB & b_var_;
	 TC & c_var_;
	explicit TripleOpExpression( TA& a_var,
			 TB& b_var, TC& c_var): a_var_(a_var), b_var_(b_var), c_var_(c_var){}
};


template<typename OPType>
class Trans{
public:
	FXNET_XINLINE void Do();
};

template< typename OP, typename Tsrc>
class Trans<UnaryOpExpression<OP, Tsrc> >{
public:
	explicit Trans( const UnaryOpExpression<OP, Tsrc>& src): src_(src){
	}
	FXNET_XINLINE void Do(){
		OP::Do(src_.src_var_);
	}
protected:
	UnaryOpExpression<OP, Tsrc> src_;
};

template<typename OP, typename Tleft, typename Tright>
class Trans<BinaryOpExpression<OP, Tleft, Tright> >{
public:
	explicit Trans( const BinaryOpExpression<OP, Tleft, Tright>& src): src_(src){
	}
	FXNET_XINLINE void Do(){
		OP::Do(src_.l_var_, src_.r_var_);
	}
protected:
	BinaryOpExpression<OP, Tleft, Tright> src_;
};

template<typename OP, typename TA, typename TB, typename TC>
class Trans<TripleOpExpression<OP, TA, TB, TC> >{
public:
	explicit Trans( const TripleOpExpression<OP, TA, TB, TC>& src): src_(src){
	}
	FXNET_XINLINE void Do(){
		OP::Do(src_.a_var_, src_.b_var_, src_.c_var_);
	}
protected:
	TripleOpExpression<OP, TA, TB, TC> src_;
};


template<typename OP, typename TA, typename TB, typename TC>
inline void fxnet_transform( TA& a, TB& b, TC& c){
	Trans<TripleOpExpression<OP, TA, TB, TC> > temp(
			TripleOpExpression<OP, TA, TB, TC>(a, b,c));
	temp.Do();
}


}


#endif /* CAFFE_BLOB_TRANSFORM_HPP_ */
