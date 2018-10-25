#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/util/insert_inceptions.hpp"

namespace caffe {
void InsertInceptions(const NetParameter& param, NetParameter* param_split)
{
	  param_split->CopyFrom(param);
	  param_split->clear_layer();
	  for (int i = 0; i < param.layer_size(); ++i){
		  const LayerParameter& layer_param = param.layer(i);
		  if(layer_param.type().compare(string("Inception")) == 0){
			  CHECK(layer_param.has_inception_param());
			  InsertInception(layer_param.inception_param(),layer_param,param_split);
		  }
		  else
			  param_split->add_layer()->CopyFrom(param.layer(i));
	  }
}


void InsertInception(const InceptionParameter& inception_param,
		const LayerParameter layer_param,NetParameter* param_split)
{
	CHECK_EQ(layer_param.bottom_size(),1);
	CHECK_EQ(layer_param.top_size(),1);
	CHECK(layer_param.has_inception_param());
	bool relu_at_top = inception_param.relu_at_top();
	string bottom_name = layer_param.bottom(0);
	vector<string> column_top_names;
	column_top_names.clear();

	for(int i = 0; i < inception_param.inception_column_size(); ++i){
		column_top_names.push_back(InsertInceptionColumn(layer_param,inception_param,bottom_name,layer_param.name(),
				inception_param.inception_column(i),param_split));
		LOG(INFO)<<"add col "<<column_top_names[i] <<" to concat";
	}
	LayerParameter* added_layer_param = NULL;
	if(inception_param.inception_column_size() > 1){
		added_layer_param = param_split->add_layer();
		added_layer_param->Clear();
		for(int i = 0; i < inception_param.inception_column_size(); ++i){
			added_layer_param->add_bottom(column_top_names[i]);
			LOG(INFO)<<"add bottom "<<column_top_names[i] <<" to concat";
		}

		added_layer_param->set_type("Concat");
		added_layer_param->set_name(layer_param.name()+string("_concat"));
		added_layer_param->add_top(layer_param.top(0));
	}
	else{

		param_split->mutable_layer(param_split->layer_size()-1)->set_top(0,layer_param.top(0));
	}

	if(relu_at_top)
	{
		added_layer_param = param_split->add_layer();
		added_layer_param->Clear();
		added_layer_param->add_bottom(layer_param.top(0));
		added_layer_param->add_top(layer_param.top(0));
		added_layer_param->set_name(layer_param.name()+string("_relu"));
		added_layer_param->set_type("ReLU");
		if(inception_param.has_relu_param()){
			added_layer_param->mutable_relu_param()->CopyFrom(inception_param.relu_param());
		}
	}
}

string InsertInceptionColumn(const LayerParameter& layer_param,const InceptionParameter& inception_param,
		const string& bottom_name, const string& layer_name, const InceptionColumnParameter& inception_column,
		NetParameter* param_split)
{
	string cur_bottom_name = bottom_name;
	string column_name = inception_column.column_name();
	bool need_relu = inception_param.need_relu();

	/**
	 * set up pooling
	 */
	if(inception_column.has_pooling_param()){
		LayerParameter* added_layer_param = param_split->add_layer();
		added_layer_param->Clear();
		added_layer_param->add_bottom(cur_bottom_name);
		added_layer_param->set_name( InceptionSubLayerName( layer_name,  column_name,  0, "pooling"));
		cur_bottom_name = InceptionSubBlobName( layer_name,  column_name,  0, "pooling");
		added_layer_param->add_top(cur_bottom_name);
		added_layer_param->set_type("Pooling");
		added_layer_param->mutable_pooling_param()->CopyFrom(inception_column.pooling_param());
		int kernel_h_, kernel_w_;
		if (added_layer_param->mutable_pooling_param()->has_kernel_size()) {
			kernel_h_ = kernel_w_ = added_layer_param->mutable_pooling_param()->kernel_size();
		} else {
			kernel_h_ = added_layer_param->mutable_pooling_param()->kernel_h();
			kernel_w_ = added_layer_param->mutable_pooling_param()->kernel_w();
		}
		CHECK_EQ((kernel_h_-1)%2, 0);
		CHECK_EQ((kernel_w_-1)%2, 0);
		added_layer_param->mutable_pooling_param()->set_pad_h((kernel_h_-1)/2);
		added_layer_param->mutable_pooling_param()->set_pad_w((kernel_w_-1)/2);
	}


	/**
	 * set up conv and relu(if necessary)
	 */
	vector<ConvolutionParameter> conv_layers_params;
	conv_layers_params.clear();
	std::copy(inception_column.convolution_param().begin(),
			inception_column.convolution_param().end(),
			std::back_inserter(conv_layers_params));

	vector<ParamSpec> blob_params;
	blob_params.clear();
	std::copy(layer_param.param().begin(), layer_param.param().end(),
			std::back_inserter(blob_params));

	for(int i=0; i < conv_layers_params.size(); ++i )
	{
		conv_layers_params[i].mutable_weight_filler()->CopyFrom(inception_param.weight_filler());
		conv_layers_params[i].mutable_bias_filler()->CopyFrom(inception_param.bias_filler());
		LayerParameter* added_layer_param = param_split->add_layer();
		cur_bottom_name = ConfigureInceptionConvLayer(  layer_name,  column_name,
		    i, cur_bottom_name, blob_params, added_layer_param,conv_layers_params[i]);
		if(need_relu && i < conv_layers_params.size() -1){
			added_layer_param = param_split->add_layer();
			added_layer_param->Clear();
			added_layer_param->add_bottom(cur_bottom_name);
			added_layer_param->add_top(cur_bottom_name);
			added_layer_param->set_name( InceptionSubLayerName( layer_name,  column_name,  i , "relu"));
			added_layer_param->set_type("ReLU");
			if(inception_param.has_relu_param()){
				added_layer_param->mutable_relu_param()->CopyFrom(inception_param.relu_param());
			}
		}
	}
	return cur_bottom_name;
}



string InceptionSubLayerName(const string& layer_name, const string& column_name,
    const int blob_idx,string postfix )
{
	ostringstream sub_layer_name;
	if(blob_idx >= 0){
		sub_layer_name << layer_name << "_" << column_name << "_" << blob_idx+1
				<< "_"<< postfix;
	}
	else{
		sub_layer_name << layer_name << "_" << column_name << "_"
						<< "_"<< postfix;
	}
	return sub_layer_name.str();
}

string InceptionSubBlobName(const string& layer_name, const string& column_name,
	    const int blob_idx,string postfix )
{
	return InceptionSubLayerName(layer_name,column_name,blob_idx,postfix);
}

string ConfigureInceptionConvLayer(const string& layer_name, const string& column_name,
    const int blob_idx,   const string& bottom_name,const vector<ParamSpec>& blob_params,
    LayerParameter* conv_layer_param,const ConvolutionParameter& conv_param)
{
	conv_layer_param->Clear();
	conv_layer_param->add_bottom(bottom_name);
	conv_layer_param->set_name(InceptionSubLayerName(layer_name,column_name,blob_idx ,"conv"));
	string top_name = InceptionSubBlobName(layer_name,column_name,blob_idx ,"conv");
	conv_layer_param->add_top(top_name);
	conv_layer_param->set_type("Convolution");
	for(int blob_id = 0 ; blob_id < blob_params.size(); ++blob_id){
		conv_layer_param->add_param()->CopyFrom(blob_params[blob_id]);
	}
	ConvolutionParameter* cur_conv_param = conv_layer_param->mutable_convolution_param();
	cur_conv_param->CopyFrom(conv_param);

//	int kernel_h_, kernel_w_;
//	if (cur_conv_param->has_kernel_size()) {
//		kernel_h_ = kernel_w_ = cur_conv_param->kernel_size();
//	} else {
//		kernel_h_ = cur_conv_param->kernel_h();
//		kernel_w_ = cur_conv_param->kernel_w();
//	}
//	CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
//	CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
//
//	CHECK_EQ((kernel_h_-1)%2, 0);
//	CHECK_EQ((kernel_w_-1)%2, 0);
//	cur_conv_param->set_pad_h((kernel_h_-1)/2);
//	cur_conv_param->set_pad_w((kernel_w_-1)/2);


	return top_name;
}


}  // namespace caffe
