#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/maxout_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaxOutLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	int count = bottom[0]->count();
	vector<int> num_size(1, count);
	eye_.Reshape(num_size);
	caffe_set(count, Dtype(1.), eye_.mutable_cpu_data());
}

template <typename Dtype>
void MaxOutLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	int num = bottom[0]->num();
	Dtype loss = caffe_cpu_dot(count, eye_.cpu_data(), bottom[0]->cpu_data()) / num;
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MaxOutLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  if (propagate_down[0]) {
	  caffe_set(count, (Dtype)1./num, bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(MaxOutLossLayer);
#endif

INSTANTIATE_CLASS(MaxOutLossLayer);
REGISTER_LAYER_CLASS(MaxOutLoss);

}  // namespace caffe
