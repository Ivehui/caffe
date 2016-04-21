#include <vector>

#include "caffe/layers/maxout_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaxOutLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
	int num = bottom[0]->num();
	Dtype loss;
	caffe_gpu_dot(count, eye_.gpu_data(), bottom[0]->gpu_data(), &loss);
	top[0]->mutable_gpu_data()[0] = loss / num;
}

template <typename Dtype>
void MaxOutLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  if (propagate_down[0]) {
	  caffe_set(count, (Dtype)1./num, bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxOutLossLayer);

}  // namespace caffe
