//caffe中roi_pooling的前向和反向传播
//cpu前传函数
template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();// 前向传播输入特征图(batch_size, channels, height, width)
  const Dtype* bottom_rois = bottom[1]->cpu_data();// 前向传播输入rois(num_rois, 5)
  int num_rois = bottom[1]->num();// roi的总数
  int batch_size = bottom[0]->num();// batch_size表示一次训练中输入的图片数量，因为roi并不是都存在于同一张图片上
  int top_count = top[0]->count();// 前向传播输出特征图的大小
  Dtype* top_data = top[0]->mutable_cpu_data();// 前向传播输出特征图(num_rois, channels, pooled_height, pooled_width)
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);// 初始化top_data置为最小值(-FLT_MAX)
  int* argmax_data = max_idx_.mutable_cpu_data();// 前向传播输出最大值索引(num_rois, channels, pooled_height, pooled_width)
  caffe_set(top_count, -1, argmax_data);// 初始化argmax_data置为-1
 
  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  // 遍历rois
  for (int n = 0; n < num_rois; ++n) {
    // roi信息，roi映射到特征图上的尺寸
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    // 检验roi的图片索引是否合法
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);
    // 该roi在特征图上面的宽高
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    // roi分成pooled_height×pooled_width个区域，每个区域的高宽
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);
    // 找到该roi对应的batch中的特征图
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    // 遍历每一个需要pooling的区域
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          // start (included) = floor(ph * roi_height / pooled_height_)
          // end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          // hstart，wstart，hend，wend为该区域相对于起始点(roi_start_h,roi_start_w)偏移量
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));
          // hstart，wstart，hend，wend为该区域的坐标
          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);
          // 检测区域是否为空，为空则池化后的值为0,最大值索引为-1
          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }
          // 找到该区域的最大值和最大值索引
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (batch_data[index] > top_data[pool_index]) {
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      // 继续计算下一个通道
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

//以下是roi_pooling反传的gpu版本。
template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // 第一个LOOP 第index线程负责(n,c,h,w)位置上梯度的计算
    // 特征图blob[n, c, h, w] 假设四个维度上的最大值分别是 MNXY 
    // index = ((n*N + c)*X + h)*Y + w = nNXY + cXY + hY + w  已知index可以计算出坐标
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    // 遍历rois对(n,c,h,w)位置上累积梯度
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;//得到一个roi的信息[index, x1, y1, x2, y2]
      int roi_batch_ind = offset_bottom_rois[0];//得到这个roi对应的训练batch的图片索引
      // Skip if ROI's batch index doesn't match n
      // 当前roi对应的图片索引与当前累积梯度位置不匹配
      if (n != roi_batch_ind) {
        continue;
      }
      // roi在特征图上的位置
      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      //当前roi是否包含(h, w)，不包含直接continue
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }
      //得到7×7区域的反向传播梯度和最大值索引，输出维度(num_rois,channels,pooled_height,pooled_width) 
      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;
 
      // roi在共享卷积层输出的特征图的尺寸
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      //roi pooled_height×pooled_width每个区域的宽高
      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);
      //计算特征图的(h,w)在pooled_height×pooled_width哪一个区域
      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);
 
      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);
      //该点对应的区域最大值坐标是否是该点，是的话累计梯度  ？？？为什么要用for循环，phend应该比phstart大1吧
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) 
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }//结束LOOP
}
 
template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }//如果不需反传，直接return
  const Dtype* bottom_rois = bottom[1]->gpu_data();//该层正向传播时roi信息(num_rois,5)
  const Dtype* top_diff = top[0]->gpu_diff();//反向传播的输入梯度(num_rois,channels,pooled_height,pooled_width) 
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();//反向传播的输出梯度(batch_size,channels,height,width)
  const int count = bottom[0]->count();//count对应bottom[0]的容量(即共享卷积层输出的特征的容量)
  caffe_gpu_set(count, Dtype(0.), bottom_diff);//反向传播的输出梯度全部初始化为零
  const int* argmax_data = max_idx_.gpu_data();//反向传播的输入最大值索引(num_rois,channels,pooled_height,pooled_width) 正向传播时记录
  // NOLINT_NEXT_LINE(whitespace/operators)
  //下面是具体的反传函数
  ROIPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}
 
INSTANTIATE_LAYER_GPU_FUNCS(ROIPoolingLayer);
 
}
