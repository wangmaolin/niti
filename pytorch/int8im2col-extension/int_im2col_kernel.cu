#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <im2col.cuh>
#include <im2col_shape_check.h>

static void im2col_out_cuda_int_template(
    torch::Tensor& output,
    const torch::Tensor& input_,
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef dilation,
    torch::IntArrayRef padding,
    torch::IntArrayRef stride) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  torch::TensorArg input_arg{input_, "input", 1};
  torch::TensorArg output_arg{output, "output", 2};
  torch::checkAllSameGPU("im2col_cuda", {input_arg, output_arg});

  at::native::im2col_shape_check(
      input_,
      torch::Tensor(),
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  torch::Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  output.zero_();

  // Launch kernel
  AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "im2col_out_cuda", [&] {
          torch::Tensor input_n;
          torch::Tensor output_n;

    for (int64_t elt = 0; elt < batch_size; elt++) {
      input_n = input.select(0, elt);
      output_n = output.select(0, elt);

      at::native::im2col<int8_t>(
          at::cuda::getCurrentCUDAStream(),
          input_n.data<int8_t>(),
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width,
          kernel_height,
          kernel_width,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          output_n.data<int8_t>());
    }

    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
}


torch::Tensor im2col_cuda_int(
    const torch::Tensor& input,
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef dilation,
    torch::IntArrayRef padding,
    torch::IntArrayRef stride) {
    torch::Tensor output = at::empty_like(input);
  im2col_out_cuda_int_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

