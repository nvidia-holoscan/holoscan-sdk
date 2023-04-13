#include "holoscan/operators/av_source/av_source.hpp"

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <thread>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace holoscan::ops {
void AVSourceOp::setup(OperatorSpec& spec) {
  auto& transmitter = spec.output<gxf::Entity>("transmitter");

  spec.param(transmitter_,
             "transmitter",
             "Entity transmitter",
             "Transmitter channel for playing files",
             &transmitter);

  spec.param(filename_, "filename", "Filename", "Filename", std::string(""));

  spec.param(
      allocator_, "allocator", "Allocator", "Allocator used to allocate render buffer output.");
}

void AVSourceOp::initialize() {
  Operator::initialize();
}

AVSourceOp::~AVSourceOp() {}

void AVSourceOp::start() {
  if (avformat_open_input(&format_ctx_, filename_.get().c_str(), nullptr, nullptr) < 0) {
    HOLOSCAN_LOG_ERROR("Failed to open input file %s; terminating.", filename_.get().c_str());
  }

  if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
    HOLOSCAN_LOG_ERROR("Failed to find stream information; terminating.");
  }

  for (unsigned int i = 0; i < format_ctx_->nb_streams; ++i) {
    if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index_ = i;
      break;
    }
  }

  if (video_stream_index_ == -1) { HOLOSCAN_LOG_ERROR("No video stream found; terminating."); }

  AVCodecParameters* codec_params = format_ctx_->streams[video_stream_index_]->codecpar;
  AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
  if (!codec) { HOLOSCAN_LOG_ERROR("No suitable decoder found; terminating."); }

  codec_ctx_ = avcodec_alloc_context3(codec);
  avcodec_parameters_to_context(codec_ctx_, codec_params);

  if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
    HOLOSCAN_LOG_ERROR("Failed to open codec; terminating.");
  }

  video_width_ = static_cast<uint32_t>(codec_ctx_->width);
  video_height_ = static_cast<uint32_t>(codec_ctx_->height);
  framerate_ = av_q2d(format_ctx_->streams[video_stream_index_]->avg_frame_rate);

  last_frame_time_ = std::chrono::system_clock::now();
}

void AVSourceOp::stop() {
  avcodec_free_context(&codec_ctx_);
  avformat_close_input(&format_ctx_);
}

void AVSourceOp::compute(InputContext& op_input, OutputContext& op_output,
                         ExecutionContext& context) {
  // avoid warning about unused variable
  (void)op_input;

  if (!playback_started_) {
    playback_started_ = true;
  } else {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame_time_);
    auto frame_duration = 1000.0 / framerate_;
    if (duration.count() < frame_duration) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds((int)(frame_duration - duration.count())));
    }
  }
  last_frame_time_ = std::chrono::system_clock::now();

  std::shared_ptr<AVFrame> frame = read_frame();

  //
  nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> video_type;
  nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(video_width_, video_height_);
  nvidia::gxf::VideoBufferInfo info{video_width_,
                                    video_height_,
                                    video_type.value,
                                    color_planes,
                                    nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};

  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                       allocator_.get()->gxf_cid());

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) {
    HOLOSCAN_LOG_ERROR("Failed to allocate output; terminating.");
    return;
  }

  auto video_buffer = output.value().add<nvidia::gxf::VideoBuffer>();
  if (!video_buffer) {
    HOLOSCAN_LOG_ERROR("Failed to allocate video buffer; terminating.");
    return;
  }

  video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      video_width_,
      video_height_,
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_BLOCK_LINEAR,
      nvidia::gxf::MemoryStorageType::kDevice,
      allocator.value());

  const size_t image_bytes = video_width_ * video_height_ * sizeof(uchar4);
  cudaMemcpy(video_buffer.value()->pointer(), frame->data[0], image_bytes, cudaMemcpyHostToDevice);

  auto result = gxf::Entity(std::move(output.value()));
  op_output.emit(result, "transmitter");
}

std::shared_ptr<AVFrame> AVSourceOp::read_frame() {
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = nullptr;
  packet.size = 0;

  while (true) {
    int response = av_read_frame(format_ctx_, &packet);
    if (response == AVERROR_EOF) {
      HOLOSCAN_LOG_INFO("End of file reached. Seeking to beginning and continuing.");
      av_seek_frame(format_ctx_, video_stream_index_, 0, AVSEEK_FLAG_FRAME);
      continue;
    }

    if (packet.stream_index == video_stream_index_) {
      if (avcodec_send_packet(codec_ctx_, &packet) < 0) {
        HOLOSCAN_LOG_ERROR("Error sending packet to decoder.");
        av_packet_unref(&packet);
        return nullptr;
      }

      std::shared_ptr<AVFrame> frame(av_frame_alloc(), [](AVFrame* f) { av_frame_free(&f); });
      response = avcodec_receive_frame(codec_ctx_, frame.get());

      if (response == 0) {
        std::shared_ptr<AVFrame> rgba_frame(av_frame_alloc(),
                                            [](AVFrame* f) { av_frame_free(&f); });

        rgba_frame->format = AV_PIX_FMT_RGBA;
        rgba_frame->width = codec_ctx_->width;
        rgba_frame->height = codec_ctx_->height;
        av_image_alloc(rgba_frame->data,
                       rgba_frame->linesize,
                       rgba_frame->width,
                       rgba_frame->height,
                       (AVPixelFormat)rgba_frame->format,
                       32);

        SwsContext* sws_ctx = sws_getContext(video_width_,
                                             video_height_,
                                             codec_ctx_->pix_fmt,
                                             video_width_,
                                             video_height_,
                                             AV_PIX_FMT_RGBA,
                                             SWS_BILINEAR,
                                             nullptr,
                                             nullptr,
                                             nullptr);

        sws_scale(sws_ctx,
                  frame->data,
                  frame->linesize,
                  0,
                  frame->height,
                  rgba_frame->data,
                  rgba_frame->linesize);

        sws_freeContext(sws_ctx);
        av_packet_unref(&packet);
        return rgba_frame;
      }
    }

    av_packet_unref(&packet);
  }

  return nullptr;
}

}  // namespace holoscan::ops