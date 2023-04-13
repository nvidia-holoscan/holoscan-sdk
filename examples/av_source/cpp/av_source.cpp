#include <holoscan/holoscan.hpp>
#include <holoscan/operators/av_source/av_source.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto source = make_operator<ops::AVSourceOp>("av", from_config("av"));
    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    // Flow definition
    add_flow(source, visualizer, {{"transmitter", "receivers"}});
  }
};

int main(int argc, char** argv) {
  App app;

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("av_source.yaml");
  app.config(config_path);

  app.run();

  return 0;
}